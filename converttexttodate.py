from typing import Literal, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute
from cjwmodule.arrow.types import ArrowRenderResult
from cjwmodule.i18n import trans
from cjwmodule.types import RenderError

Unit = Literal["day", "week", "month", "quarter", "year"]

REGEXES = {
    "YYYY-MM-DD": r"(?P<yyyy>\d+)-(?P<mm>\d\d?)-(?P<dd>\d\d?)",
    "YYYYMMDD": r"(?P<yyyy>\d\d\d\d)(?P<mm>\d\d)(?P<dd>\d\d)",
    "M/D/YYYY": r"(?P<mm>\d\d?)/(?P<dd>\d\d?)/(?P<yyyy>\d+)",
    "D/M/YYYY": r"(?P<dd>\d\d?)/(?P<mm>\d\d?)/(?P<yyyy>\d+)",
    "M/D/YY": r"(?P<mm>\d\d?)/(?P<dd>\d\d?)/(?P<yy>\d\d?)",
    "D/M/YY": r"(?P<dd>\d\d?)/(?P<mm>\d\d?)/(?P<yy>\d\d?)",
}


_EPOCH_ORDINAL = -719469


class RenderErrorException(Exception):
    def __init__(self, render_error):
        super().__init__(render_error)
        self.render_error = render_error


def i32(i: int) -> pa.Int32Scalar:
    return pa.scalar(i, pa.int32())


def i8(i: int) -> pa.Int8Scalar:
    return pa.scalar(i, pa.int8())


def _years_months_days_to_date32s(
    y: pa.Int32Array, m: pa.Int8Array, d: pa.Int8Array
) -> pa.Date32Array:
    # http://pmyers.pcug.org.au/IndexedMultipleYearCalendar/Calgo_199.PDF

    # if m > 2 then m-=3 else { m+=9, y-=1 }
    jan_or_feb = pa.compute.less(m, i8(3))
    y = pa.compute.subtract(y, jan_or_feb.cast(pa.int32()))
    m = pa.compute.add(
        m,
        # when jan_or_feb==1, +9; when jan_or_feb==0, -3
        pa.compute.add(pa.compute.multiply(jan_or_feb.cast(pa.int8()), i8(12)), i8(-3)),
    )

    # c = y/100
    c = pa.compute.divide(y, i32(100))
    # ya = y - 100*c
    ya = pa.compute.subtract(y, pa.compute.multiply(c, i32(100)))

    # j = (146097*c)/4 + ...
    j = pa.compute.divide(pa.compute.multiply(c, i32(146097)), i32(4))
    # ... (1461 * ya)/4 + ...
    j = pa.compute.add(j, pa.compute.divide(pa.compute.multiply(ya, i32(1461)), i32(4)))
    # ... (153*m + 2)/5 + ...
    j = pa.compute.add(
        j,
        pa.compute.divide(
            pa.compute.add(pa.compute.multiply(m, i32(153)), i32(2)), i32(5)
        ),
    )
    # ... d + ...
    j = pa.compute.add(j, d)
    # ... [offset]
    return pa.compute.add(j, i32(_EPOCH_ORDINAL)).cast(pa.date32())


def _date32s_to_years_months_days(
    date32s: pa.Date32Array,
) -> Tuple[pa.Int32Array, pa.Int8Array, pa.Int8Array]:
    # A date is valid iff ymd=>date=>ymd is idempotent
    # http://pmyers.pcug.org.au/IndexedMultipleYearCalendar/Calgo_199.PDF
    j = date32s.cast(pa.int32())
    # j -= [offset]
    j = pa.compute.subtract(j, i32(_EPOCH_ORDINAL))
    # y = (4L * j - 1) / 146097
    y = pa.compute.divide(
        pa.compute.subtract(pa.compute.multiply(j, i32(4)), i32(-1)), i32(146097)
    )
    # j = 4L * j - 1 - 146097*y
    j = pa.compute.subtract(
        pa.compute.subtract(pa.compute.multiply(j, i32(4)), i32(1)),
        pa.compute.multiply(y, i32(146097)),
    )
    # d = j/4L
    d = pa.compute.divide(j, i32(4))
    # j = (4*d + 3) / 1461
    j = pa.compute.divide(
        pa.compute.add(pa.compute.multiply(d, i32(4)), i32(3)), i32(1461)
    )
    # d = 4*d + 3 - 1461*j
    d = pa.compute.subtract(
        pa.compute.add(pa.compute.multiply(d, i32(4)), i32(3)),
        pa.compute.multiply(j, i32(1461)),
    )
    # d = (d + 4) / 4
    d = pa.compute.divide(pa.compute.add(d, i32(4)), i32(4))
    # m = (5*d - 3) / 153
    m = pa.compute.divide(
        pa.compute.subtract(pa.compute.multiply(d, i32(5)), i32(3)), i32(153)
    )
    # d = 5*d - 3 - 153*m
    d = pa.compute.subtract(
        pa.compute.subtract(pa.compute.multiply(d, i32(5)), i32(3)),
        pa.compute.multiply(m, i32(153)),
    )
    # d = (d + 5) / 5
    d = pa.compute.divide(pa.compute.add(d, i32(5)), i32(5))
    # y = 100L * y + j
    y = pa.compute.add(pa.compute.multiply(y, i32(100)), j)
    # if m<10 { m += 3 } else { m -= 9; y += 1 }
    oct_nov_dec = pa.compute.greater(m, 9)
    m = pa.compute.add(
        m,
        # m>=10: add -9; m<10: add +3
        pa.compute.add(
            pa.compute.multiply(oct_nov_dec.cast(pa.int8()), i8(-12)), i8(3)
        ),
    )
    y = pa.compute.add(y, oct_nov_dec.cast(pa.int32()))

    return y, m.cast(pa.int8()), d.cast(pa.int8())


def _struct_string_field_with_nulls(array: pa.StructArray, name: str) -> pa.Array:
    """Like `array.field(name)`, but nulls become null."""
    assert array.offset == 0
    field = array.field(name)
    _, value_offsets, data = field.buffers()
    return pa.StringArray.from_buffers(
        len(array), value_offsets, data, array.buffers()[0], array.null_count, 0
    )


def _extract_regex_workaround_arrow_12670(
    array: pa.StringArray, *, pattern: str
) -> pa.StructArray:
    ok = pa.compute.match_substring_regex(array, pattern=pattern)
    good = array.filter(ok)
    good_matches = pa.compute.extract_regex(good, pattern=pattern)

    # Build array that looks like [None, 1, None, 2, 3, 4, None, 5]
    # ... ok_nonnull: [False, True, False, True, True, True, False, True]
    # (not ok.fill_null(False).cast(pa.int8()) because of ARROW-12672 segfault)
    ok_nonnull = pa.compute.and_kleene(ok.is_valid(), ok)
    # ... np_ok: [0, 1, 0, 1, 1, 1, 0, 1]
    np_ok = ok_nonnull.cast(pa.int8()).to_numpy(zero_copy_only=False)
    # ... np_index: [0, 1, 1, 2, 3, 4, 4, 5]
    np_index = np.cumsum(np_ok, dtype=np.int64) - 1
    # ...index_or_null: [None, 1, None, 3, 4, 5, None, 5]
    valid = ok_nonnull.buffers()[1]
    index_or_null = pa.Array.from_buffers(
        pa.int64(), len(array), [valid, pa.py_buffer(np_index)]
    )

    return good_matches.take(index_or_null)


def _parse_unvalidated_years_months_days(
    column_name: str,
    strings: pa.StringArray,
    pattern: str,
    format: str,
    error_means_null: bool,
) -> Tuple[pa.Int32Array, pa.Int8Array, pa.Int8Array]:
    """Parse (years, months, days) from strings, using pattern.

    Raise RenderErrorException("error.formatMismatch") if the pattern does not match.

    Assume the pattern filters out years/months/days that won't fit int32/int8/int8.

    The returned years, months and days might be invalid -- e.g., month=31.
    """
    structs = _extract_regex_workaround_arrow_12670(strings, pattern=pattern)
    mismatches = pa.compute.and_(strings.is_valid(), structs.is_null())

    if mismatches.true_count:
        if not error_means_null:
            invalid_strings = strings.filter(mismatches)
            raise RenderErrorException(
                RenderError(
                    trans(
                        "error.formatMismatch",
                        "In “{column}”, the value “{value}” does not look like “{format}”. Try changing this step's parameters; or using 'Clean text' before this step; or using 'Convert text to timestamp' instead of this step.",
                        dict(
                            column=column_name,
                            value=invalid_strings[0].as_py(),
                            format=format,
                        ),
                    )
                )
            )

        matches_or_null = pa.compute.or_kleene(
            pa.compute.invert(mismatches), pa.scalar(None, pa.bool_())
        )
        structs = structs.filter(matches_or_null, null_selection_behavior="emit_null")

    if "?P<yy>" in pattern:
        yy = _struct_string_field_with_nulls(structs, "yy").cast(pa.int32())
        offset = pa.compute.add(
            pa.compute.multiply(
                pa.compute.greater(yy, i32(69)).cast(pa.int32()),
                pa.scalar(-100, pa.int32()),
            ),
            pa.scalar(2000, pa.int32()),
        )
        years = pa.compute.add(yy, offset)
    else:
        years = _struct_string_field_with_nulls(structs, "yyyy").cast(pa.int32())
    months = _struct_string_field_with_nulls(structs, "mm").cast(pa.int8())
    days = _struct_string_field_with_nulls(structs, "dd").cast(pa.int8())
    return years, months, days


def _validate_date32s(
    name: str,
    strings: pa.StringArray,
    years: pa.Int32Array,
    months: pa.Int8Array,
    days: pa.Int8Array,
    date32s: pa.Date32Array,
    error_means_null: bool,
) -> Tuple[pa.Int32Array, pa.Int8Array, pa.Int8Array, pa.Date32Array]:
    """Raise RenderErrorException("error.invalidDate") on invalid dates.

    If error_means_null, then return copies of input arrays, replacing invalid
    dates with null.
    """
    check_years, check_months, check_days = _date32s_to_years_months_days(date32s)

    valid = pa.compute.and_(
        pa.compute.and_(
            pa.compute.equal(years, check_years), pa.compute.equal(months, check_months)
        ),
        pa.compute.equal(days, check_days),
    )

    if valid.false_count:
        if not error_means_null:
            invalid_strings = strings.filter(pa.compute.invert(valid))
            raise RenderErrorException(
                RenderError(
                    trans(
                        "error.invalidDate",
                        "Invalid date “{value}” in column “{column}”: there is no such month/day in that year.",
                        dict(column=name, value=invalid_strings[0].as_py()),
                    )
                )
            )

        valid_or_null = pa.compute.or_kleene(valid, pa.scalar(None, pa.bool_()))
        years = years.filter(valid_or_null, null_selection_behavior="emit_null")
        months = months.filter(valid_or_null, null_selection_behavior="emit_null")
        days = days.filter(valid_or_null, null_selection_behavior="emit_null")
        date32s = date32s.filter(valid_or_null, null_selection_behavior="emit_null")
    return years, months, days, date32s


def convert_array(
    *,
    name: str,
    array: Union[pa.DictionaryArray, pa.StringArray],
    unit: Unit,
    pattern: str,
    format: str,
    error_means_null: bool
) -> pa.Date32Array:
    if pa.types.is_dictionary(array.type):
        # raises RenderErrorException
        converted_dictionary = convert_array(
            name=name,
            array=array.dictionary,
            unit=unit,
            pattern=pattern,
            format=format,
            error_means_null=error_means_null,
        )
        return converted_dictionary.take(array.indices)

    years, months, days = _parse_unvalidated_years_months_days(
        name, array, pattern=pattern, format=format, error_means_null=error_means_null
    )
    date32s = _years_months_days_to_date32s(years, months, days)

    years, months, days, date32s = _validate_date32s(
        name, array, years, months, days, date32s, error_means_null=error_means_null
    )

    if unit == "year":
        return _years_months_days_to_date32s(
            years,
            # Hack to build array of 1 or null
            pa.compute.not_equal(years, pa.scalar(0, pa.int8())).cast(pa.int8()),
            pa.compute.not_equal(years, pa.scalar(0, pa.int8())).cast(pa.int8()),
        )
    elif unit == "quarter":
        return _years_months_days_to_date32s(
            years,
            # 1, 2, 3 => 1; 4, 5, 6 => 4; 7, 8, 9 => 7; 10, 11, 12 => 10
            pa.array([1, 1, 1, 1, 4, 4, 4, 7, 7, 7, 10, 10, 10], pa.int8()).take(
                months
            ),
            # Hack to build array of 1 or null
            pa.compute.not_equal(years, pa.scalar(0, pa.int8())).cast(pa.int8()),
        )
    elif unit == "month":
        return _years_months_days_to_date32s(
            years,
            months,
            # Hack to build array of 1 or null
            pa.compute.not_equal(years, pa.scalar(0, pa.int8())).cast(pa.int8()),
        )
    elif unit == "week":
        # date32==0 is Thursday (1970-01-01). Offset dates so Monday is 0;
        # round; and then un-offset dates
        #
        # Negative dates (1969 and under) would round towards 0. Subtract 6
        # so they round down.
        negative_offset = pa.compute.multiply(
            pa.compute.less(date32s, pa.scalar(0, pa.date32())).cast(pa.int32()),
            i32(-6),
        )
        return pa.compute.subtract(
            pa.compute.multiply(
                pa.compute.divide(
                    pa.compute.add(
                        pa.compute.add(date32s.cast(pa.int32()), i32(3)),
                        negative_offset,
                    ),
                    i32(7),
                ),
                i32(7),
            ),
            i32(3),
        ).cast(pa.date32())
    else:
        return date32s


def convert_chunked_array(
    *,
    name: str,
    chunked_array: pa.ChunkedArray,
    unit: Unit,
    pattern: str,
    format: str,
    error_means_null: bool
) -> pa.ChunkedArray:
    chunks = [
        convert_array(
            name=name,
            array=chunk,
            unit=unit,
            pattern=pattern,
            format=format,
            error_means_null=error_means_null,
        )
        for chunk in chunked_array.chunks
    ]
    return pa.chunked_array(chunks, pa.date32())


def render_arrow_v1(table: pa.Table, params, **kwargs):
    error_means_null = params["error_means_null"]
    unit = params["unit"]
    pattern = r"\A" + REGEXES[params["format"]] + r"\z"

    for colname in params["colnames"]:
        i = table.schema.get_field_index(colname)

        if pa.types.is_date32(table.columns[i].type):
            continue  # it's already date

        try:
            table = table.set_column(
                i,
                pa.field(colname, pa.date32(), metadata={"unit": unit}),
                convert_chunked_array(
                    name=colname,
                    chunked_array=table.columns[i],
                    unit=unit,
                    pattern=pattern,
                    format=params["format"],
                    error_means_null=error_means_null,
                ),
            )
        except RenderErrorException as err:
            return ArrowRenderResult(pa.table({}), [err.render_error])

    return ArrowRenderResult(table)


def migrate_params(params):
    if "format" not in params and "search_in_text" not in params:
        params = _migrate_params_v0_to_v1(params)
    if "format" not in params:
        params = _migrate_params_v1_to_v2(params)
    return params


def _migrate_params_v0_to_v1(params):
    """v1 adds 'search_in_text=False'."""
    return dict(**params, search_in_text=False)


def _migrate_params_v1_to_v2(params):
    """v2 nixes 'search_in_text' and adds 'format'."""
    return dict(
        colnames=params["colnames"],
        format="YYYY-MM-DD",
        error_means_null=params["error_means_null"],
        unit=params["unit"],
    )
