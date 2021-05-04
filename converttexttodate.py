from datetime import date, timedelta
from typing import Callable, Literal, Optional

import pyarrow as pa
import pyarrow.compute
from cjwmodule.arrow.types import ArrowRenderResult
from cjwmodule.i18n import trans
from cjwmodule.types import RenderError

Unit = Literal["day", "week", "month", "quarter", "year"]

SearchDateRegex = (
    r"(?:"
    r"(?:(?P<YYYY_MM_DD_year>\d{4})-(?P<YYYY_MM_DD_month>\d\d)-(?P<YYYY_MM_DD_day>\d\d))"
    r"|"
    r"(?:(?P<YYYYMMDD_year>\d{4})(?P<YYYYMMDD_month>\d\d)(?P<YYYYMMDD_day>\d\d))"
    r")"
)
ExactDateRegex = r"\A" + SearchDateRegex + r"\z"


_WEEKDAY_OFFSETS = [timedelta(days=n) for n in range(7)]
_QUARTERS = [0, 1, 1, 1, 4, 4, 4, 7, 7, 7, 10, 10, 10]


def to_day(y: int, m: int, d: int) -> date:
    return date(y, m, d)


def to_week(y: int, m: int, d: int) -> date:
    date1 = date(y, m, d)
    return date1 - _WEEKDAY_OFFSETS[date1.weekday()]


def to_month(y: int, m: int, _: int) -> date:
    return date(y, m, 1)


def to_quarter(y: int, m: int, _: int) -> date:
    return date(y, _QUARTERS[m], 1)


def to_year(y: int, _: int, _2: int) -> date:
    return date(y, 1, 1)


class RenderErrorException(Exception):
    def __init__(self, render_error):
        super().__init__(render_error)
        self.render_error = render_error


def str_to_date(
    name: str,
    ymd_to_date: Callable[[int, int, int], date],
    value: pa.StructScalar,
) -> date:
    year = value.get("year").as_py()
    month = value.get("month").as_py()
    day = value.get("day").as_py()

    if month <= 0 or month > 12:
        raise RenderErrorException(
            RenderError(
                trans(
                    "error.invalidMonth",
                    "Invalid month in column “{column}”, value “{value}”. Month must be between 1 and 12. Please clean the text column or enable 'Convert non-dates to null'",
                    dict(column=name, value=value.get("text").as_py()),
                )
            )
        )

    try:
        return ymd_to_date(year, month, day)
    except ValueError:
        raise RenderErrorException(
            RenderError(
                trans(
                    "error.invalidDay",
                    "Invalid day in column “{column}”, value “{value}”. The day does not exist in that month. Please clean the text column or enable 'Convert non-dates to null'",
                    dict(column=name, value=value.get("text").as_py()),
                )
            )
        )


def str_to_date_or_null(
    name: str,
    ymd_to_date: Callable[[int, int, int], date],
    value: pa.StructScalar,
) -> Optional[date]:
    try:
        return str_to_date(name, ymd_to_date, value)
    except RenderErrorException:
        return None


def _convert_str_array_to_int_default_0(array: pa.Array) -> pa.Array:
    array = pa.compute.replace_substring_regex(array, pattern=r"\A\z", replacement="0")
    return array.cast(pa.int32())


def _gather_first_number(array1: pa.Array, array2: pa.Array) -> pa.Array:
    return pa.compute.add(
        _convert_str_array_to_int_default_0(array1),
        _convert_str_array_to_int_default_0(array2),
    )


def convert_array(
    *,
    name: str,
    array: pa.TimestampArray,
    unit: Unit,
    search_in_text: bool,
    error_means_null: bool
) -> pa.Date32Array:
    if pa.types.is_dictionary(array.type):
        # raises RenderErrorException
        converted_dictionary = convert_array(
            name=name,
            array=array.dictionary,
            unit=unit,
            search_in_text=search_in_text,
            error_means_null=error_means_null,
        )
        return converted_dictionary.take(array.indices)

    structs = pa.compute.extract_regex(
        array, pattern=SearchDateRegex if search_in_text else ExactDateRegex
    )

    if error_means_null:
        convert_value = str_to_date_or_null
    else:
        if structs.null_count > array.null_count:
            invalid_strings = array.filter(
                pa.compute.and_(pa.compute.is_valid(array), pa.compute.is_null(structs))
            )
            raise RenderErrorException(
                RenderError(
                    trans(
                        "error.invalidIso8601Syntax",
                        "Invalid date value in column “{column}”. Got “{value}”; expected “YYYY-MM-DD”. Please clean the text column or enable 'Convert non-dates to null'",
                        dict(column=name, value=invalid_strings[0].as_py()),
                    )
                )
            )

        convert_value = str_to_date

    ymd_to_date = {
        "day": to_day,
        "week": to_week,
        "month": to_month,
        "quarter": to_quarter,
        "year": to_year,
    }[unit]

    years = _gather_first_number(
        structs.field("YYYY_MM_DD_year"), structs.field("YYYYMMDD_year")
    )
    months = _gather_first_number(
        structs.field("YYYY_MM_DD_month"), structs.field("YYYYMMDD_month")
    )
    days = _gather_first_number(
        structs.field("YYYY_MM_DD_day"), structs.field("YYYYMMDD_day")
    )

    data = pa.StructArray.from_arrays(
        [array, years, months, days], ["text", "year", "month", "day"]
    )

    date_list = [
        convert_value(name, ymd_to_date, value) if structs[i].is_valid else None
        for i, value in enumerate(data)
    ]
    return pa.array(date_list, pa.date32())


def convert_chunked_array(
    *,
    name: str,
    chunked_array: pa.ChunkedArray,
    unit: Unit,
    search_in_text: bool,
    error_means_null: bool
) -> pa.ChunkedArray:
    chunks = [
        convert_array(
            name=name,
            array=chunk,
            unit=unit,
            search_in_text=search_in_text,
            error_means_null=error_means_null,
        )
        for chunk in chunked_array.chunks
    ]
    return pa.chunked_array(chunks, pa.date32())


def render_arrow_v1(table: pa.Table, params, **kwargs):
    search_in_text = params["search_in_text"]
    error_means_null = params["error_means_null"]
    unit = params["unit"]

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
                    search_in_text=search_in_text,
                    error_means_null=error_means_null,
                ),
            )
        except RenderErrorException as err:
            return ArrowRenderResult(pa.table({}), [err.render_error])

    return ArrowRenderResult(table)


def migrate_params(params):
    if "search_in_text" not in params:
        params = _migrate_params_v0_to_v1(params)
    return params


def _migrate_params_v0_to_v1(params):
    """v1 adds 'search_in_text=False'."""
    return dict(**params, search_in_text=False)
