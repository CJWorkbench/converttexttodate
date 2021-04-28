from datetime import date
from typing import Optional

import pyarrow as pa
import pyarrow.compute
import re2
from cjwmodule.arrow.types import ArrowRenderResult
from cjwmodule.i18n import trans
from cjwmodule.types import RenderError

DateRegex = re2.compile(
    r"\A(?:"
    r"(?:(?P<YYYY_MM_DD_year>\d{4})-(?P<YYYY_MM_DD_month>\d\d)-(?P<YYYY_MM_DD_day>\d\d))"
    r"|"
    r"(?:(?P<YYYYMMDD_year>\d{4})(?P<YYYYMMDD_month>\d\d)(?P<YYYYMMDD_day>\d\d))"
    r")\z"
)


class RenderErrorException(Exception):
    def __init__(self, render_error):
        super().__init__(render_error)
        self.render_error = render_error


def str_to_date(name: str, s: str) -> date:
    m = DateRegex.fullmatch(s)
    if m is None:
        raise RenderErrorException(
            RenderError(
                trans(
                    "error.invalidIso8601Syntax",
                    "Invalid date value in column “{column}”. Got “{value}”; expected “YYYY-MM-DD”. Please clean the text column or enable 'Convert non-dates to null'",
                    dict(column=name, value=s),
                )
            )
        )

    year = int(
        m.group("YYYY_MM_DD_year")
        or m.group("YYYYMMDD_year")
        # one is always set
    )
    month = int(
        m.group("YYYY_MM_DD_month")
        or m.group("YYYYMMDD_month")
        # one is always set
    )
    day = int(
        m.group("YYYY_MM_DD_day")
        or m.group("YYYYMMDD_day")
        # one is always set
    )

    if month <= 0 or month > 12:
        raise RenderErrorException(
            RenderError(
                trans(
                    "error.invalidMonth",
                    "Invalid month in column “{column}”, value “{value}”. Month must be between 1 and 12. Please clean the text column or enable 'Convert non-dates to null'",
                    dict(column=name, value=s),
                )
            )
        )

    try:
        return date(year, month, day)
    except ValueError:
        raise RenderErrorException(
            RenderError(
                trans(
                    "error.invalidDay",
                    "Invalid day in column “{column}”, value “{value}”. The day does not exist in that month. Please clean the text column or enable 'Convert non-dates to null'",
                    dict(column=name, value=s),
                )
            )
        )


def str_to_date_or_null(name: str, s: str) -> Optional[date]:
    try:
        return str_to_date(name, s)
    except RenderErrorException:
        return None


def convert_array(
    name: str, array: pa.TimestampArray, error_means_null: bool
) -> pa.Date32Array:
    if pa.types.is_dictionary(array.type):
        # raises RenderErrorException
        converted_dictionary = convert_array(name, array.dictionary, error_means_null)
        return pa.DictionaryArray.from_arrays(array.indices, converted_dictionary).cast(
            pa.date32()
        )

    if error_means_null:
        convert_value = str_to_date_or_null
    else:
        convert_value = str_to_date

    str_list = array.to_pylist()
    date_list = [None if s is None else convert_value(name, s) for s in str_list]
    return pa.array(date_list, pa.date32())


def convert_chunked_array(
    name: str, chunked_array: pa.ChunkedArray, error_means_null: bool
) -> pa.ChunkedArray:
    chunks = [
        convert_array(name, chunk, error_means_null) for chunk in chunked_array.chunks
    ]
    return pa.chunked_array(chunks, pa.date32())


def render_arrow_v1(table: pa.Table, params, **kwargs):
    for colname in params["colnames"]:
        i = table.schema.get_field_index(colname)

        if pa.types.is_date32(table.columns[i].type):
            continue  # it's already date

        try:
            table = table.set_column(
                i,
                pa.field(colname, pa.date32(), metadata={"unit": "day"}),
                convert_chunked_array(
                    colname, table.columns[i], params["error_means_null"]
                ),
            )
        except RenderErrorException as err:
            return ArrowRenderResult(pa.table({}), [err.render_error])

    return ArrowRenderResult(table)
