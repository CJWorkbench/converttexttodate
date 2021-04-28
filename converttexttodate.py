from datetime import date, timedelta
from typing import Callable, Literal, Optional

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

Unit = Literal["day", "week", "month", "quarter", "year"]


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
    name: str, ymd_to_date: Callable[[int, int, int], date], s: str
) -> date:
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
        return ymd_to_date(year, month, day)
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


def str_to_date_or_null(
    name: str, ymd_to_date: Callable[[int, int, int], date], s: str
) -> Optional[date]:
    try:
        return str_to_date(name, ymd_to_date, s)
    except RenderErrorException:
        return None


def convert_array(
    *, name: str, array: pa.TimestampArray, unit: Unit, error_means_null: bool
) -> pa.Date32Array:
    if pa.types.is_dictionary(array.type):
        # raises RenderErrorException
        converted_dictionary = convert_array(
            name=name,
            array=array.dictionary,
            unit=unit,
            error_means_null=error_means_null,
        )
        return pa.DictionaryArray.from_arrays(array.indices, converted_dictionary).cast(
            pa.date32()
        )

    if error_means_null:
        convert_value = str_to_date_or_null
    else:
        convert_value = str_to_date

    ymd_to_date = {
        "day": to_day,
        "week": to_week,
        "month": to_month,
        "quarter": to_quarter,
        "year": to_year,
    }[unit]

    str_list = array.to_pylist()
    date_list = [
        None if s is None else convert_value(name, ymd_to_date, s) for s in str_list
    ]
    return pa.array(date_list, pa.date32())


def convert_chunked_array(
    *, name: str, chunked_array: pa.ChunkedArray, unit: Unit, error_means_null: bool
) -> pa.ChunkedArray:
    chunks = [
        convert_array(
            name=name, array=chunk, unit=unit, error_means_null=error_means_null
        )
        for chunk in chunked_array.chunks
    ]
    return pa.chunked_array(chunks, pa.date32())


def render_arrow_v1(table: pa.Table, params, **kwargs):
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
                    error_means_null=error_means_null,
                ),
            )
        except RenderErrorException as err:
            return ArrowRenderResult(pa.table({}), [err.render_error])

    return ArrowRenderResult(table)
