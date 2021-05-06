from datetime import date
from pathlib import Path

import pyarrow as pa
from cjwmodule.arrow.testing import assert_result_equals, make_column, make_table
from cjwmodule.arrow.types import ArrowRenderResult
from cjwmodule.spec.testing import param_factory
from cjwmodule.testing.i18n import i18n_message
from cjwmodule.types import RenderError

from converttexttodate import render_arrow_v1 as render

P = param_factory(Path(__name__).parent.parent / "converttexttodate.yaml")


def test_no_columns_no_op():
    table = make_table(make_column("A", [1]))
    assert_result_equals(render(table, P(colnames=[])), ArrowRenderResult(table))


def test_convert_date_to_date():
    table = make_table(make_column("A", [date(2021, 4, 26)], unit="week"))
    assert_result_equals(render(table, P(colnames=["A"])), ArrowRenderResult(table))


def test_convert_yyyy_mm_dd():
    assert_result_equals(
        render(make_table(make_column("A", ["2021-04-28"])), P(colnames=["A"])),
        ArrowRenderResult(
            make_table(make_column("A", [date(2021, 4, 28)], unit="day"))
        ),
    )


def test_convert_yyyymmdd():
    assert_result_equals(
        render(
            make_table(make_column("A", ["20210428"])),
            P(colnames=["A"], format="YYYYMMDD"),
        ),
        ArrowRenderResult(
            make_table(make_column("A", [date(2021, 4, 28)], unit="day"))
        ),
    )


def test_convert_m_d_yyyy():
    assert_result_equals(
        render(
            make_table(make_column("A", ["4/28/2021"])),
            P(colnames=["A"], format="M/D/YYYY"),
        ),
        ArrowRenderResult(
            make_table(make_column("A", [date(2021, 4, 28)], unit="day"))
        ),
    )


def test_convert_d_m_yyyy():
    assert_result_equals(
        render(
            make_table(make_column("A", ["28/4/2021"])),
            P(colnames=["A"], format="D/M/YYYY"),
        ),
        ArrowRenderResult(
            make_table(make_column("A", [date(2021, 4, 28)], unit="day"))
        ),
    )


def test_convert_m_d_yy():
    assert_result_equals(
        render(
            make_table(make_column("A", ["1/2/3", "1/2/80"])),
            P(colnames=["A"], format="M/D/YY"),
        ),
        ArrowRenderResult(
            make_table(
                make_column("A", [date(2003, 1, 2), date(1980, 1, 2)], unit="day")
            )
        ),
    )


def test_convert_d_m_yy():
    assert_result_equals(
        render(
            make_table(make_column("A", ["1/2/3", "1/2/70"])),
            P(colnames=["A"], format="D/M/YY"),
        ),
        ArrowRenderResult(
            make_table(
                make_column("A", [date(2003, 2, 1), date(1970, 2, 1)], unit="day")
            )
        ),
    )


def test_convert_regex_mismatch():
    assert_result_equals(
        render(make_table(make_column("A", ["2021-04-28 "])), P(colnames=["A"])),
        ArrowRenderResult(
            make_table(),
            [
                RenderError(
                    i18n_message(
                        "error.formatMismatch",
                        dict(column="A", value="2021-04-28 "),
                    )
                )
            ],
        ),
    )


def test_convert_regex_mismatch_to_null():
    assert_result_equals(
        render(
            make_table(make_column("A", ["2021-04-28 "])),
            P(colnames=["A"], error_means_null=True),
        ),
        ArrowRenderResult(
            make_table(make_column("A", [None], pa.date32(), unit="day")),
        ),
    )


def test_convert_invalid_month():
    assert_result_equals(
        render(make_table(make_column("A", ["2021-14-28"])), P(colnames=["A"])),
        ArrowRenderResult(
            make_table(),
            [
                RenderError(
                    i18n_message(
                        "error.invalidDate",
                        dict(column="A", value="2021-14-28"),
                    )
                )
            ],
        ),
    )


def test_convert_invalid_day():
    assert_result_equals(
        render(make_table(make_column("A", ["2021-02-29"])), P(colnames=["A"])),
        ArrowRenderResult(
            make_table(),
            [
                RenderError(
                    i18n_message(
                        "error.invalidDate",
                        dict(column="A", value="2021-02-29"),
                    )
                )
            ],
        ),
    )


def test_convert_invalid_date_to_null():
    assert_result_equals(
        render(
            make_table(make_column("A", ["2021-02-29", None, "2021-02-28"])),
            P(colnames=["A"], error_means_null=True),
        ),
        ArrowRenderResult(
            make_table(
                make_column(
                    "A", [None, None, date(2021, 2, 28)], pa.date32(), unit="day"
                )
            )
        ),
    )


def test_convert_null_to_date():
    assert_result_equals(
        render(make_table(make_column("A", [None], pa.utf8())), P(colnames=["A"])),
        ArrowRenderResult(
            make_table(make_column("A", [None], pa.date32(), unit="day"))
        ),
    )


def test_convert_dictionary_to_date():
    assert_result_equals(
        render(
            make_table(make_column("A", ["2021-04-28"], dictionary=True)),
            P(colnames=["A"]),
        ),
        ArrowRenderResult(
            make_table(make_column("A", [date(2021, 4, 28)], unit="day"))
        ),
    )


def test_convert_to_week():
    assert_result_equals(
        render(
            make_table(
                make_column(
                    "A",
                    # Include pre-epoch date -- negatives are weird
                    [
                        "2021-04-26",
                        "2021-05-02",
                        "2021-05-03",
                        "1966-03-07",
                        "1966-03-13",
                        None,
                    ],
                    dictionary=True,
                )
            ),
            P(colnames=["A"], unit="week"),
        ),
        ArrowRenderResult(
            make_table(
                make_column(
                    "A",
                    [
                        date(2021, 4, 26),
                        date(2021, 4, 26),
                        date(2021, 5, 3),
                        date(1966, 3, 7),
                        date(1966, 3, 7),
                        None,
                    ],
                    unit="week",
                )
            )
        ),
    )


def test_convert_to_month():
    assert_result_equals(
        render(
            make_table(make_column("A", ["2021-04-28"])),
            P(colnames=["A"], unit="month"),
        ),
        ArrowRenderResult(
            make_table(make_column("A", [date(2021, 4, 1)], unit="month"))
        ),
    )


def test_convert_to_quarter():
    assert_result_equals(
        render(
            make_table(make_column("A", ["2021-04-28"])),
            P(colnames=["A"], unit="quarter"),
        ),
        ArrowRenderResult(
            make_table(make_column("A", [date(2021, 4, 1)], unit="quarter"))
        ),
    )


def test_convert_to_year():
    assert_result_equals(
        render(
            make_table(make_column("A", ["2021-04-28"])),
            P(colnames=["A"], unit="year"),
        ),
        ArrowRenderResult(
            make_table(make_column("A", [date(2021, 1, 1)], unit="year"))
        ),
    )
