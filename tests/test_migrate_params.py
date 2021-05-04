from pathlib import Path

from cjwmodule.spec.testing import param_factory

from converttexttodate import migrate_params

P = param_factory(Path(__name__).parent.parent / "converttexttodate.yaml")


def test_migrate_v0_to_v1():
    assert migrate_params(dict(colnames=["A"], error_means_null=True, unit="day")) == P(
        colnames=["A"], search_in_text=False, error_means_null=True, unit="day"
    )


def test_migrate_v1_to_v1():
    assert migrate_params(
        dict(colnames=["A"], search_in_text=True, error_means_null=False, unit="day")
    ) == P(colnames=["A"], search_in_text=True, error_means_null=False, unit="day")
