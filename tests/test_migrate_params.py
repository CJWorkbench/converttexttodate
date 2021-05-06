from pathlib import Path

from cjwmodule.spec.testing import param_factory

from converttexttodate import migrate_params

P = param_factory(Path(__name__).parent.parent / "converttexttodate.yaml")


def test_migrate_v0_to_v2():
    assert migrate_params(dict(colnames=["A"], error_means_null=True, unit="day")) == P(
        colnames=["A"], format="YYYY-MM-DD", error_means_null=True, unit="day"
    )


def test_migrate_v1_to_v2():
    assert migrate_params(
        dict(colnames=["A"], search_in_text=True, error_means_null=False, unit="day")
    ) == P(colnames=["A"], format="YYYY-MM-DD", error_means_null=False, unit="day")


def test_migrate_v2_to_v2():
    assert migrate_params(
        dict(colnames=["A"], format="M/D/YYYY", error_means_null=True, unit="day")
    ) == P(colnames=["A"], format="M/D/YYYY", error_means_null=True, unit="day")
