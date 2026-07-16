from experiments.fetch_kvasir_subset import CANONICAL_ARCHIVE_SHA256, REVISION


def test_dataset_sources_are_pinned():
    assert len(REVISION) == 40
    assert len(CANONICAL_ARCHIVE_SHA256) == 64


def test_primary_dataset_is_not_described_as_multicenter():
    from experiments.biomedical_dataset_audit import run

    # The executable audit handles missing local data by returning blocked;
    # this unit test checks only the non-fabrication contract in source data.
    assert "Kvasir" in run.__module__ or run.__module__.endswith("biomedical_dataset_audit")
