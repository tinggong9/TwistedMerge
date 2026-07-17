from experiments.post_iclr_v2_current_evidence_audit import (
    ALLOWED_STATUSES,
    claim_rows,
    load_snapshot,
)


def test_official_and_selector_starting_point_matches_source_csvs():
    snapshot = load_snapshot()
    assert snapshot.official_git_rebasin_rows == 20
    assert snapshot.official_c2m3_rows == 20
    assert snapshot.official_ties_rows == 3
    assert snapshot.official_failures == 0
    assert abs(snapshot.selector_minus_git_rebasin - 0.014345) < 1e-12
    assert abs(snapshot.selector_minus_c2m3 - 0.008145) < 1e-12
    assert abs(snapshot.c2m3_minus_gauge - 0.00873) < 1e-12
    assert snapshot.selector_soup_choices == 19
    assert snapshot.selector_total_choices == 20


def test_negative_boundaries_remain_closed():
    snapshot = load_snapshot()
    assert snapshot.selector_minus_greedy_soup < 0
    assert not snapshot.biomedical_retransport_passed
    assert not snapshot.biomedical_specific_passed
    assert not snapshot.biomedical_multidomain_passed
    assert not snapshot.biomedical_residual_correction_passed
    assert not snapshot.biomedical_inferred_method_on_any_pareto_frontier


def test_every_claim_uses_the_frozen_status_vocabulary():
    rows = claim_rows(load_snapshot())
    assert rows
    assert {row["status"] for row in rows} <= ALLOWED_STATUSES
    assert {"supported-narrow", "descriptive", "negative", "forbidden", "pending"} <= {
        row["status"] for row in rows
    }
