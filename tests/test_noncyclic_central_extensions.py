from experiments.noncyclic_central_extensions import run


def test_noncyclic_extensions_are_exact_and_not_filled_with_trivial_rows():
    extensions, ranks, claims = run()
    assert {row["group"] for row in extensions} >= {"S3", "D4", "Q8", "A4", "finite_Heisenberg_3"}
    assert all(row["candidate_status"] != "normalized_trivial" for row in extensions)
    assert all(row["normalized"] for row in extensions)
    assert all(row["cocycle_identity_error"] == 0 for row in extensions)
    assert all(row["associative"] and row["kernel_central"] and row["quotient_verified"] for row in extensions)
    assert any(row["group"] == "S3" and row["coefficient_group"] == "mu_2" and not row["coboundary"] for row in extensions)
    assert any(row["group"] == "A4" and row["coefficient_group"] == "mu_3" and not row["coboundary"] for row in extensions)
    assert all(row["value"] for row in claims)
    assert any(row["minimal_successful_rank_verified"] for row in ranks)
