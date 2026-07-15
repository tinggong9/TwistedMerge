from experiments.official_baseline_integration import BASELINES


def test_official_manifest_has_canonical_repositories_and_no_internal_aliases():
    assert len(BASELINES) >= 6
    assert all(repository.startswith("https://github.com/") and repository.endswith(".git") for _, repository, _ in BASELINES)
    assert all(license_name for _, _, license_name in BASELINES)
