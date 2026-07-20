# Model-lineage holonomy data freeze

Freeze ID: `tm-mlh-v1-c9fd5ae1094b8859`

This directory locks the completed natural model-lineage holonomy experiment to
`348` exact files (`730167637` referenced bytes). Every input,
checkpoint, representation, pre-label logit bundle, transport bundle, split,
and committed result is identified by byte size and SHA-256 in
`freeze_manifest.csv`.

The freeze deliberately does not commit the large binary payloads to Git.
`git_tracked` records are stored normally; `ignored_local` and `external_input`
records remain at the paths recorded in the manifest. This is an integrity
freeze, not a remote backup of those payloads.

The scientific outcome is also frozen without reinterpretation: H1, H2, H3,
and H4 all failed their preregistered gates.

Verify the complete freeze from the repository root with:

```bash
python3 experiments/freeze_model_lineage_data.py --verify-only
```

Recreating the freeze is intentionally separate and overwrites only this
directory's generated ledgers:

```bash
python3 experiments/freeze_model_lineage_data.py --create
```
