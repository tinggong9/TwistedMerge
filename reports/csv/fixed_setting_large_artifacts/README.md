# Fixed-Setting Large Artifacts

The full monomial/fixed-setting verification run produces repeated serialized
alignment maps and triangle permutations. Those columns are too large for
GitHub as inline CSV fields, so the main CSV files keep the analysis columns
and these bulky fields are stored here as gzip shards.

Use `manifest.csv` to map each compact CSV to its shard files. The manifest
records row ranges, key columns, large columns, and compression format.

The compact CSVs preserve the method metrics, deltas, capacity metadata,
claim statuses, and scale-stability quantities used by the reports.
