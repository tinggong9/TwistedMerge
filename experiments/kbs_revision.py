#!/usr/bin/env python3
"""KBS revision verification, readiness audit and immutable reporting CLI.

Executes CPU algebra, safe local multilayer language training/evaluation, and
readiness checks. Tiny software fixtures are distinct from research pilots.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kbs_revision.protocol import check_frozen, freeze, load_config, readiness_errors
from src.kbs_revision.provenance import environment, manifest, write_once


def log(out, event):
    with (out / "RUN_LOG.md").open("a") as handle:
        handle.write(f"- {event}\n")


def write_csv(path, rows, fields):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inventory_command(out):
    from src.kbs_revision.evidence_inventory import inventory
    write_once(out / "provenance.json", environment(ROOT))
    write_once(out / "environment.lock.txt", "\n".join(f"{name}=={version}" for name, version in sorted(environment(ROOT)["packages"].items())) + "\n")
    data = inventory(ROOT)
    pinned_external = ROOT / "configs/kbs_revision/external_sources.json"
    if pinned_external.is_file():
        data["external"] = json.loads(pinned_external.read_text())
        data["external"]["replay_note"] = "Saved read-only source audit; offline inventory does not refresh moving upstream revisions."
    write_once(out / "inventory.json", data)
    log(out, "Inventory completed; model files and pickled checkpoints were not executed.")


def verify_command(out):
    from src.kbs_revision import finite_certificates, merge_baselines
    from src.kbs_revision.policies import gate_controls
    begin = time.perf_counter()
    algebra = finite_certificates.verify_all()
    parity = merge_baselines.verify_all()
    write_once(out / "algebraic_checks.json", algebra)
    write_once(out / "baseline_parity.json", parity)
    if not parity["passed"]:
        raise ValueError("primary baseline parity failed; repair before any model study")
    write_once(out / "cpu_shape_profile.json", merge_baselines.architecture_shaped_cpu_timings())
    write_once(out / "gate_controls.json", gate_controls())
    write_once(out / "gate_ablation_controls.json", finite_certificates.gate_ablation_controls())
    write_once(out / "verification_cost.json", {"phase": "controlled_algebraic", "device": "cpu",
                "wall_seconds": time.perf_counter() - begin, "gpu_seconds": 0,
                "scope": "CPU correctness and rule fixtures only; excludes preceding inventory, tests and reporting"})
    log(out, "CPU algebra and baseline parity executed. Gate-rule fixtures carry no neural task metrics.")


def pilot_command(out, config_path, *, resume=False):
    config = load_config(config_path)
    errors = readiness_errors(config, ROOT)
    if not errors:
        from src.kbs_revision.neural_runner import execute_config
        results = execute_config(config, out / "pilot" / "episodes", root=ROOT, resume=resume)
        if not (out / "pilot" / "results.json").exists():
            write_once(out / "pilot" / "results.json", results)
        elif not resume:
            raise FileExistsError("pilot results already exist")
        log(out, "Real local language pilot completed; final test was not evaluated.")
        return 0
    value = {"phase": "blocked", "config": config, "blockers": errors, "training_executed": False, "final_test_accessed": False}
    existing = out / "pilot" / "readiness.json"
    if existing.exists() and resume:
        if json.loads(existing.read_text()) != value:
            raise ValueError("readiness/configuration changed; use a fresh run")
    else:
        write_once(existing, value)
    log(out, f"Language pilot readiness failed ({len(errors)} explicit blockers); no test data accessed.")
    return 2


def flatten(value, prefix=""):
    """A factual index of scalar measured outputs; never substitute missing values."""
    result = []
    if isinstance(value, dict):
        for key, item in value.items():
            result.extend(flatten(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(flatten(item, f"{prefix}[{index}]"))
    elif isinstance(value, (float, int, bool)):
        result.append({"field": prefix, "value": value})
    return result


def report_command(out):
    if (out / "software_integration.json").is_file():
        from src.kbs_revision.neural_reporting import report
    else:
        from src.kbs_revision.reporting import report
    report(ROOT, out)
    log(out, "Reports/tables/figures exported from measured CPU and historical records only.")
    manifest(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "verify", "pilot", "freeze", "run", "report", "backend-smoke", "prepare-splits", "train-cohort"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"command": args.command, "out": str(args.out),
                          "readiness_errors": readiness_errors(load_config(args.config), ROOT) if args.config else [],
                          "writes": False}, indent=2))
        return 0
    if args.command in ("pilot", "freeze", "run", "prepare-splits", "train-cohort") and args.config is None:
        parser.error("this command requires --config")
    out = args.out.resolve()
    if args.command == "inventory" and out.exists() and any(out.iterdir()):
        # budget.yaml is intentionally created before inventory/long-run work.
        if {p.name for p in out.iterdir()} != {"budget.yaml"}:
            parser.error("inventory requires a fresh directory (only budget.yaml may exist)")
    out.mkdir(parents=True, exist_ok=True)
    log(out, "Command: " + json.dumps([sys.executable, *sys.argv]))
    if args.command == "inventory":
        inventory_command(out)
    elif args.command == "verify":
        if not (out / "budget.yaml").is_file():
            parser.error("create budget.yaml before verification")
        verify_command(out)
    elif args.command == "pilot":
        return pilot_command(out, args.config, resume=args.resume)
    elif args.command == "freeze":
        freeze(load_config(args.config), ROOT, out)
    elif args.command == "run":
        config, expected = check_frozen(args.config, out)
        errors = readiness_errors(config, ROOT)
        if errors:
            raise ValueError("run blocked: " + "; ".join(errors))
        from src.kbs_revision.neural_runner import execute_config
        execute_config(config, out, root=ROOT, resume=args.resume)
    elif args.command == "backend-smoke":
        from src.kbs_revision.neural_driver import software_smoke
        software_smoke(out, ROOT)
        log(out, "Software-only two-block backend and four-task candidate/policy panel executed; no research test evaluation.")
    elif args.command == "prepare-splits":
        from src.kbs_revision.neural_driver import prepare_local_splits
        prepare_local_splits(args.config, out / "splits", ROOT)
    elif args.command == "train-cohort":
        from src.kbs_revision.neural_driver import train_cohort
        train_cohort(args.config, out / "cohort", ROOT)
    elif args.command == "report":
        report_command(out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
