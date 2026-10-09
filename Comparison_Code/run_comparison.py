#!/usr/bin/env python3
"""Run the three toy routing examples and print a verified comparison.

No result files or graphs are written unless --output is supplied. The released
LightSABRE and TKET libraries run from the fixed identity placement. The printout
shows that initial placement and the placement after every SWAP. Every route is
checked against the input circuit and its final logical-to-physical mapping.
"""

import sys

sys.dont_write_bytecode = True

import argparse
import csv
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
os.environ.setdefault("XDG_CONFIG_HOME", str(PROJECT_ROOT / ".config"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

from exact_routing import solve_exact, solve_gurobi
from validation import validate_route


def _check_examples(parser, examples):
    if not isinstance(examples, list) or not examples:
        parser.error("The examples file must contain a nonempty list of examples")
    ids = set()
    for example in examples:
        if not isinstance(example, dict):
            parser.error("Every example must be a JSON object")
        example_id = example.get("id")
        if not isinstance(example_id, str) or not example_id or Path(example_id).name != example_id:
            parser.error("Each example needs a nonempty ID without a folder path")
        if example_id in {".", ".."} or example_id in ids:
            parser.error("Example IDs must be unique and cannot be '.' or '..'")
        ids.add(example_id)
        n = example.get("n_qubits")
        gates = example.get("gates")
        if not isinstance(n, int) or isinstance(n, bool) or not isinstance(gates, list):
            parser.error("Each example needs an integer qubit count and a list of CX gates")
        if example.get("initial_layout", list(range(n))) != list(range(n)):
            parser.error("The library adapters require the identity initial layout")
        if not isinstance(example.get("edges"), list):
            parser.error(f"Example {example_id} needs a list of physical coupling edges")


def _save_results(output, rows, results):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")


def _format_placement(logical_to_physical):
    """Format which logical qubit sits on each physical qubit."""
    held = [0] * len(logical_to_physical)
    for logical, physical in enumerate(logical_to_physical):
        held[physical] = logical
    return ", ".join(f"P{physical}=Q{logical}" for physical, logical in enumerate(held))


def _print_results(rows, results, used_gurobi):
    if used_gurobi:
        print("\nExact baselines: Gurobi proved optimality; exhaustive search confirmed both objectives.")
    else:
        print("\nExact baselines: exhaustive state-space search proved optimality; Gurobi was skipped.")
    print("Extra CX = 3 per SWAP or BRIDGE. Optimal extra CX allows both operations.")
    print("\nExample          Algorithm    SWAP  BRIDGE  Extra CX  Optimal extra  Gap  Check")
    for row in rows:
        print(
            f'{row["example"]:16} {row["algorithm"]:11} '
            f'{row["swaps"]:4} {row["bridges"]:7} '
            f'{row["added_cx"]:9} {row["optimal_added_cx_swap_bridge"]:14} '
            f'{row["added_cx_gap"]:4}  PASS'
        )
    for result in results:
        example = result["example"]
        for algorithm, route in result["heuristics"].items():
            print(f'\n{example["id"]} / {algorithm}')
            n = example["n_qubits"]
            layout = list(example.get("initial_layout", range(n)))
            print(f"  Initial placement: {_format_placement(layout)}")
            swap_count = 0
            for op in route["operations"]:
                qubits = op["physical_qubits"]
                if op["kind"] == "bridge":
                    control, middle, target = qubits
                    print(
                        f"  BRIDGE: P{control} -> P{middle} -> P{target} "
                        "(placement unchanged)"
                    )
                    continue
                if op["kind"] != "swap":
                    continue
                a, b = qubits
                swap_count += 1
                print(f"  SWAP {swap_count}: P{a}<->P{b}")
                layout = [b if physical == a else a if physical == b else physical for physical in layout]
                print(f"    {_format_placement(layout)}")
            if swap_count == 0:
                print("  SWAP sequence: none")
            if layout != route["final_layout"]:
                raise AssertionError("Printed placements disagree with the validated final layout")
            mapping = ", ".join(
                f"Q{logical}->P{physical}" for logical, physical in enumerate(layout)
            )
            print(f"  Final logical-to-physical mapping: {mapping}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=ROOT / "examples.json")
    parser.add_argument("--example", default="all", help="Example ID, or all")
    parser.add_argument("--algorithm", choices=["all", "lightsabre", "tket"], default="all")
    parser.add_argument("--output", type=Path, default=None, help="Optional folder for result files and Gurobi model exports")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--lookahead", type=int, default=10)
    parser.add_argument("--time-limit", type=float, default=60)
    parser.add_argument("--no-gurobi", action="store_true", help="Use only the exact exhaustive-search baseline")
    args = parser.parse_args(argv)
    if args.trials < 1 or args.lookahead < 1:
        parser.error("trials and lookahead must be positive")
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("time-limit must be positive and finite")
    if not 0 <= args.seed < 1 << 64:
        parser.error("seed must be an integer from 0 through 2**64 - 1")
    try:
        examples = json.loads(args.examples.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"Cannot read the examples file: {exc}")
    _check_examples(parser, examples)
    if args.example != "all":
        examples = [example for example in examples if example["id"] == args.example]
    if not examples:
        parser.error("No matching example")
    output = args.output.expanduser().resolve() if args.output is not None else None
    algorithms = ["lightsabre", "tket"] if args.algorithm == "all" else [args.algorithm]
    rows, results = [], []
    for example in examples:
        n = example["n_qubits"]
        baselines, oracles = {}, {}
        for label, objective, allow_bridge in [
            ("swap_only", "swaps", False),
            ("swap_bridge", "added_cx", True),
        ]:
            options = dict(initial_layout=example.get("initial_layout"), objective=objective, allow_bridge=allow_bridge)
            try:
                oracle = solve_exact(n, example["edges"], example["gates"], **options)
            except (ValueError, TypeError) as exc:
                parser.error(f'Invalid example {example["id"]}: {exc}')
            validate_route(example, oracle)
            oracles[label] = oracle
            certified = oracle
            if not args.no_gurobi:
                prefix = output / "gurobi" / example["id"] / label if output is not None else None
                certified = solve_gurobi(
                    n, example["edges"], example["gates"], **options,
                    output_prefix=prefix, time_limit=args.time_limit,
                )
                if certified["status"] != "OPTIMAL":
                    detail = certified.get("error", certified["status"])
                    raise RuntimeError(f'Gurobi did not prove optimality for {example["id"]}/{label}: {detail}. Use --no-gurobi to run the exhaustive baseline.')
                if certified["objective_value"] != oracle["objective_value"]:
                    raise AssertionError("Gurobi and the independent exhaustive oracle disagree")
                validate_route(example, certified)
            baselines[label] = certified
        heuristic_results = {}
        for algorithm in algorithms:
            if algorithm == "lightsabre":
                from adapters_lightsabre import route_lightsabre
                route = route_lightsabre(example, seed=args.seed, trials=args.trials)
            else:
                from adapters_tket import route_tket
                route = route_tket(example, lookahead=args.lookahead)
            validate_route(example, route)
            heuristic_results[algorithm] = route
            common_optimum = baselines["swap_bridge"]["objective_value"]
            added_gap = route["added_cx"] - common_optimum
            if added_gap < 0:
                raise AssertionError("A valid heuristic route beat the certified exact optimum")
            swap_gap = route["swaps"] - baselines["swap_only"]["objective_value"] if not route["bridges"] else None
            if swap_gap is not None and swap_gap < 0:
                raise AssertionError("A SWAP-only heuristic route beat the certified SWAP optimum")
            rows.append({
                "example": example["id"], "algorithm": algorithm,
                "qubits": n, "input_cx": len(example["gates"]),
                "swaps": route["swaps"], "bridges": route["bridges"],
                "added_cx": route["added_cx"],
                "routed_cx_after_decomposition": len(example["gates"]) + route["added_cx"],
                "optimal_swaps_swap_only": baselines["swap_only"]["objective_value"],
                "swap_gap_when_no_bridges": swap_gap,
                "optimal_added_cx_swap_bridge": common_optimum,
                "added_cx_gap": added_gap,
                "added_cx_gap_percent": round(100 * added_gap / common_optimum, 2) if common_optimum else (0.0 if route["added_cx"] == 0 else None),
                "routing_seconds": route["runtime_seconds"],
                "exact_status": baselines["swap_bridge"]["status"],
                "exact_backend": "exhaustive" if args.no_gurobi else "gurobi",
                "validation": "PASS",
            })
        results.append({"example": example, "heuristics": heuristic_results, "baselines": baselines, "oracles": oracles})
    _print_results(rows, results, used_gurobi=not args.no_gurobi)
    if output is not None:
        _save_results(output, rows, results)
        print(f"\nSaved comparison.json and summary.csv to {output}")
        if not args.no_gurobi:
            print(f"Saved Gurobi models and solver artifacts to {output / 'gurobi'}")
    return rows, results


if __name__ == "__main__":
    main()
