"""Exact routing baselines for small CX-only circuits.

The hardware graph is undirected: both directions of CX are available on an
edge. There are exactly ``n`` logical and physical qubits, initial placement is
fixed (identity by default), and final placement is unrestricted. Gates sharing
a logical qubit retain their input order; disjoint gates may be reordered.

A state is (executed_gate_bitmask, logical_to_physical_permutation). A legal
transition executes a ready adjacent CX at zero cost or performs one physical
SWAP. Optional BRIDGE transitions execute a ready distance-two CX without
changing placement. A BRIDGE is four CX gates in place of one input CX, so both
a SWAP and a BRIDGE add three physical CX gates under the assumed decomposition.

``solve_exact`` uses Dijkstra's algorithm and needs only Python's standard
library. ``solve_gurobi`` constructs a binary shortest-path unit-flow IP over
the entire reachable state graph. It has no selected time/SWAP horizon. This
formulation is intended for tiny examples: the state bound is n! * 2**gates.

The defaults minimize SWAP count with BRIDGE disabled. To compare a router
that can insert BRIDGEs, use ``objective='added_cx', allow_bridge=True``.
Hardware fidelity, parallel depth, direction-reversal overhead, and placement
optimization are outside these baselines.

Gurobi API documentation:
https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


State = tuple[int, tuple[int, ...]]


@dataclass(frozen=True)
class _Problem:
    n: int
    edges: tuple[tuple[int, int], ...]
    gates: tuple[tuple[int, int], ...]
    initial_layout: tuple[int, ...]
    dependencies: tuple[int, ...]
    neighbors: tuple[frozenset[int], ...]
    objective: str
    allow_bridge: bool

    @property
    def start(self) -> State:
        return 0, self.initial_layout

    @property
    def complete_mask(self) -> int:
        return (1 << len(self.gates)) - 1

    @property
    def swap_cost(self) -> int:
        return 1 if self.objective == "swaps" else 3

    @property
    def bridge_cost(self) -> int:
        return 0 if self.objective == "swaps" else 3


def _qubit(value: object, n: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < n:
        raise ValueError(f"Qubit labels must be integers from 0 to {n - 1}: {value!r}")
    return value


def _prepare(
    n: int,
    edges: Iterable[Sequence[int]],
    gates: Iterable[Sequence[int]],
    initial_layout: Sequence[int] | None,
    objective: str,
    allow_bridge: bool,
) -> _Problem:
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if objective not in {"swaps", "added_cx"}:
        raise ValueError("objective must be 'swaps' or 'added_cx'")
    hardware_edges = set()
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"A coupling edge must contain two qubits: {edge!r}")
        a, b = (_qubit(q, n) for q in edge)
        if a == b:
            raise ValueError("Coupling edges cannot be self-loops")
        hardware_edges.add(tuple(sorted((a, b))))
    neighbors = [set() for _ in range(n)]
    for a, b in hardware_edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    reached = {0}
    frontier = [0]
    while frontier:
        for q in neighbors[frontier.pop()]:
            if q not in reached:
                reached.add(q)
                frontier.append(q)
    if len(reached) != n:
        raise ValueError("The physical coupling graph must be connected")

    input_gates = []
    for gate in gates:
        if len(gate) != 2:
            raise ValueError(f"An input CX must contain control and target: {gate!r}")
        a, b = (_qubit(q, n) for q in gate)
        if a == b:
            raise ValueError("CX control and target must differ")
        input_gates.append((a, b))
    layout = tuple(range(n)) if initial_layout is None else tuple(initial_layout)
    if len(layout) != n or set(layout) != set(range(n)):
        raise ValueError("initial_layout must be a permutation of physical qubits")
    for q in layout:
        _qubit(q, n)

    # Each gate needs the latest predecessor on EACH of its two wires. Using
    # only one globally latest overlapping gate would lose a wire dependency.
    last_on_wire: list[int | None] = [None] * n
    dependencies = []
    for i, gate in enumerate(input_gates):
        mask = 0
        for q in gate:
            previous = last_on_wire[q]
            if previous is not None:
                mask |= 1 << previous
            last_on_wire[q] = i
        dependencies.append(mask)
    return _Problem(
        n=n,
        edges=tuple(sorted(hardware_edges)),
        gates=tuple(input_gates),
        initial_layout=layout,
        dependencies=tuple(dependencies),
        neighbors=tuple(frozenset(ns) for ns in neighbors),
        objective=objective,
        allow_bridge=bool(allow_bridge),
    )


def _successors(problem: _Problem, state: State):
    mask, layout = state
    if mask == problem.complete_mask:
        return
    for i, (a, b) in enumerate(problem.gates):
        if mask & (1 << i) or problem.dependencies[i] & mask != problem.dependencies[i]:
            continue
        p, q = layout[a], layout[b]
        if q in problem.neighbors[p]:
            yield (
                (mask | (1 << i), layout),
                0,
                {"kind": "cx", "physical_qubits": [p, q],
                 "logical_qubits": [a, b], "gate_index": i},
            )
        elif problem.allow_bridge:
            for middle in sorted(problem.neighbors[p] & problem.neighbors[q]):
                yield (
                    (mask | (1 << i), layout),
                    problem.bridge_cost,
                    {"kind": "bridge", "physical_qubits": [p, middle, q],
                     "logical_qubits": [a, b], "gate_index": i},
                )
    inverse = {physical: logical for logical, physical in enumerate(layout)}
    for p, q in problem.edges:
        a, b = inverse[p], inverse[q]
        next_layout = list(layout)
        next_layout[a], next_layout[b] = q, p
        yield (
            (mask, tuple(next_layout)),
            problem.swap_cost,
            {"kind": "swap", "physical_qubits": [p, q], "logical_qubits": [a, b]},
        )


def validate_route(
    n: int,
    edges: Iterable[Sequence[int]],
    gates: Iterable[Sequence[int]],
    operations: Iterable[dict],
    initial_layout: Sequence[int] | None = None,
    *,
    allow_bridge: bool = False,
) -> dict:
    """Check adjacency, mapping updates, every gate, and both wire dependencies.

    Return route metrics. Raise ValueError for an invalid route; a successful
    validation includes completion of every input gate exactly once.
    """
    problem = _prepare(n, edges, gates, initial_layout, "added_cx", allow_bridge)
    mask, layout = problem.start
    swaps = bridges = 0
    order = []
    for step, op in enumerate(operations):
        kind = op.get("kind")
        physical = tuple(op.get("physical_qubits", []))
        for q in physical:
            _qubit(q, n)
        if kind == "swap":
            if len(physical) != 2 or tuple(sorted(physical)) not in problem.edges:
                raise ValueError(f"Step {step}: SWAP must use a physical edge")
            p, q = physical
            values = list(layout)
            a, b = values.index(p), values.index(q)
            values[a], values[b] = q, p
            layout = tuple(values)
            swaps += 1
            continue
        if kind not in {"cx", "bridge"}:
            raise ValueError(f"Step {step}: unknown operation kind {kind!r}")
        i = op.get("gate_index")
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(problem.gates):
            raise ValueError(f"Step {step}: gate_index does not identify an input CX")
        if mask & (1 << i):
            raise ValueError(f"Step {step}: gate {i} executes twice")
        if problem.dependencies[i] & mask != problem.dependencies[i]:
            raise ValueError(f"Step {step}: gate {i} violates a wire dependency")
        a, b = problem.gates[i]
        if tuple(op.get("logical_qubits", [])) != (a, b):
            raise ValueError(f"Step {step}: logical CX does not match input gate {i}")
        if kind == "cx":
            if physical != (layout[a], layout[b]) or physical[1] not in problem.neighbors[physical[0]]:
                raise ValueError(f"Step {step}: CX mapping or adjacency is invalid")
        else:
            if not allow_bridge:
                raise ValueError(f"Step {step}: BRIDGE is disabled")
            if len(physical) != 3 or len(set(physical)) != 3:
                raise ValueError(f"Step {step}: BRIDGE must use three distinct qubits")
            p, middle, q = physical
            if (p, q) != (layout[a], layout[b]) or middle not in problem.neighbors[p] or q not in problem.neighbors[middle]:
                raise ValueError(f"Step {step}: BRIDGE mapping or path is invalid")
            bridges += 1
        mask |= 1 << i
        order.append(i)
    if mask != problem.complete_mask:
        raise ValueError("The route does not execute every input CX")
    return {
        "valid": True,
        "swaps": swaps,
        "bridges": bridges,
        "added_cx": 3 * (swaps + bridges),
        "final_layout": list(layout),
        "execution_order": order,
    }


def _result(problem: _Problem, operations: list[dict], runtime: float, certificate: dict) -> dict:
    checked = validate_route(
        problem.n, problem.edges, problem.gates, operations,
        problem.initial_layout, allow_bridge=problem.allow_bridge,
    )
    value = checked["swaps"] if problem.objective == "swaps" else checked["added_cx"]
    return {
        "status": "OPTIMAL",
        "objective": problem.objective,
        "objective_value": value,
        "swaps": checked["swaps"],
        "bridges": checked["bridges"],
        "added_cx": checked["added_cx"],
        "operations": operations,
        "initial_layout": list(problem.initial_layout),
        "final_layout": checked["final_layout"],
        "runtime_seconds": runtime,
        "runtime": runtime,
        "certificate": certificate,
    }


def solve_exact(
    n: int,
    edges: Iterable[Sequence[int]],
    gates: Iterable[Sequence[int]],
    initial_layout: Sequence[int] | None = None,
    *,
    objective: str = "swaps",
    allow_bridge: bool = False,
) -> dict:
    """Return a globally optimal toy route via exact shortest-path search."""
    started = time.perf_counter()
    problem = _prepare(n, edges, gates, initial_layout, objective, allow_bridge)
    distance = {problem.start: 0}
    predecessor: dict[State, tuple[State, dict]] = {}
    heap = [(0, problem.start)]
    settled = 0
    transitions = 0
    goal = None
    optimum = None
    while heap:
        cost, state = heapq.heappop(heap)
        if distance[state] != cost:
            continue
        settled += 1
        if state[0] == problem.complete_mask:
            goal, optimum = state, cost
            break
        for successor, weight, operation in _successors(problem, state):
            transitions += 1
            candidate = cost + weight
            if candidate < distance.get(successor, math.inf):
                distance[successor] = candidate
                predecessor[successor] = state, operation
                heapq.heappush(heap, (candidate, successor))
    if goal is None:
        raise RuntimeError("A connected hardware graph should make every CX circuit routable")
    operations = []
    cursor = goal
    while cursor != problem.start:
        previous, operation = predecessor[cursor]
        operations.append(operation)
        cursor = previous
    operations.reverse()
    return _result(
        problem, operations, time.perf_counter() - started,
        {
            "method": "Dijkstra shortest path over legal finite routing states",
            "lower_bound": optimum,
            "upper_bound": optimum,
            "optimality_gap": 0,
            "settled_states": settled,
            "discovered_states": len(distance),
            "examined_transitions": transitions,
            "possible_state_bound": math.factorial(n) * (1 << len(problem.gates)),
            "arbitrary_swap_horizon": False,
            "route_verified": True,
        },
    )


def _state_graph(problem: _Problem):
    """Enumerate every reachable state, then append one shared terminal sink."""
    states: list[State | None] = [problem.start]
    ids = {problem.start: 0}
    arcs: list[tuple[int, int, int, dict]] = []
    goals = []
    cursor = 0
    while cursor < len(states):
        state = states[cursor]
        if state[0] == problem.complete_mask:
            goals.append(cursor)
        else:
            for successor, cost, operation in _successors(problem, state):
                if successor not in ids:
                    ids[successor] = len(states)
                    states.append(successor)
                arcs.append((cursor, ids[successor], cost, operation))
        cursor += 1
    sink = len(states)
    states.append(None)
    for state_id in goals:
        arcs.append((state_id, sink, 0, {"kind": "finish"}))
    incoming: list[list[int]] = [[] for _ in states]
    outgoing: list[list[int]] = [[] for _ in states]
    for i, (source, target, _, _) in enumerate(arcs):
        outgoing[source].append(i)
        incoming[target].append(i)
    return states, arcs, incoming, outgoing, sink


def _write_graph(prefix: Path, problem: _Problem, states, arcs, sink: int):
    payload = {
        "n": problem.n,
        "edges": problem.edges,
        "gates": problem.gates,
        "initial_layout": problem.initial_layout,
        "objective": problem.objective,
        "allow_bridge": problem.allow_bridge,
        "source_state": 0,
        "sink_state": sink,
        "states": [
            {"id": i, "executed_gate_mask": state[0], "layout": state[1]}
            if state is not None else {"id": i, "kind": "sink"}
            for i, state in enumerate(states)
        ],
        "arcs": [
            {"id": i, "variable": f"x[{i}]", "source": u, "target": v,
             "cost": cost, "operation": operation}
            for i, (u, v, cost, operation) in enumerate(arcs)
        ],
    }
    path = Path(f"{prefix}.states.json")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return str(path)


def _write_result(prefix: Path, result: dict):
    json_path = Path(f"{prefix}.route.json")
    csv_path = Path(f"{prefix}.route.csv")
    paths = {"route_json": str(json_path), "route_csv": str(csv_path)}
    result.setdefault("output_files", {}).update(paths)
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    with csv_path.open("w", newline="") as stream:
        fields = ["step", "kind", "gate_index", "logical_qubits", "physical_qubits"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step, operation in enumerate(result.get("operations", [])):
            writer.writerow({
                "step": step,
                "kind": operation["kind"],
                "gate_index": operation.get("gate_index", ""),
                "logical_qubits": " ".join(map(str, operation.get("logical_qubits", []))),
                "physical_qubits": " ".join(map(str, operation["physical_qubits"])),
            })


def _selected_path(arcs, outgoing, selected: set[int], sink: int) -> list[int]:
    """Extract a source-sink path even if a nonoptimal incumbent has cycles."""
    queue = deque([0])
    predecessors: dict[int, tuple[int, int] | None] = {0: None}
    while queue:
        u = queue.popleft()
        if u == sink:
            break
        for arc_id in outgoing[u]:
            if arc_id not in selected:
                continue
            v = arcs[arc_id][1]
            if v not in predecessors:
                predecessors[v] = u, arc_id
                queue.append(v)
    if sink not in predecessors:
        raise RuntimeError("The Gurobi incumbent did not contain a source-to-sink path")
    path = []
    current = sink
    while current != 0:
        previous, arc_id = predecessors[current]
        path.append(arc_id)
        current = previous
    return list(reversed(path))


def solve_gurobi(
    n: int,
    edges: Iterable[Sequence[int]],
    gates: Iterable[Sequence[int]],
    initial_layout: Sequence[int] | None = None,
    output_prefix: str | Path | None = None,
    time_limit: float = 60,
    *,
    objective: str = "swaps",
    allow_bridge: bool = False,
) -> dict:
    """Solve an exact binary unit-flow IP with Gurobi, importing it lazily.

    For each legal transition a, x[a] is binary. Minimize sum(cost[a]*x[a])
    subject to outgoing minus incoming flow = +1 at the fixed initial state,
    -1 at the terminal sink, and 0 elsewhere. An integral unit flow contains
    a valid route. Positive-cost cycles cannot improve the objective, and
    zero-cost gate transitions strictly increase the executed-gate mask.

    With output_prefix, export .lp, .mps, .log, .sol, .states.json,
    .route.json, and .route.csv. Nonoptimal statuses retain their solver bound;
    they are never presented as an optimality certificate. Missing Gurobi or
    license problems return UNAVAILABLE/ERROR with an error message.
    """
    started = time.perf_counter()
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive and finite")
    problem = _prepare(n, edges, gates, initial_layout, objective, allow_bridge)
    prefix = Path(output_prefix).expanduser().resolve() if output_prefix is not None else None
    if prefix is not None:
        prefix.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "status": "UNAVAILABLE", "objective": objective, "objective_value": None,
        "swaps": None, "bridges": None, "added_cx": None, "operations": [],
        "initial_layout": list(problem.initial_layout), "final_layout": None,
        "certificate": {}, "output_files": {},
    }
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        base.update(error=f"gurobipy is not installed in this Python environment: {exc}",
                    runtime_seconds=time.perf_counter() - started)
        base["runtime"] = base["runtime_seconds"]
        if prefix is not None:
            _write_result(prefix, base)
        return base

    states, arcs, incoming, outgoing, sink = _state_graph(problem)
    if prefix is not None:
        base["output_files"]["states_json"] = _write_graph(prefix, problem, states, arcs, sink)
    status_names = {
        1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
        5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT",
        9: "TIME_LIMIT", 10: "SOLUTION_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC",
        13: "SUBOPTIMAL", 14: "INPROGRESS", 15: "USER_OBJ_LIMIT",
        16: "WORK_LIMIT", 17: "MEM_LIMIT",
    }
    try:
        with gp.Model("exact_toy_qubit_routing") as model:
            model.Params.OutputFlag = 1 if prefix is not None else 0
            model.Params.LogToConsole = 0
            model.Params.TimeLimit = float(time_limit)
            model.Params.MIPGap = 0
            model.Params.MIPGapAbs = 0
            model.Params.Threads = 1
            model.Params.Seed = 0
            if prefix is not None:
                log = Path(f"{prefix}.log")
                model.Params.LogFile = str(log)
                base["output_files"]["log"] = str(log)
            x = model.addVars(range(len(arcs)), vtype=GRB.BINARY, name="x")
            for state_id in range(len(states)):
                supply = 1 if state_id == 0 else -1 if state_id == sink else 0
                model.addConstr(
                    gp.quicksum(x[a] for a in outgoing[state_id])
                    - gp.quicksum(x[a] for a in incoming[state_id]) == supply,
                    name=f"flow_{state_id}",
                )
            model.setObjective(gp.quicksum(cost * x[i] for i, (_, _, cost, _) in enumerate(arcs)), GRB.MINIMIZE)
            model.update()
            if prefix is not None:
                for suffix in ("lp", "mps"):
                    path = Path(f"{prefix}.{suffix}")
                    model.write(str(path))
                    base["output_files"][suffix] = str(path)
            model.optimize()
            result = dict(base)
            result["status"] = status_names.get(model.Status, f"STATUS_{model.Status}")
            result["gurobi_status_code"] = int(model.Status)
            result["gurobi_runtime_seconds"] = float(model.Runtime)
            bound = float(model.ObjBound)
            certificate = {
                "method": "binary shortest-path unit-flow IP over all reachable routing states",
                "states_in_model": len(states), "arcs_in_model": len(arcs),
                "possible_state_bound": math.factorial(n) * (1 << len(problem.gates)) + 1,
                "lower_bound": bound if math.isfinite(bound) else None,
                "upper_bound": None,
                "optimality_gap": None,
                "arbitrary_swap_horizon": False,
                "route_verified": False,
            }
            if model.SolCount > 0:
                selected = {i for i in range(len(arcs)) if x[i].X > 0.5}
                path = _selected_path(arcs, outgoing, selected, sink)
                operations = [arcs[i][3] for i in path if arcs[i][3]["kind"] != "finish"]
                verified = _result(problem, operations, time.perf_counter() - started, certificate)
                verified.update(status=result["status"], gurobi_status_code=result["gurobi_status_code"],
                                gurobi_runtime_seconds=result["gurobi_runtime_seconds"],
                                output_files=result["output_files"])
                result = verified
                result["solver_objective_value"] = float(model.ObjVal)
                certificate["upper_bound"] = result["objective_value"]
                certificate["optimality_gap"] = float(model.MIPGap)
                certificate["route_verified"] = True
                if model.Status == GRB.OPTIMAL and not math.isclose(result["objective_value"], model.ObjVal, abs_tol=1e-6):
                    raise RuntimeError("Extracted route cost does not match Gurobi's optimum")
                if prefix is not None:
                    solution = Path(f"{prefix}.sol")
                    model.write(str(solution))
                    result["output_files"]["solution"] = str(solution)
            result["certificate"] = certificate
            result["runtime_seconds"] = time.perf_counter() - started
            result["runtime"] = result["runtime_seconds"]
    except gp.GurobiError as exc:
        result = dict(base)
        result.update(status="ERROR", error=str(exc), gurobi_error_code=exc.errno,
                      runtime_seconds=time.perf_counter() - started)
        result["runtime"] = result["runtime_seconds"]
    if prefix is not None:
        _write_result(prefix, result)
    return result


def _self_test(check_gurobi: bool = False):
    cases = [
        (3, [(0, 1), (1, 2)], [(0, 1), (1, 2)], 0),
        (3, [(0, 1), (1, 2)], [(0, 2)], 1),
        (4, [(0, 1), (1, 2), (2, 3)], [(0, 3)], 2),
        (4, [(0, 1), (1, 2), (2, 3)], [(0, 3)], 0, [0, 2, 3, 1]),
        (3, [(0, 1), (1, 2)], [], 0),
    ]
    for case in cases:
        n, edges, gates, expected, *placement = case
        layout = placement[0] if placement else None
        exact = solve_exact(n, edges, gates, layout)
        assert exact["swaps"] == expected, (case, exact)
        if check_gurobi:
            result = solve_gurobi(n, edges, gates, layout)
            assert result["status"] == "OPTIMAL", result
            assert result["objective_value"] == expected, result
    n, edges, gates = 3, [(0, 1), (1, 2)], [(0, 2)]
    bridge = [{"kind": "bridge", "physical_qubits": [0, 1, 2],
               "logical_qubits": [0, 2], "gate_index": 0}]
    assert validate_route(n, edges, gates, bridge, allow_bridge=True)["added_cx"] == 3
    exact = solve_exact(n, edges, gates, objective="added_cx", allow_bridge=True)
    assert exact["objective_value"] == 3
    if check_gurobi:
        result = solve_gurobi(n, edges, gates, objective="added_cx", allow_bridge=True)
        assert result["status"] == "OPTIMAL" and result["objective_value"] == 3, result

    # Gates 0 and 1 are disjoint and may reorder. Gate 2 requires BOTH of them.
    problem = _prepare(4, [(0, 1), (1, 2), (2, 3)],
                       [(0, 1), (2, 3), (1, 2)], None, "swaps", False)
    assert problem.dependencies == (0, 0, 3)
    successors = list(_successors(problem, (2, (0, 1, 2, 3))))
    assert 2 not in [op.get("gate_index") for _, _, op in successors]
    reordered = [
        {"kind": "cx", "physical_qubits": [2, 3], "logical_qubits": [2, 3], "gate_index": 1},
        {"kind": "cx", "physical_qubits": [0, 1], "logical_qubits": [0, 1], "gate_index": 0},
        {"kind": "cx", "physical_qubits": [1, 2], "logical_qubits": [1, 2], "gate_index": 2},
    ]
    validate_route(4, problem.edges, problem.gates, reordered)
    try:
        validate_route(4, problem.edges, problem.gates, reordered[1:])
    except ValueError as exc:
        assert "dependency" in str(exc)
    else:
        raise AssertionError("A missing predecessor should have failed validation")
    print("Exact routing checks passed" + ("; Gurobi objectives agree." if check_gurobi else "."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run standard-library exact-routing checks")
    parser.add_argument("--gurobi", action="store_true", help="Also solve the checks with Gurobi")
    args = parser.parse_args()
    _self_test(check_gurobi=args.gurobi)
