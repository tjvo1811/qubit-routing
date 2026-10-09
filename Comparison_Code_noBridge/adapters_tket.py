"""Run the local SWAP-only copy of TKET 2.18.4 LexiRoute.

``tket-swap-only`` is pytket 2.18.4 with ``LexiRoute::check_bridge`` changed so
it always keeps the SWAP LexiRoute already chose. The lookahead and SWAP
selection are otherwise the released algorithm. This adapter only constructs
inputs, calls that library, and records output.
"""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter


def route_tket(example: dict, lookahead: int = 10) -> dict:
    """Route a CX-only example from the fixed identity placement.

    ``example`` has ``id``, ``n_qubits``, undirected ``edges`` and ``gates``
    (ordered ``[control, target]`` pairs). The returned final_layout maps a
    logical qubit index to its physical output index. No placement, gate
    simplification, or post-routing optimization is performed.
    """
    # pytket initializes a config file on its first import. Keep that file local.
    os.environ.setdefault(
        "XDG_CONFIG_HOME", str(Path(__file__).resolve().parent.parent / ".config")
    )
    import pytket
    from pytket import Circuit, OpType
    from pytket.architecture import Architecture
    from pytket.circuit import Node, Qubit
    from pytket.mapping import LexiRouteRoutingMethod, MappingManager
    from pytket.qasm import circuit_to_qasm_str

    if pytket.__version__ != "2.18.4":
        raise RuntimeError(
            f"This experiment pins pytket==2.18.4; found {pytket.__version__}."
        )
    if getattr(pytket, "LEXIROUTE_BRIDGE", True):
        raise RuntimeError(
            "This folder must use the SWAP-only TKET copy in tket-swap-only, "
            "not the unmodified pytket package."
        )
    n = int(example["n_qubits"])
    edges = [tuple(map(int, edge)) for edge in example["edges"]]
    gates = [tuple(map(int, gate)) for gate in example["gates"]]
    if lookahead < 1:
        raise ValueError("lookahead must be a positive integer.")
    for a, b in edges + gates:
        if a == b or not (0 <= a < n and 0 <= b < n):
            raise ValueError(f"Invalid pair {(a, b)} for {n} qubits.")
    if {q for edge in edges for q in edge} != set(range(n)):
        raise ValueError("Architecture edges must include every physical qubit.")

    circuit = Circuit(n)
    for control, target in gates:
        circuit.CX(control, target)
    # Architecture uses Node IDs. All inputs are explicitly placed, avoiding
    # LexiLabellingMethod and any freedom to choose an initial mapping.
    circuit.rename_units({Qubit(i): Node(i) for i in range(n)})
    expected_units = {Node(i) for i in range(n)}
    assert set(circuit.qubits) == expected_units
    manager = MappingManager(Architecture(edges))
    started = perf_counter()
    modified = manager.route_circuit(circuit, [LexiRouteRoutingMethod(lookahead)])
    elapsed = perf_counter() - started
    if set(circuit.qubits) != expected_units:
        raise RuntimeError("Router unexpectedly changed the placed qubit set.")

    kinds = {OpType.CX: "cx", OpType.SWAP: "swap", OpType.BRIDGE: "bridge"}
    operations = []
    logical_to_physical = list(range(n))
    for command in circuit.get_commands():
        if command.op.type not in kinds:
            raise RuntimeError(f"Unexpected routed operation: {command}")
        qubits = []
        for qubit in command.qubits:
            if qubit.reg_name != "node" or len(qubit.index) != 1:
                raise RuntimeError(f"Unexpected placed unit: {qubit}")
            qubits.append(int(qubit.index[0]))
        kind = kinds[command.op.type]
        operations.append({"kind": kind, "physical_qubits": qubits})
        if kind == "swap":
            a, b = qubits
            logical_to_physical = [
                b if physical == a else a if physical == b else physical
                for physical in logical_to_physical
            ]

    swaps = sum(op["kind"] == "swap" for op in operations)
    bridges = sum(op["kind"] == "bridge" for op in operations)
    routed_cx = sum(op["kind"] == "cx" for op in operations)
    if routed_cx + bridges != len(gates):
        raise RuntimeError("Routing did not preserve the original CX count.")
    if bridges:
        raise RuntimeError("SWAP-only TKET inserted a BRIDGE.")
    return {
        "algorithm": "tket",
        "example_id": example["id"],
        "operations": operations,
        "initial_layout": list(range(n)),
        "final_layout": logical_to_physical,
        "swaps": swaps,
        "bridges": bridges,
        "added_cx": 3 * (swaps + bridges),
        "cx_after_decomposition": routed_cx + 3 * swaps + 4 * bridges,
        "runtime_seconds": elapsed,
        "version": pytket.__version__,
        "settings": {
            "method": "MappingManager + LexiRouteRoutingMethod",
            "lookahead": lookahead,
            "initial_placement": "fixed identity",
            "connectivity": "undirected",
            "placement_pass": None,
            "optimization_passes": [],
            "routing_modified_circuit": modified,
            "bridge_policy": "LexiRoute SWAP selection unchanged; BRIDGE substitution removed",
            "cost": "3 CX per SWAP; BRIDGE replaces 1 CX with 4 CX",
        },
        # QASM describes the physical commands. Read logical outputs through
        # final_layout; no restoration SWAPs are charged or appended.
        "qasm": circuit_to_qasm_str(circuit),
        "native_circuit": circuit.to_dict(),
    }
