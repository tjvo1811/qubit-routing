"""Run the official Qiskit 1.2.0 LightSABRE router on a fixed placement.

The heuristic is implemented by Qiskit's compiled Rust extension.  This file
only converts toy inputs to the library API and records its output.
"""

from time import perf_counter


def route_lightsabre(example, seed=7, trials=20, heuristic="decay"):
    """Route a CX-only example, beginning with logical q[i] on physical i.

    ``example`` has keys ``id``, ``n_qubits``, undirected ``edges``, and ordered
    logical CX ``gates``.  Output physical operations include explicit SWAPs;
    no layout search, synthesis, gate cancellation, or SWAP decomposition is
    performed.  The final logical-to-physical permutation need not be identity.
    """
    import qiskit
    from qiskit import QuantumCircuit, qasm2
    from qiskit.transpiler import CouplingMap, PassManager
    from qiskit.transpiler.passes import SabreSwap

    if qiskit.__version__ != "1.2.0":
        raise RuntimeError(f"This experiment pins qiskit==1.2.0; found {qiskit.__version__}.")

    n = int(example["n_qubits"])
    if n < 2:
        raise ValueError("An example must contain at least two qubits.")
    if trials < 1:
        raise ValueError("LightSABRE trials must be positive.")

    circuit = QuantumCircuit(n)
    for control, target in example["gates"]:
        circuit.cx(int(control), int(target))

    coupling = CouplingMap()
    for physical in range(n):
        coupling.add_physical_qubit(physical)
    for a, b in example["edges"]:
        coupling.add_edge(int(a), int(b))
        coupling.add_edge(int(b), int(a))

    router = SabreSwap(coupling, heuristic=heuristic, seed=seed, trials=trials)
    manager = PassManager([router])
    started = perf_counter()
    routed = manager.run(circuit)
    elapsed = perf_counter() - started

    operations = []
    swap_edges = []
    physical_to_logical = list(range(n))
    for instruction in routed.data:
        kind = instruction.operation.name
        physical = [routed.find_bit(q).index for q in instruction.qubits]
        if kind not in ("cx", "swap"):
            raise RuntimeError(f"Unexpected LightSABRE operation: {kind}")
        operations.append({"kind": kind, "physical_qubits": physical})
        if kind == "swap":
            a, b = physical
            physical_to_logical[a], physical_to_logical[b] = (
                physical_to_logical[b], physical_to_logical[a]
            )
            swap_edges.append(physical)

    final_layout = [0] * n
    for physical, logical in enumerate(physical_to_logical):
        final_layout[logical] = physical
    # Qiskit stores the routing permutation separately from the actual circuit.
    # Check our explicit SWAP replay against that independent library metadata.
    reported = manager.property_set["final_layout"]
    if reported is not None:
        library_layout = [reported[circuit.qubits[i]] for i in range(n)]
        if library_layout != final_layout:
            raise RuntimeError("SWAP replay disagrees with Qiskit's final layout.")

    return {
        "example_id": example["id"],
        "heuristic": "LightSABRE",
        "version": qiskit.__version__,
        "operations": operations,
        "final_layout": final_layout,
        "swaps": len(swap_edges),
        "swap_edges": swap_edges,
        "bridges": 0,
        "added_cx": 3 * len(swap_edges),
        "runtime_seconds": elapsed,
        "settings": {
            "initial_layout": list(range(n)),
            "seed": seed,
            "trials": trials,
            "heuristic": heuristic,
            "pass": "SabreSwap",
            "objective": "SWAP count; decay heuristic also penalizes depth",
        },
        "qasm": qasm2.dumps(routed),
        "drawing": str(routed.draw(output="text", fold=120)),
    }
