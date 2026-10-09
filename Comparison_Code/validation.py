"""Independent structural and all-basis-state checks for CX/SWAP/BRIDGE routes.

CX-only circuits are permutation matrices with no phases. Comparing all 2**n
basis states therefore proves unitary equivalence for these toy circuits,
after accounting for the reported final logical-to-physical permutation.
"""


def validate_route(example, route):
    n = example["n_qubits"]
    initial = list(example.get("initial_layout", range(n)))
    layout = initial.copy()
    edges = {tuple(sorted(edge)) for edge in example["edges"]}
    gates = [tuple(gate) for gate in example["gates"]]
    completed = set()
    swaps = bridges = 0
    for operation in route["operations"]:
        kind = operation["kind"]
        physical = operation["physical_qubits"]
        if any(p not in range(n) for p in physical) or len(set(physical)) != len(physical):
            raise AssertionError(f"Invalid operands: {operation}")
        if kind == "swap":
            if len(physical) != 2 or tuple(sorted(physical)) not in edges:
                raise AssertionError(f"Nonlocal SWAP: {operation}")
            u, v = physical
            layout = [v if p == u else u if p == v else p for p in layout]
            swaps += 1
            continue
        if kind == "cx":
            if len(physical) != 2 or tuple(sorted(physical)) not in edges:
                raise AssertionError(f"Nonlocal CX: {operation}")
            control, target = physical
        elif kind == "bridge":
            if len(physical) != 3:
                raise AssertionError(f"Invalid BRIDGE: {operation}")
            control, middle, target = physical
            if tuple(sorted((control, middle))) not in edges or tuple(sorted((middle, target))) not in edges:
                raise AssertionError(f"Nonlocal BRIDGE: {operation}")
            bridges += 1
        else:
            raise AssertionError(f"Unexpected operation: {kind}")
        logical = (layout.index(control), layout.index(target))
        candidates = []
        for index, gate in enumerate(gates):
            if index in completed or gate != logical:
                continue
            if all(j in completed for j in range(index) if set(gates[j]) & set(gate)):
                candidates.append(index)
        if not candidates:
            raise AssertionError(f"Gate missing or violates a wire dependency: {logical}")
        index = min(candidates)
        if "gate_index" in operation and operation["gate_index"] != index:
            raise AssertionError("Reported gate index disagrees with route replay")
        operation["gate_index"] = index
        operation["logical_qubits"] = list(logical)
        completed.add(index)
    if len(completed) != len(gates):
        raise AssertionError("The routed circuit does not execute every input gate")
    if route.get("final_layout", layout) != layout:
        raise AssertionError("Reported final layout disagrees with explicit SWAPs")
    if route.get("swaps", swaps) != swaps or route.get("bridges", bridges) != bridges:
        raise AssertionError("Reported operation counts disagree with replay")
    if route.get("added_cx", 3 * (swaps + bridges)) != 3 * (swaps + bridges):
        raise AssertionError("Reported added-CX cost disagrees with operation counts")
    for logical_input in range(1 << n):
        expected = [(logical_input >> q) & 1 for q in range(n)]
        physical_bits = [0] * n
        for q, p in enumerate(initial):
            physical_bits[p] = expected[q]
        for control, target in gates:
            expected[target] ^= expected[control]
        for operation in route["operations"]:
            operands = operation["physical_qubits"]
            if operation["kind"] == "swap":
                u, v = operands
                physical_bits[u], physical_bits[v] = physical_bits[v], physical_bits[u]
            elif operation["kind"] == "cx":
                control, target = operands
                physical_bits[target] ^= physical_bits[control]
            else:
                # The standard four-CX BRIDGE preserves the middle qubit.
                control, middle, target = operands
                for u, v in [(control, middle), (middle, target), (control, middle), (middle, target)]:
                    physical_bits[v] ^= physical_bits[u]
        actual = [physical_bits[p] for p in layout]
        if actual != expected:
            raise AssertionError(f"Circuit equivalence failed on basis state {logical_input}")
    route["final_layout"] = layout
    route["validation"] = {
        "adjacency": "PASS",
        "wire_dependencies": "PASS",
        "all_basis_states": "PASS",
        "basis_states_checked": 1 << n,
        "final_permutation_accounted_for": True,
    }
    return route


def operations_to_qasm(n, operations):
    """Portable OpenQASM 2: explicitly decompose BRIDGE, retain SWAP gates."""
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";']
    if any(operation["kind"] == "swap" for operation in operations):
        lines.append('gate swap a,b { cx a,b; cx b,a; cx a,b; }')
    lines.append(f'qreg q[{n}];')
    for operation in operations:
        kind = operation["kind"]
        operands = operation["physical_qubits"]
        if kind == "bridge":
            a, b, c = operands
            lines.append(f"// BRIDGE q[{a}],q[{b}],q[{c}]")
            lines.extend(f"cx q[{u}],q[{v}];" for u, v in [(a,b),(b,c),(a,b),(b,c)])
        else:
            a, b = operands
            lines.append(f"{kind} q[{a}],q[{b}];")
    return "\n".join(lines) + "\n"
