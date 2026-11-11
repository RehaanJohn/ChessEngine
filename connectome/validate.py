
import json
from pathlib import Path

import numpy as np

from connectome.loader import DATA_DIR, load_connectome


EXPECTED_NEURONS = 166_700
EXPECTED_CONNECTIONS = 25_582_938


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def validate_connectome():
    print("\n=== FRUIT FLY CONNECTOME VALIDATION ===\n")

    fly = load_connectome()
    n = fly.num_neurons
    e = fly.num_connections

    check(
        n == EXPECTED_NEURONS,
        f"Neuron count: {n:,}"
    )

    check(
        e == EXPECTED_CONNECTIONS,
        f"Connection count: {e:,}"
    )

    check(
        len(fly.weights) == e,
        "Every connection has a weight"
    )

    check(
        len(fly.offsets) == n + 1,
        "CSR offsets have the correct length"
    )

    check(
        int(fly.offsets[0]) == 0,
        "First CSR offset is zero"
    )

    check(
        int(fly.offsets[-1]) == e,
        "Last CSR offset equals connection count"
    )

    check(
        bool(np.all(np.diff(fly.offsets) >= 0)),
        "CSR offsets are monotonically increasing"
    )

    check(
        bool(np.all(fly.sources >= 0))
        and bool(np.all(fly.sources < n)),
        "Every source neuron index is valid"
    )

    check(
        len(np.unique(fly.body_ids)) == n,
        "All biological neuron IDs are unique"
    )

    check(
        bool(np.all(np.isfinite(fly.weights))),
        "All connection weights are finite"
    )

    check(
        bool(np.all(fly.weights != 0)),
        "All stored connections have nonzero weights"
    )

    for name, indices in [
        ("Retina", fly.retina),
        ("Descending", fly.descending),
    ]:
        check(
            bool(np.all(indices >= 0))
            and bool(np.all(indices < n)),
            f"{name} neuron indices are valid"
        )
        check(
            len(np.unique(indices)) == len(indices),
            f"{name} population has no duplicate indices"
        )

    # Count incoming connections to every neuron.
    in_degree = np.diff(fly.offsets).astype(np.int64)

    # Count outgoing connections from every neuron.
    out_degree = np.bincount(
        fly.sources, minlength=n
    )

    print("\n=== GRAPH STATISTICS ===")
    print(f"Neurons:            {n:,}")
    print(f"Connections:        {e:,}")
    print(f"Mean in-degree:     {in_degree.mean():.2f}")
    print(f"Median in-degree:   {np.median(in_degree):.0f}")
    print(f"Max in-degree:      {in_degree.max():,}")
    print(f"Mean out-degree:    {out_degree.mean():.2f}")
    print(f"Max out-degree:     {out_degree.max():,}")

    positive = int(np.count_nonzero(fly.weights > 0))
    negative = int(np.count_nonzero(fly.weights < 0))

    print(f"Positive edges:     {positive:,}")
    print(f"Negative edges:     {negative:,}")
    print(f"Retina neurons:     {len(fly.retina):,}")
    print(f"Descending neurons: {len(fly.descending):,}")

    print("\n=== SAMPLE NEURON ===")
    neuron = 12345
    src, weights = fly.incoming(neuron)

    print(f"Internal index: {neuron}")
    print(f"Biological ID:  {fly.body_ids[neuron]}")
    print(f"Incoming edges: {len(src)}")

    for s, w in list(zip(src[:5], weights[:5])):
        print(
            f"  {int(fly.body_ids[s])} -> "
            f"{int(fly.body_ids[neuron])} "
            f"| weight={float(w):.4f}"
        )

    print("\nALL STRUCTURAL CHECKS PASSED")

    return {
        "neurons": n,
        "connections": e,
        "positive_edges": positive,
        "negative_edges": negative,
        "mean_in_degree": float(in_degree.mean()),
        "max_in_degree": int(in_degree.max()),
    }


if __name__ == "__main__":
    results = validate_connectome()

    report = DATA_DIR / "validation_report.json"
    with report.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nReport saved to: {report}")
