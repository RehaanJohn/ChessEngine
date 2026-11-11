
from time import perf_counter

import numpy as np

from connectome.loader import load_connectome
from neural.simulator import FlyBrainSimulator


def cosine_similarity(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator < 1e-20:
        return float("nan")

    return float(np.dot(a, b) / denominator)


def measure(name, a, b):
    difference = a - b

    rms = float(np.sqrt(
        np.mean(difference.astype(np.float64) ** 2)
    ))

    reference_rms = float(np.sqrt(
        0.5 * (
            np.mean(a.astype(np.float64) ** 2)
            + np.mean(b.astype(np.float64) ** 2)
        )
    ))

    relative_difference = (
        rms / reference_rms
        if reference_rms > 1e-12
        else float("nan")
    )

    print(f"\n{name}")
    print(f"  RMS difference: {rms:.10f}")
    print(f"  Relative difference: {relative_difference:.6f}")
    print(f"  Cosine similarity: {cosine_similarity(a, b):.6f}")
    print(f"  Mean |A|: {np.mean(np.abs(a)):.10f}")
    print(f"  Mean |B|: {np.mean(np.abs(b)):.10f}")


def propagate(sim, pattern, steps=20):
    state = np.zeros(sim.n, dtype=np.float32)
    external = np.zeros(sim.n, dtype=np.float32)

    external[sim.retina] = pattern

    for _ in range(steps):
        state = sim.step(state, external)

    return state


def main():
    fly = load_connectome()

    sim = FlyBrainSimulator(
        W=fly.sparse_matrix(),
        retina=fly.retina,
        descending=fly.descending,
        alpha=0.25,
        gain=1.5,
    )

    rng = np.random.default_rng(42)

    shape = (len(sim.retina),)

    pattern_a = rng.choice(
        [-1.0, 1.0], size=shape
    ).astype(np.float32)

    pattern_b = rng.choice(
        [-1.0, 1.0], size=shape
    ).astype(np.float32)

    print("\n=== INPUT SEPARATION EXPERIMENT ===")
    print("Gain: 1.5")
    print("Timesteps: 20")

    start = perf_counter()

    output_a = propagate(sim, pattern_a)
    output_b = propagate(sim, pattern_b)
    output_a_repeat = propagate(sim, pattern_a)

    print(f"\nElapsed: {perf_counter()-start:.3f}s")

    assert np.array_equal(output_a, output_a_repeat)
    print("[PASS] Reproducibility")

    # Exclude sensory neurons to measure propagated activity.
    non_sensory = np.ones(sim.n, dtype=bool)
    non_sensory[sim.retina] = False

    measure(
        "Non-sensory neurons",
        output_a[non_sensory],
        output_b[non_sensory],
    )

    measure(
        "Descending neurons",
        output_a[sim.descending],
        output_b[sim.descending],
    )

    # A zero-input control.
    zero = np.zeros_like(pattern_a)
    output_zero = propagate(sim, zero)

    assert np.array_equal(
        output_zero,
        np.zeros(sim.n, dtype=np.float32)
    )
    print("\n[PASS] Zero-input control")


if __name__ == "__main__":
    main()
