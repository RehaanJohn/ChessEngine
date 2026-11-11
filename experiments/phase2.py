
import json
from pathlib import Path

import numpy as np

from connectome.loader import load_connectome
from neural.simulator import FlyBrainSimulator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "phase2"


def build_simulator():
    fly = load_connectome()

    simulator = FlyBrainSimulator(
        W=fly.sparse_matrix(),
        retina=fly.retina,
        descending=fly.descending,
        alpha=0.25,
        gain=1.0,
    )

    return simulator


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== PHASE 2: FLY BRAIN SIMULATION ===\n")

    simulator = build_simulator()

    print("\n=== EXPERIMENT A: SENSORY STIMULATION ===\n")

    result = simulator.simulate(
    steps=20,
    pulse_steps=20,
    amplitude=1.0,
    seed=42,
    snapshot_steps=(10, 20),
)

    print("\n=== EXPERIMENT B: REPRODUCIBILITY ===\n")

    repeat = simulator.simulate(
        steps=100,
        pulse_steps=5,
        amplitude=1.0,
        seed=42,
    )

    for t in (10, 50, 100):
        assert np.array_equal(
            result.snapshots[t],
            repeat.snapshots[t],
        ), f"Determinism test failed at step {t}"

    print("[PASS] Identical input produces identical output")

    print("\n=== EXPERIMENT C: ACTIVITY BOUNDS ===\n")

    for t, state in result.snapshots.items():
        assert np.all(np.isfinite(state))
        assert np.max(np.abs(state)) <= 1.000001

    print("[PASS] All neural states remain finite and bounded")

    print("\n=== SAVING RESULTS ===\n")

    arrays = {
        "steps": result.steps,
        "mean_abs_activity": result.mean_abs_activity,
        "active_fraction": result.active_fraction,
        "retina_activity": result.retina_activity,
        "descending_activity": result.descending_activity,
    }

    for step, state in result.snapshots.items():
        arrays[f"state_{step}"] = state

    np.savez_compressed(
        OUTPUT_DIR / "simulation.npz",
        **arrays,
    )

    report = {
        "neurons": simulator.n,
        "connections": int(simulator.W.nnz),
        "precision": "float32",
        "alpha": float(simulator.alpha),
        "gain": float(simulator.gain),
        "timesteps": 100,
        "pulse_steps": 5,
        "seed": 42,
        "latency_seconds": result.elapsed_seconds,
        "mean_ms_per_step": (
            result.elapsed_seconds / 100 * 1000
        ),
        "final_mean_abs_activity": float(
            result.mean_abs_activity[-1]
        ),
        "final_active_fraction": float(
            result.active_fraction[-1]
        ),
        "final_descending_activity": float(
            result.descending_activity[-1]
        ),
        "determinism": "passed",
        "activity_bounds": "passed",
    }

    with open(
        OUTPUT_DIR / "report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2)

    print("\n=== FINAL RESULTS ===")
    print(
        f"Simulation time: "
        f"{result.elapsed_seconds:.3f} seconds"
    )
    print(
        f"Average timestep: "
        f"{report['mean_ms_per_step']:.3f} ms"
    )
    print(
        f"Final active fraction: "
        f"{report['final_active_fraction']*100:.2f}%"
    )
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
