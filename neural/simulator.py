
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix, diags


@dataclass
class SimulationResult:
    steps: np.ndarray
    mean_abs_activity: np.ndarray
    active_fraction: np.ndarray
    retina_activity: np.ndarray
    descending_activity: np.ndarray
    snapshots: dict
    elapsed_seconds: float


class FlyBrainSimulator:
    """
    Rate-based recurrent simulator using real MaleCNS connectivity.

    W[destination, source] represents the signed connection
    from a source neuron to a destination neuron.

    The original connectome weights are never modified.
    """

    def __init__(
        self,
        W: csr_matrix,
        retina: np.ndarray,
        descending: np.ndarray,
        alpha: float = 0.25,
        gain: float = 1.0,
    ):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")

        if gain < 0 or not np.isfinite(gain):
            raise ValueError("gain must be finite and nonnegative")

        if W.shape[0] != W.shape[1]:
            raise ValueError("Connectivity matrix must be square")

        self.n = W.shape[0]
        self.alpha = np.float32(alpha)
        self.gain = np.float32(gain)

        self.retina = np.asarray(retina, dtype=np.int32)
        self.descending = np.asarray(
            descending, dtype=np.int32
        )

        for indices in (self.retina, self.descending):
            if np.any(indices < 0) or np.any(indices >= self.n):
                raise ValueError("Neuron index out of bounds")

        print("Normalizing neural connections...")

        # Make our own matrix, leaving the original untouched.
        W = W.astype(np.float32, copy=True)

        # L1 normalization of incoming weights.
        # Preserves signs and ratios within each destination row.
        row_sum = np.asarray(
            np.abs(W).sum(axis=1)
        ).ravel().astype(np.float32)

        inv = np.zeros_like(row_sum)
        np.divide(
            1.0,
            row_sum,
            out=inv,
            where=row_sum > 0,
        )

        self.W = (
            diags(inv, format="csr") @ W
        ).tocsr()

        self.W.sort_indices()

        self.zero_input_neurons = int(
            np.count_nonzero(row_sum == 0)
        )

        print("Normalization complete.")
        print(f"Neurons: {self.n:,}")
        print(f"Connections: {self.W.nnz:,}")
        print(
            f"Neurons without incoming edges: "
            f"{self.zero_input_neurons:,}"
        )

    def step(
        self,
        state: np.ndarray,
        external_input: np.ndarray,
    ) -> np.ndarray:
        """
        Compute one synchronous neural update.

        h_next = (1-alpha)*h + alpha*tanh(gain*W@h + input)
        """
        if state.shape != (self.n,):
            raise ValueError("Invalid neural state shape")

        if external_input.shape != (self.n,):
            raise ValueError("Invalid input shape")

        drive = self.W @ state

        drive *= self.gain
        drive += external_input

        np.tanh(drive, out=drive)

        next_state = (
            (1.0 - self.alpha) * state
            + self.alpha * drive
        )

        return next_state.astype(np.float32, copy=False)

    def simulate(
        self,
        steps: int = 100,
        pulse_steps: int = 5,
        amplitude: float = 1.0,
        seed: int = 42,
        snapshot_steps=(10, 50, 100),
    ) -> SimulationResult:
        """
        Inject a reproducible sensory pattern.

        The input is applied only during the first pulse_steps.
        Afterward the network evolves with zero external input.
        """
        if steps < 1:
            raise ValueError("steps must be positive")

        if pulse_steps < 0:
            raise ValueError("pulse_steps cannot be negative")

        if not np.isfinite(amplitude) or amplitude < 0:
            raise ValueError("Invalid amplitude")

        rng = np.random.default_rng(seed)

        state = np.zeros(self.n, dtype=np.float32)
        zero_input = np.zeros(self.n, dtype=np.float32)
        sensory_input = np.zeros(self.n, dtype=np.float32)

        # A fixed balanced +/- pattern makes propagation observable.
        pattern = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32),
            size=len(self.retina),
        )

        sensory_input[self.retina] = (
            np.float32(amplitude) * pattern
        )

        recorded_steps = []
        mean_abs = []
        active_fraction = []
        retina_activity = []
        descending_activity = []
        snapshots = {}

        start_time = perf_counter()

        for t in range(1, steps + 1):
            current_input = (
                sensory_input
                if t <= pulse_steps
                else zero_input
            )

            state = self.step(state, current_input)

            if not np.all(np.isfinite(state)):
                raise FloatingPointError(
                    f"Nonfinite activity at step {t}"
                )

            recorded_steps.append(t)
            mean_abs.append(
                float(np.mean(np.abs(state)))
            )
            active_fraction.append(
                float(np.mean(np.abs(state) > 1e-4))
            )
            retina_activity.append(
                float(np.mean(np.abs(state[self.retina])))
                if len(self.retina) else 0.0
            )
            descending_activity.append(
                float(np.mean(
                    np.abs(state[self.descending])
                ))
                if len(self.descending) else 0.0
            )

            if t in snapshot_steps:
                snapshots[t] = state.copy()

            if t in (1, 5, 10, 25, 50, 75, 100, steps):
                print(
                    f"Step {t:3d} | "
                    f"Mean |h|: {mean_abs[-1]:.7f} | "
                    f"Active: {active_fraction[-1]*100:.2f}% | "
                    f"Descending: "
                    f"{descending_activity[-1]:.7f}"
                )

        elapsed = perf_counter() - start_time

        return SimulationResult(
            steps=np.asarray(
                recorded_steps, dtype=np.int32
            ),
            mean_abs_activity=np.asarray(
                mean_abs, dtype=np.float32
            ),
            active_fraction=np.asarray(
                active_fraction, dtype=np.float32
            ),
            retina_activity=np.asarray(
                retina_activity, dtype=np.float32
            ),
            descending_activity=np.asarray(
                descending_activity, dtype=np.float32
            ),
            snapshots=snapshots,
            elapsed_seconds=elapsed,
        )
