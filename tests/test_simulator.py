
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from neural.simulator import FlyBrainSimulator


class TestFlyBrainSimulator(unittest.TestCase):

    def setUp(self):
        # W[destination, source]
        #
        # Neuron 0 receives from neuron 1.
        # Neuron 1 receives from neurons 0 and 2.
        # Neuron 2 receives from neuron 1.
        W = csr_matrix(
            np.array(
                [
                    [0.0, 2.0, 0.0],
                    [1.0, 0.0, -3.0],
                    [0.0, 4.0, 0.0],
                ],
                dtype=np.float32,
            )
        )

        self.original = W.copy()

        self.sim = FlyBrainSimulator(
            W=W,
            retina=np.array([0], dtype=np.int32),
            descending=np.array([2], dtype=np.int32),
            alpha=0.25,
            gain=1.0,
        )

    def test_normalization(self):
        W = self.sim.W.toarray()

        expected = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.25, 0.0, -0.75],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            W, expected, rtol=1e-6, atol=1e-6
        )

    def test_original_weights_unchanged(self):
        np.testing.assert_array_equal(
            self.original.toarray(),
            np.array(
                [
                    [0.0, 2.0, 0.0],
                    [1.0, 0.0, -3.0],
                    [0.0, 4.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_zero_input_zero_state(self):
        state = np.zeros(3, dtype=np.float32)
        inputs = np.zeros(3, dtype=np.float32)

        output = self.sim.step(state, inputs)

        np.testing.assert_array_equal(
            output, np.zeros(3, dtype=np.float32)
        )

    def test_single_step(self):
        state = np.array(
            [1.0, 0.0, 0.0], dtype=np.float32
        )
        inputs = np.zeros(3, dtype=np.float32)

        output = self.sim.step(state, inputs)

        expected = (
            0.75 * state
            + 0.25 * np.tanh(self.sim.W @ state)
        )

        np.testing.assert_allclose(
            output, expected, rtol=1e-6, atol=1e-6
        )

    def test_reproducible_simulation(self):
        a = self.sim.simulate(
            steps=10, seed=42, snapshot_steps=(10,)
        )
        b = self.sim.simulate(
            steps=10, seed=42, snapshot_steps=(10,)
        )

        np.testing.assert_array_equal(
            a.snapshots[10], b.snapshots[10]
        )

    def test_states_bounded(self):
        result = self.sim.simulate(
            steps=10, seed=42, snapshot_steps=(10,)
        )

        state = result.snapshots[10]

        self.assertTrue(np.all(np.isfinite(state)))
        self.assertLessEqual(
            float(np.max(np.abs(state))), 1.000001
        )


if __name__ == "__main__":
    unittest.main()
