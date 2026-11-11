
import unittest

import chess
import numpy as np
from scipy.sparse import csr_matrix

from chess_engine.encoder import encode_board
from chess_engine.model import FlyChessModel
from neural.simulator import FlyBrainSimulator


def make_simulator(with_connections=True):
    # Two input neurons, four internal neurons.
    W = np.zeros((6, 6), dtype=np.float32)

    if with_connections:
        W[2, 0] = 1.0
        W[3, 1] = -1.0
        W[4, 2] = 1.0
        W[5, 3] = 1.0

    return FlyBrainSimulator(
        csr_matrix(W),
        retina=np.array([0, 1], dtype=np.int32),
        descending=np.array([4, 5], dtype=np.int32),
        alpha=0.25,
        gain=1.5,
    )


class TestChessPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = FlyChessModel(
            make_simulator(),
            seed=42,
            steps=20,
            readout_size=4,
        )

    def test_encoder_shape(self):
        features = encode_board(chess.Board())

        self.assertEqual(features.shape, (1152,))
        self.assertEqual(features.dtype, np.float32)

    def test_starting_position_pieces(self):
        planes = encode_board(
            chess.Board()
        ).reshape(18, 64)

        # 16 pieces on each side.
        self.assertEqual(int(planes[:6].sum()), 16)
        self.assertEqual(int(planes[6:12].sum()), 16)

        # White king on e1.
        self.assertEqual(
            planes[5, chess.E1], 1.0
        )

        # Black king on e8.
        self.assertEqual(
            planes[11, chess.E8], 1.0
        )

    def test_legal_moves(self):
        board = chess.Board()

        moves, _ = self.model.predict(board)

        self.assertEqual(len(moves), 20)

        for prediction in moves:
            self.assertIn(
                chess.Move.from_uci(prediction.uci),
                board.legal_moves,
            )

    def test_probabilities(self):
        moves, _ = self.model.predict(chess.Board())

        total = sum(m.probability for m in moves)

        self.assertAlmostEqual(total, 1.0, places=7)
        self.assertTrue(
            all(m.probability >= 0 for m in moves)
        )

    def test_determinism(self):
        board = chess.Board()

        first, _ = self.model.predict(board)
        second, _ = self.model.predict(board)

        np.testing.assert_array_equal(
            [m.probability for m in first],
            [m.probability for m in second],
        )

    def test_neural_graph_affects_policy(self):
        # Compare a connected graph to a disconnected one.
        connected = FlyChessModel(
            make_simulator(True),
            seed=42,
            steps=20,
            readout_size=4,
        )

        disconnected = FlyChessModel(
            make_simulator(False),
            seed=42,
            steps=20,
            readout_size=4,
        )

        board = chess.Board()

        a, _ = connected.predict(board)
        b, _ = disconnected.predict(board)

        probabilities_a = {
            move.uci: move.probability for move in a
        }
        probabilities_b = {
            move.uci: move.probability for move in b
        }

        # Removing every connection forces the internal
        # readout to zero, yielding a uniform policy.
        np.testing.assert_allclose(
            list(probabilities_b.values()),
            np.full(20, 1.0 / 20),
            atol=1e-7,
        )

        # The connected network should produce a
        # non-uniform move distribution.
        self.assertGreater(
            max(probabilities_a.values())
            - min(probabilities_a.values()),
            1e-8,
        )

    def test_terminal_position(self):
        # Black is checkmated.
        board = chess.Board(
            "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1"
        )

        moves, diagnostics = self.model.predict(board)

        self.assertEqual(len(moves), 0)
        self.assertTrue(diagnostics["game_over"])


if __name__ == "__main__":
    unittest.main()
