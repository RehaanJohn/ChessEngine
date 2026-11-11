
from dataclasses import dataclass

import chess
import numpy as np
from scipy.sparse import csr_matrix

from chess_engine.encoder import encode_board


@dataclass
class MovePrediction:
    uci: str
    san: str
    probability: float
    logit: float


class FlyChessModel:
    """
    Deterministic, initially untrained chess interface
    for a fixed MaleCNS recurrent neural network.
    """

    def __init__(
        self,
        simulator,
        seed=42,
        steps=20,
        readout_size=8192,
        input_scale=4.0,
    ):
        self.sim = simulator
        self.steps = int(steps)
        self.seed = int(seed)
        self.input_scale = np.float32(input_scale)

        if self.steps < 1:
            raise ValueError("steps must be positive")

        rng = np.random.default_rng(seed)

        self.feature_count = 18 * 64
        retina_count = len(self.sim.retina)

        # Sparse, fixed projection from 1152 chess
        # features to the 3335 retinal neurons.
        #
        # Each retinal neuron samples 32 input features.
        connections_per_neuron = 32

        rows = np.repeat(
            np.arange(retina_count, dtype=np.int32),
            connections_per_neuron,
        )

        cols = rng.integers(
            0,
            self.feature_count,
            size=len(rows),
            dtype=np.int32,
        )

        signs = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32),
            size=len(rows),
        )

        values = (
            signs / np.sqrt(connections_per_neuron)
        ).astype(np.float32)

        self.sensory_projection = csr_matrix(
            (values, (rows, cols)),
            shape=(retina_count, self.feature_count),
            dtype=np.float32,
        )

        # Exclude retinal neurons from the policy readout.
        mask = np.ones(self.sim.n, dtype=bool)
        mask[self.sim.retina] = False

        candidates = np.flatnonzero(mask)

        self.readout_indices = np.sort(
            rng.choice(
                candidates,
                size=min(readout_size, len(candidates)),
                replace=False,
            )
        )

        # Fixed random readout matrices.
        # Training will replace these with learned weights.
        dimension = len(self.readout_indices)
        scale = np.float32(1.0 / np.sqrt(dimension))

        self.source_weights = (
            rng.standard_normal((dimension, 64))
            .astype(np.float32) * scale
        )

        self.target_weights = (
            rng.standard_normal((dimension, 64))
            .astype(np.float32) * scale
        )

        self.promotion_weights = (
            rng.standard_normal((dimension, 4))
            .astype(np.float32) * scale
        )

    def encode_sensory(self, board):
        features = encode_board(board)

        retinal_input = (
            self.sensory_projection @ features
        ).astype(np.float32)

        retinal_input *= self.input_scale
        np.tanh(retinal_input, out=retinal_input)

        return retinal_input

    def forward(self, board):
        """
        Run a chess position through the fly connectome.

        Returns source-square scores, destination-square
        scores, promotion scores, and diagnostics.
        """
        retinal_input = self.encode_sensory(board)

        external = np.zeros(self.sim.n, dtype=np.float32)
        external[self.sim.retina] = retinal_input

        state = np.zeros(self.sim.n, dtype=np.float32)

        # Constant input throughout the inference period.
        for _ in range(self.steps):
            state = self.sim.step(state, external)

        features = state[self.readout_indices].copy()

        # Normalize the neural readout for numerical stability.
        # The epsilon avoids division by zero.
        rms = np.sqrt(
            np.mean(features.astype(np.float64) ** 2)
        )

        normalized = features / np.float32(max(rms, 1e-8))

        source_scores = normalized @ self.source_weights
        target_scores = normalized @ self.target_weights
        promotion_scores = normalized @ self.promotion_weights

        diagnostics = {
            "retina_mean_abs": float(
                np.mean(np.abs(state[self.sim.retina]))
            ),
            "readout_mean_abs": float(
                np.mean(np.abs(features))
            ),
            "readout_rms": float(rms),
            "descending_mean_abs": float(
                np.mean(np.abs(state[self.sim.descending]))
            ),
            "active_fraction": float(
                np.mean(np.abs(state) > 1e-4)
            ),
        }

        return (
            source_scores,
            target_scores,
            promotion_scores,
            diagnostics,
        )

    def predict(self, board):
        if not board.is_valid():
            raise ValueError("Invalid chess position")

        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return [], {
                "game_over": True,
                "result": board.result(claim_draw=True),
            }

        source, target, promotion, diagnostics = (
            self.forward(board)
        )

        promotion_map = {
            chess.QUEEN: 0,
            chess.ROOK: 1,
            chess.BISHOP: 2,
            chess.KNIGHT: 3,
        }

        logits = []

        for move in legal_moves:
            score = (
                source[move.from_square]
                + target[move.to_square]
            )

            if move.promotion is not None:
                score += promotion[
                    promotion_map[move.promotion]
                ]

            logits.append(float(score))

        logits = np.asarray(logits, dtype=np.float64)

        # Stable softmax over legal moves only.
        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities)

        predictions = []

        for move, score, probability in zip(
            legal_moves, logits, probabilities
        ):
            predictions.append(
                MovePrediction(
                    uci=move.uci(),
                    san=board.san(move),
                    probability=float(probability),
                    logit=float(score),
                )
            )

        predictions.sort(
            key=lambda prediction: prediction.probability,
            reverse=True,
        )

        diagnostics["legal_moves"] = len(legal_moves)
        diagnostics["game_over"] = False

        return predictions, diagnostics
