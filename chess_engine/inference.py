
import argparse
from time import perf_counter

import chess
import numpy as np

from connectome.loader import load_connectome
from neural.simulator import FlyBrainSimulator
from chess_engine.model import FlyChessModel


def main():
    parser = argparse.ArgumentParser(
        description="Fruit-Fly Connectome Chess Engine"
    )

    parser.add_argument(
        "--fen",
        type=str,
        default=chess.STARTING_FEN,
        help="Chess position in FEN notation",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of neural simulation steps",
    )

    parser.add_argument(
        "--gain",
        type=float,
        default=1.5,
        help="Recurrent neural network gain",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of moves to display",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    board = chess.Board(args.fen)

    if not board.is_valid():
        parser.error("The supplied FEN is not a valid position")

    print("\n=== FRUIT FLY CHESS ENGINE ===\n")
    print(board.unicode(borders=True))
    print(f"\nFEN: {board.fen()}")

    print("\nLoading MaleCNS connectome...")
    fly = load_connectome()

    simulator = FlyBrainSimulator(
        W=fly.sparse_matrix(),
        retina=fly.retina,
        descending=fly.descending,
        alpha=0.25,
        gain=args.gain,
    )

    model = FlyChessModel(
        simulator=simulator,
        steps=args.steps,
        seed=args.seed,
    )

    print("\nRunning neural inference...")

    start = perf_counter()
    predictions, diagnostics = model.predict(board)
    elapsed = perf_counter() - start

    if not predictions:
        print("\nNo legal moves available.")
        print(f"Game result: {board.result(claim_draw=True)}")
        return

    print("\n=== TOP LEGAL MOVES ===\n")

    for rank, move in enumerate(
        predictions[:args.top], start=1
    ):
        print(
            f"{rank:2d}. "
            f"{move.san:8s} "
            f"UCI={move.uci:5s} "
            f"P={move.probability * 100:7.3f}%"
        )

    print("\n=== NEURAL DIAGNOSTICS ===")
    for key, value in diagnostics.items():
        print(f"{key}: {value}")

    print(f"\nInference latency: {elapsed:.3f}s")

    total_probability = sum(
        prediction.probability
        for prediction in predictions
    )

    assert np.isclose(total_probability, 1.0)
    assert all(
        chess.Move.from_uci(prediction.uci)
        in board.legal_moves
        for prediction in predictions
    )

    print("\n[PASS] All moves are legal")
    print("[PASS] Probabilities sum to 1.0")


if __name__ == "__main__":
    main()
