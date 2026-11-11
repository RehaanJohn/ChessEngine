
import chess
import numpy as np


PIECE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode a chess position into 18 x 64 FP32 features.

    Square index follows python-chess:
    a1 = 0, h1 = 7, a8 = 56, h8 = 63.
    """
    features = np.zeros((18, 64), dtype=np.float32)

    # Piece-location planes.
    for square, piece in board.piece_map().items():
        piece_index = PIECE_ORDER.index(piece.piece_type)

        if piece.color == chess.BLACK:
            piece_index += 6

        features[piece_index, square] = 1.0

    # Side to move: +1 for White, -1 for Black.
    features[12, :] = (
        1.0 if board.turn == chess.WHITE else -1.0
    )

    # Castling rights.
    rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]

    for i, allowed in enumerate(rights):
        if allowed:
            features[13 + i, :] = 1.0

    # En passant target.
    if board.ep_square is not None:
        features[17, board.ep_square] = 1.0

    return features.reshape(-1)


def encode_fen(fen: str) -> np.ndarray:
    board = chess.Board(fen)

    if not board.is_valid():
        raise ValueError("Invalid chess position")

    return encode_board(board)
