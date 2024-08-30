import os
from neuralnetwork_updated_working import Board, Piece
import random
import copy
import numpy as np

BEIGE = (210, 181, 159)
BROWN = (39, 38, 35)


def find_possible_moves(board_instance: Board) -> list:
    possible_moves = []

    for x in range(8):
        for y in range(8):
            if board_instance.matrix[x][y].occupant is not None and board_instance.matrix[x][y].occupant.color == BROWN:
                moves = board_instance.legal_moves((x, y))
                for move in moves:
                    possible_moves.append(((x, y), move))

    random.shuffle(possible_moves)
    return possible_moves


def find_best_move(board_instance: Board, possible_moves: list) -> tuple[tuple[int]]:
    best_score = None
    best_move = None
    has_capture_move = False

    for move in possible_moves:
        if move[1] not in board_instance.adjacent(move[0]):
            has_capture_move = True
            new_board = copy.deepcopy(board_instance)
            new_board.move_piece(move[0], move[1])
            score = new_board.alpha_beta(3, -float('inf'), float('inf'), False)
            if best_score is None or best_score < score:
                best_score = score
                best_move = move

    if not has_capture_move:
        for move in possible_moves:
            new_board = copy.deepcopy(board_instance)
            new_board.move_piece(move[0], move[1])
            score = new_board.alpha_beta(3, -float('inf'), float('inf'), False)
            if best_score is None or best_score < score:
                best_score = score
                best_move = move

    return best_move  # ((2,4), (2,5)) like this for example


def draw_board(board_instance: Board):
    matrix = generate_training_matrix(board_instance)
    for row in matrix:
        for square in row:
            print(f"[{str(int(square)).ljust(4)}]", end=" ")
        print()
    print("\n\n")

def draw_board_old(board_instance: Board):
    for row in board_instance.matrix:
        for square in row:
            if square.occupant is None:
                print("[    ]", end=" ")
            else:
                piece = square.occupant
                pawn_number = piece_to_number(piece)
                print(f"[{str(pawn_number).ljust(4)}]", end=" ")
        print()
    print("\n")


def generate_training_matrix(board_instance: Board) -> np.ndarray:
    training_matrix = np.zeros((8, 8))

    for row_index in range(8):
        for column_index in range(8):
            square = board_instance.matrix[row_index][column_index]
            piece_number = piece_to_number(square.occupant)

            training_matrix[row_index][column_index] = piece_number

    return training_matrix


def piece_to_number(piece: Piece):
    if piece is None:
        pawn_number = 0
    elif piece.king:
        pawn_number = 3
    else:
        pawn_number = 1

    if piece is not None and piece.color == BEIGE:
        pawn_number *= -1

    return pawn_number

board = Board()
board.create_random_board()
# draw_board_old(board)

epochs = 100000
x = []
y = []

for i in range(epochs):
    print("Training Epoch:", i + 1)
    board.create_random_board()
    possible_moves = find_possible_moves(board)
    current_board_matrix = generate_training_matrix(board)
    draw_board(board)
    x.append(current_board_matrix)
    y.append(board.evaluate_board())



x = np.array(x, dtype=np.float16)
y = np.array(y, dtype=np.float16)

x = np.expand_dims(x, axis=-1)
y = np.expand_dims(y, axis=-1)



# train_index = epochs * 8 // 10  # 80% of the data for training
# test_index = epochs * 2 // 10  # 20% of the data for testing
train_index = len(x) * 8 // 10  # 80% of the data for training
test_index = len(x) * 2 // 10  # 20% of the data for testing


# Check if the files exist and load the existing data if available
if os.path.exists("train_x.npy"):
    existing_train_x = np.load("train_x.npy")
    train_x = np.concatenate((existing_train_x, x[:train_index]))
else:
    train_x = x[:train_index]

if os.path.exists("train_y.npy"):
    existing_train_y = np.load("train_y.npy")
    train_y = np.concatenate((existing_train_y, y[:train_index]))
else:
    train_y = y[:train_index]

if os.path.exists("test_x.npy"):
    existing_test_x = np.load("test_x.npy")
    test_x = np.concatenate((existing_test_x, x[train_index:]))
else:
    test_x = x[train_index:]

if os.path.exists("test_y.npy"):
    existing_test_y = np.load("test_y.npy")
    test_y = np.concatenate((existing_test_y, y[train_index:]))
else:
    test_y = y[train_index:]

# Save the updated data
np.save("train_x.npy", train_x)
np.save("train_y.npy", train_y)
np.save("test_x.npy", test_x)
np.save("test_y.npy", test_y)
