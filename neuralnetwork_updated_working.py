# neuralnetwork-updated.py
import os
from typing import Optional

import keras
import pygame, sys
from pygame.locals import *
import copy, random
import numpy as np
from tensorflow.keras.models import load_model

pygame.font.init()


##COLORS##
#             R    G    B
WHITE = (255, 255, 255)
BEIGE = (210, 181, 159)
BROWN = (39, 38, 35)
BLACK = (39, 38, 35)
GOLD = (255, 210, 0)
HIGH = (140, 180, 140)


##DIRECTIONS##
left = "left"
right = "right"
up = "up"
down = "down"

MODEL: keras.Model = load_model('random_model.keras', compile=False)

class Game:
    """
    The main game control.
    """

    def __init__(self):
        self.graphics = Graphics()
        self.board = Board()
        self.turn = BEIGE
        self.selected_piece = None  # a board location.
        self.hop = False
        self.selected_legal_moves = []
        self.board_history = []  # x
        self.move_score_history = []  # y

    def add_to_history(self, best_move):
        new_board = self.board.clone()
        new_board.move_piece(best_move[0], best_move[1])
        board_matrix = new_board.generate_training_matrix(self.turn)

        # Add the channel dimension
        board_matrix = np.expand_dims(board_matrix, axis=-1)

        self.board_history.append(board_matrix)
        self.move_score_history.append(new_board.evaluate_board())

    def setup(self):
        """Draws the window and board at the beginning of the game"""
        self.graphics.setup_window()

    def event_loop(self):
        """
        The event loop. This is where events are triggered
        (like a mouse click) and then effect the game state.
        """
        self.mouse_pos = self.graphics.board_coords(pygame.mouse.get_pos())  # what square is the mouse in?
        if self.selected_piece is not None:
            self.selected_legal_moves = self.board.legal_moves(self.selected_piece, self.hop)

        if self.turn == BROWN:
            # AI player's turn

            move = self.ai_move(self.hop)
            if move is None:
                self.end_turn()
                return

            if self.hop and move[1] in self.board.adjacent(move[0]):
                self.end_turn()
                return

            self.board.move_piece(move[0], move[1])
            if move[1] not in self.board.adjacent(move[0]):
                captured_piece = ((move[0][0] + move[1][0]) >> 1, (move[0][1] + move[1][1]) >> 1)
                self.board.remove_piece(captured_piece)
                self.hop = True
                self.selected_piece = move[1]
            else:
                self.end_turn()
            self.add_to_history(move)  # Record the game history after the AI move

        else:
            # Human player's turn
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.terminate_game()

                if event.type == MOUSEBUTTONDOWN:
                    if not self.hop:
                        if (
                                self.board.location(self.mouse_pos).occupant is not None
                                and self.board.location(self.mouse_pos).occupant.color == self.turn
                        ):
                            self.selected_piece = self.mouse_pos
                        elif (
                                self.selected_piece is not None
                                and self.mouse_pos in self.board.legal_moves(self.selected_piece)
                        ):
                            self.add_to_history(
                                (self.selected_piece, self.mouse_pos))  # Record the game history after the human move
                            self.board.move_piece(self.selected_piece, self.mouse_pos)
                            if self.mouse_pos not in self.board.adjacent(self.selected_piece):
                                captured_piece = (
                                    (self.selected_piece[0] + self.mouse_pos[0]) >> 1,
                                    (self.selected_piece[1] + self.mouse_pos[1]) >> 1,
                                )
                                self.board.remove_piece(captured_piece)
                                self.hop = True
                                self.selected_piece = self.mouse_pos
                            else:
                                self.end_turn()

                    if self.hop:
                        if (
                                self.selected_piece is not None
                                and self.mouse_pos in self.board.legal_moves(self.selected_piece, self.hop)
                        ):
                            self.board.move_piece(self.selected_piece, self.mouse_pos)
                            captured_piece = (
                                (self.selected_piece[0] + self.mouse_pos[0]) >> 1,
                                (self.selected_piece[1] + self.mouse_pos[1]) >> 1,
                            )
                            self.board.remove_piece(captured_piece)

                        if not self.board.legal_moves(self.mouse_pos, self.hop):
                            self.end_turn()
                        else:
                            self.selected_piece = self.mouse_pos


    def update(self):
        """Calls on the graphics class to update the game display."""
        self.graphics.update_display(self.board, self.selected_legal_moves, self.selected_piece)

    def check_for_endgame(self):
        brown_pieces = 0
        beige_pieces = 0
        brown_has_moves = False
        beige_has_moves = False

        for x in range(8):
            for y in range(8):
                if self.board.matrix[x][y].occupant is not None:
                    if self.board.matrix[x][y].occupant.color == BROWN:
                        brown_pieces += 1
                        if self.board.legal_moves((x, y)):
                            brown_has_moves = True
                    elif self.board.matrix[x][y].occupant.color == BEIGE:
                        beige_pieces += 1
                        if self.board.legal_moves((x, y)):
                            beige_has_moves = True

        if brown_pieces == 0 or (beige_pieces > 0 and not brown_has_moves):
            self.show_popup("Beige Wins!")
            self.save_game_history('train_x.npy', 'train_y.npy')  # Save game history to training data
            return True
        elif beige_pieces == 0 or (brown_pieces > 0 and not beige_has_moves):
            self.show_popup("Black Wins!")
            self.save_game_history('train_x.npy', 'train_y.npy')  # Save game history to training data
            return True
        else:
            return False

    def save_game_history(self, x_file, y_file):
        try:
            existing_history_x = np.load(x_file)
            existing_history_y = np.load(y_file)
        except FileNotFoundError:
            existing_history_x = None
            existing_history_y = None

        output_x = np.array(self.board_history)
        output_x = np.expand_dims(output_x, -1)
        output_y = np.array(self.move_score_history)
        output_y = np.expand_dims(output_y, -1)
        if existing_history_x is not None:
            output_x = np.vstack((existing_history_x, output_x))
        if existing_history_y is not None:
            output_y = np.vstack((existing_history_y, output_y))

        np.save(x_file, output_x)
        np.save(y_file, output_y)


    def show_popup(self, message):
        font = pygame.font.Font(None, 36)
        text = font.render(message, True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.graphics.window_size // 2, self.graphics.window_size // 2))

        popup_width = 200
        popup_height = 100
        popup_rect = pygame.Rect((self.graphics.window_size - popup_width) // 2,
                                 (self.graphics.window_size - popup_height) // 2,
                                 popup_width, popup_height)

        pygame.draw.rect(self.graphics.screen, (0, 0, 0), popup_rect)
        pygame.draw.rect(self.graphics.screen, (255, 255, 255), popup_rect, 2)
        self.graphics.screen.blit(text, text_rect)
        pygame.display.flip()

        pygame.time.delay(3000)  # Delay for 3 seconds

    def terminate_game(self):
        """Terminates the game."""
        pygame.quit()
        sys.exit()

    def main(self):
        """This executes the game and controls its flow."""
        self.setup()

        game_running = True
        while game_running:
            self.event_loop()
            self.update()

            if self.check_for_endgame():
                game_running = False

        self.terminate_game()

    def end_turn(self):
        """
        End the turn. Switches the current player.
        end_turn() also checks for and game and resets a lot of class attributes.
        """
        if self.check_for_endgame():
            return

        if self.turn == BEIGE:
            self.turn = BROWN
        else:
            self.turn = BEIGE

        self.selected_piece = None
        self.selected_legal_moves = []
        self.hop = False

    def legal_moves(self):
        return self.board.legal_moves(self.selected_piece)

    def move_piece(self, start, end):
        return self.board.move_piece(start, end)

    def alpha_beta(self, depth, alpha, beta, is_maximizing):
        return self.board.alpha_beta(depth, is_maximizing, self.board)


    def ai_move(self, hop: bool) -> Optional[tuple[int, int]]:
        best_move = None
        best_score = -float('inf')
        max_captures = 0
        possible_moves = []

        for x in range(8):
            for y in range(8):
                if self.board.matrix[x][y].occupant is not None and self.board.matrix[x][y].occupant.color == self.turn:
                    moves = self.board.legal_moves((x, y))
                    for move in moves:
                        possible_moves.append(((x, y), move))

        # Check if any moves are capture moves
        capture_moves = [move for move in possible_moves if self.is_capture_move(move[0], move[1])]

        if capture_moves:
            for move in capture_moves:
                captures = self.count_captures(move[0], move[1])
                if captures > max_captures:
                    max_captures = captures
                    best_move = move
        elif hop:
            return
        else:
            is_maximizing = self.turn == BROWN
            best_score, best_move = self.board.alpha_beta(3, is_maximizing, self.board)

        return best_move

    def is_capture_move(self, start, end):
        return abs(start[0] - end[0]) > 1 or abs(start[1] - end[1]) > 1

    def count_captures(self, start, end):
        captures = 0
        current = start
        while current != end:
            next_x = current[0] + (1 if end[0] > current[0] else -1 if end[0] < current[0] else 0)
            next_y = current[1] + (1 if end[1] > current[1] else -1 if end[1] < current[1] else 0)
            if self.board.location((next_x, next_y)).occupant is not None:
                captures += 1
            current = (next_x, next_y)
        return captures


class Graphics:
    def __init__(self):
        self.caption = "Checkers"

        self.fps = 60
        self.clock = pygame.time.Clock()

        self.window_size = 600
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        self.background = pygame.image.load("board.jpg")

        self.square_size = self.window_size >> 3
        self.piece_size = self.square_size >> 1

        self.message = False

    def setup_window(self):
        """
        This initializes the window and sets the caption at the top.
        """
        pygame.init()
        pygame.display.set_caption(self.caption)

    def update_display(self, board, legal_moves, selected_piece):
        """
        This updates the current display.
        """
        self.screen.blit(self.background, (0, 0))

        self.highlight_squares(legal_moves, selected_piece)
        self.draw_board_pieces(board)

        if self.message:
            self.screen.blit(self.text_surface_obj, self.text_rect_obj)

        pygame.display.update()
        self.clock.tick(self.fps)

    def draw_board_squares(self, board):
        """
        Takes a board object and draws all of its squares to the display
        """
        for x in range(8):
            for y in range(8):
                pygame.draw.rect(self.screen, board[x][y].color,
                                 (x * self.square_size, y * self.square_size, self.square_size, self.square_size), )

    def draw_board_pieces(self, board):
        """
        Takes a board object and draws all of its pieces to the display
        """
        for x in range(8):
            for y in range(8):
                if board.matrix[x][y].occupant is not None:
                    pygame.draw.circle(self.screen, board.matrix[x][y].occupant.color, self.pixel_coords((x, y)),
                                       self.piece_size)

                    if board.location((x, y)).occupant.king:
                        pygame.draw.circle(self.screen, GOLD, self.pixel_coords((x, y)), int(self.piece_size / 1.7),
                                           self.piece_size >> 2)

    def pixel_coords(self, board_coords):
        """
        Takes in a tuple of board coordinates (x,y)
        and returns the pixel coordinates of the center of the square at that location.
        """
        return (
            board_coords[0] * self.square_size + self.piece_size, board_coords[1] * self.square_size + self.piece_size)

    def board_coords(self, pixel):
        """
        Does the reverse of pixel_coords(). Takes in a tuple of of pixel coordinates and returns what square they are in.
        """
        return pixel[0] // self.square_size, pixel[1] // self.square_size

    def highlight_squares(self, squares, origin):
        """
        Squares is a list of board coordinates.
        highlight_squares highlights them.
        """
        for square in squares:
            pygame.draw.rect(self.screen, HIGH, (
                square[0] * self.square_size, square[1] * self.square_size, self.square_size, self.square_size))

        if origin != None:
            pygame.draw.rect(self.screen, HIGH, (
                origin[0] * self.square_size, origin[1] * self.square_size, self.square_size, self.square_size))

    def draw_message(self, message):
        """
        Draws message to the screen.
        """
        self.message = True
        self.font_obj = pygame.font.Font('freesansbold.ttf', 44)
        self.text_surface_obj = self.font_obj.render(message, True, HIGH, BLACK)
        self.text_rect_obj = self.text_surface_obj.get_rect()
        self.text_rect_obj.center = (self.window_size >> 1, self.window_size >> 1)



class Board:
    def __init__(self):
        self.matrix = self.new_board()

    def game_over(self):
        print("Checking if game is over")  # Add this debug print
        for x in range(8):
            for y in range(8):
                if self.matrix[x][y].occupant is not None:
                    if self.legal_moves((x, y)):
                        print(f"Found legal move at {x}, {y}")  # Add this debug print
                        return False
        print("No legal moves found, game over")  # Add this debug print
        return True

    def new_board(self):
        """
        Create a new board matrix.
        """

        # initialize squares and place them in matrix

        matrix = [[None] * 8 for i in range(8)]

        # The following code block has been adapted from
        # http://itgirl.dreamhosters.com/itgirlgames/games/Program%20Leaders/ClareR/Checkers/checkers.py
        for x in range(8):
            for y in range(8):
                if (x % 2 != 0) and (y % 2 == 0):
                    matrix[y][x] = Square(WHITE)
                elif (x % 2 != 0) and (y % 2 != 0):
                    matrix[y][x] = Square(BLACK)
                elif (x % 2 == 0) and (y % 2 != 0):
                    matrix[y][x] = Square(WHITE)
                elif (x % 2 == 0) and (y % 2 == 0):
                    matrix[y][x] = Square(BLACK)

        # initialize the pieces and put them in the appropriate squares

        for x in range(8):
            for y in range(1, 3):
                matrix[x][y].occupant = Piece(BROWN)
            for y in range(5, 7):
                matrix[x][y].occupant = Piece(BEIGE)

        return matrix

    def create_random_board(self):
        matrix = [[None] * 8 for i in range(8)]

        # The following code block has been adapted from
        # http://itgirl.dreamhosters.com/itgirlgames/games/Program%20Leaders/ClareR/Checkers/checkers.py
        for x in range(8):
            for y in range(8):
                if (x % 2 != 0) and (y % 2 == 0):
                    matrix[y][x] = Square(WHITE)
                elif (x % 2 != 0) and (y % 2 != 0):
                    matrix[y][x] = Square(BLACK)
                elif (x % 2 == 0) and (y % 2 != 0):
                    matrix[y][x] = Square(WHITE)
                elif (x % 2 == 0) and (y % 2 == 0):
                    matrix[y][x] = Square(BLACK)

        for pawn_color in [BROWN, BEIGE]:
            player_pawns_count: int = random.randint(1, 16)
            available_positions = [(row, col) for row in range(8) for col in range(8) if
                                   matrix[row][col].occupant is None]

            for _ in range(player_pawns_count):
                if not available_positions:
                    break
                random_row, random_column = random.choice(available_positions)
                available_positions.remove((random_row, random_column))

                new_piece = Piece(pawn_color)
                if pawn_color == BROWN and random_row == 0:
                    new_piece.king = True
                elif pawn_color == BEIGE and random_row == 7:
                    new_piece.king = True

                matrix[random_row][random_column].occupant = new_piece

        self.matrix = matrix

    def board_string(self, board):
        """
        Takes a board and returns a matrix of the board space colors. Used for testing new_board()
        """

        board_string = [[None] * 8] * 8

        for x in range(8):
            for y in range(8):
                if board[x][y].color == WHITE:
                    board_string[x][y] = "WHITE"
                else:
                    board_string[x][y] = "BLACK"

        return board_string

    def rel(self, dir, pixel):

        x = pixel[0]
        y = pixel[1]
        if dir == left:
            return (x - 1, y)
        elif dir == right:
            return (x + 1, y)
        elif dir == up:
            return (x, y + 1)
        elif dir == down:
            return (x, y -1)
        else:
            return 0

    def adjacent(self, pixel):
        """
        Returns a list of squares locations that are adjacent (on a diagonal) to (x,y).
        """
        x = pixel[0]
        y = pixel[1]

        return [self.rel(left, (x, y)), self.rel(right, (x, y)), self.rel(up, (x, y)),
                self.rel(down, (x, y))]

    def location(self, pixel):
        """
        Takes a set of coordinates as arguments and returns self.matrix[x][y]
        This can be faster than writing something like self.matrix[coords[0]][coords[1]]
        """
        x = pixel[0]
        y = pixel[1]

        return self.matrix[x][y]

    def blind_legal_moves(self, pixel):
        """
        Returns a list of blind legal move locations from a set of coordinates (x,y) on the board.
        If that location is empty, then blind_legal_moves() return an empty list.
        """

        x = pixel[0]
        y = pixel[1]
        if self.matrix[x][y].occupant is not None:

            if self.matrix[x][y].occupant.king == False and self.matrix[x][y].occupant.color == BEIGE:
                blind_legal_moves = [self.rel(left, (x, y)), self.rel(right, (x, y)), self.rel(down, (x, y))]

            elif self.matrix[x][y].occupant.king == False and self.matrix[x][y].occupant.color == BROWN:
                blind_legal_moves = [self.rel(up, (x, y)), self.rel(left, (x, y)), self.rel(right, (x, y))]

            else:
                blind_legal_moves = [self.rel(left, (x, y)), self.rel(right, (x, y)),
                                     self.rel(up, (x, y)), self.rel(down, (x, y))]

        else:
            blind_legal_moves = []

        return blind_legal_moves

    def legal_moves(self, pixel, hop=False):
        """
        Returns a list of legal move locations from a given set of coordinates (x,y) on the board.
        If that location is empty, then legal_moves() returns an empty list.
        """

        x, y = pixel
        blind_legal_moves = self.blind_legal_moves((x, y))
        capture_moves = []
        non_capture_moves = []

        for move in blind_legal_moves:
            if self.on_board(move):
                if self.location(move).occupant is None:
                    if not hop:
                        non_capture_moves.append(move)
                elif self.location(move).occupant.color != self.location((x, y)).occupant.color:
                    jump = (move[0] + (move[0] - x), move[1] + (move[1] - y))
                    if self.on_board(jump) and self.location(jump).occupant is None:
                        capture_moves.append(jump)

        return capture_moves if capture_moves else non_capture_moves

    def remove_piece(self, pixel):
        """
        Removes a piece from the board at position (x,y).
        """
        x = pixel[0]
        y = pixel[1]
        self.matrix[x][y].occupant = None

    def move_piece(self, pixel_start, pixel_end):
        """
        Move a piece from (start_x, start_y) to (end_x, end_y).
        """
        start_x = pixel_start[0]
        start_y = pixel_start[1]
        end_x = pixel_end[0]
        end_y = pixel_end[1]

        self.matrix[end_x][end_y].occupant = self.matrix[start_x][start_y].occupant
        self.remove_piece((start_x, start_y))

        self.king((end_x, end_y))

    def is_end_square(self, coords):
        """
        Is passed a coordinate tuple (x,y), and returns true or
        false depending on if that square on the board is an end square.

        """

        if coords[1] == 0 or coords[1] == 7:
            return True
        else:
            return False

    def on_board(self, pixel):

        x = pixel[0]
        y = pixel[1]
        if x < 0 or y < 0 or x > 7 or y > 7:
            return False
        else:
            return True

    def king(self, pixel):
        """
        Takes in (x,y), the coordinates of square to be considered for kinging.
        If it meets the criteria, then king() kings the piece in that square and kings it.
        """
        x = pixel[0]
        y = pixel[1]
        if self.location((x, y)).occupant != None:
            if (self.location((x, y)).occupant.color == BEIGE and y == 0) or (
                    self.location((x, y)).occupant.color == BROWN and y == 7):
                self.location((x, y)).occupant.king = True

    """" this is added to the board class because when saving game history in generate_random_training_data2 it is used
       the old evaluation function instead of cnn so that is why to not get an error I added this function to the code 
       but if needed this can be removed with the save_game_history function  !!!  IT IS NOT USED IN ANY PLACE OF THE 
       CODE EXCEPT IT IS USED FOR SAVING GAME HISTORY SO CNN IS THE ONLY SOLUTION FOR HEURISTIC EVALUATION !!!  """

    def evaluate_board(self):
        brown_score = 0
        beige_score = 0
        score_min = -50
        score_max = 50
        for x in range(8):
            for y in range(8):
                occupant = self.matrix[x][y].occupant
                if occupant is not None:
                    # Base values for piece types
                    base_value = 3 if occupant.king else 1

                    # Additional score for moving towards becoming a king
                    progression_bonus = 0.5 * (y if occupant.color == BROWN else 7 - y)

                    # Bonus for controlling the center of the board
                    center_control_bonus = 0.5 if (x in [3, 4] and y in [3, 4]) else 0

                    # Capture prevention bonus
                    prevention_bonus = self.capture_prevention_bonus(x, y, occupant.color)

                    # Total score for this piece
                    piece_score = base_value + progression_bonus + center_control_bonus + prevention_bonus

                    if occupant.color == BROWN:
                        brown_score += piece_score
                    else:
                        beige_score += piece_score

        score = brown_score - beige_score
        score = (score - score_min) / (score_max - score_min)
        return score

    def capture_prevention_bonus(self, x, y, color):
        """
        Calculate a bonus for positioning that prevents opponent captures.
        """
        bonus = 0
        # Get opponent's color and potential capture moves
        opponent_color = BEIGE if color == BROWN else BROWN
        opponent_moves = self.get_opponent_moves(opponent_color)
        # Check if the current piece's position blocks any captures
        for start, end in opponent_moves:
            if end == (x, y):
                bonus += 5  # Increment bonus for blocking a capture move
        return bonus

    def get_opponent_moves(self, opponent_color):
        """
        Collect all legal moves available to the opponent.
        """
        moves = []
        for x in range(8):
            for y in range(8):
                if self.matrix[x][y].occupant and self.matrix[x][y].occupant.color == opponent_color:
                    legal_ends = self.legal_moves((x, y), hop=True)
                    moves.extend([(x, y), end] for end in legal_ends)

        return moves

    def evaluate_move(self, board_state):
        return MODEL.predict(board_state)    #CNN as Heuristic Evaluation, takes a board state and uses a pre-trained model (CNN) to predict

    def generate_training_matrix(self, player_color) -> np.ndarray:
        training_matrix = np.zeros((8, 8))

        for row_index in range(8):
            for column_index in range(8):
                square = self.matrix[row_index][column_index]
                piece_number = self.piece_weight(square.occupant, player_color)

                training_matrix[row_index][column_index] = piece_number

        return training_matrix

    """ Once all squares have been processed, the method returns training_matrix. This matrix provides 
    a numerical representation of the current state of the game board from the perspective of the player 
    with the color player_color, which can be used as input for a machine learning model."""

    def piece_weight(self, piece, player_color):
        if piece is None:
            pawn_number = 0
        elif piece.king:
            pawn_number = 3
            # pawn_number = 1  # king currently plays like a pawn
        else:
            pawn_number = 1

        if piece is not None and not piece.color == player_color:
            pawn_number *= -1

        return pawn_number

    @staticmethod
    def alpha_beta(depth: int, is_maximizing: bool, current_board: "Board"):
        color = BROWN if is_maximizing else BEIGE
        if not is_maximizing:
            # use the opponent's color
            if color == BEIGE:
                color = BROWN
            else:
                color = BEIGE
        all_board_states = []
        all_legal_moves = []
        scores = []

        for x in range(8):
            for y in range(8):
                if current_board.matrix[x][y].occupant is not None and current_board.matrix[x][
                    y].occupant.color == color:
                    moves = current_board.legal_moves((x, y))
                    for move in moves:
                        selected_move = ((x, y), move)
                        all_legal_moves.append(selected_move)

                        new_board = current_board.clone()
                        new_board.move_piece(selected_move[0], selected_move[1])
                        board_matrix = new_board.generate_training_matrix(color)
                        all_board_states.append(board_matrix)

                        if depth > 1:
                            future_score = current_board.alpha_beta(depth - 1, not is_maximizing, new_board)[0]
                            scores.append(future_score)

        # Convert to numpy array and add channel dimension
        all_board_states = np.array(all_board_states)
        all_board_states = np.expand_dims(all_board_states, axis=-1)

        evaluation_results = current_board.evaluate_move(all_board_states)
        if scores:
            evaluation_results = np.add(evaluation_results, np.array(scores).reshape((-1, 1)))

        best_score_index = np.argmax(evaluation_results) if is_maximizing else np.argmin(evaluation_results)
        best_score = float(evaluation_results[best_score_index][0])
        best_move = all_legal_moves[best_score_index]

        return best_score, best_move

    def clone(self):
        new_object = Board()
        new_object.matrix = self.clone_matrix()

        return new_object

    def clone_matrix(self):
        new_matrix = []

        for column_index in range(len(self.matrix)):
            row = []
            for row_index in range(len(self.matrix[column_index])):
                row.append(self.matrix[column_index][row_index].clone())
            new_matrix.append(row)

        return new_matrix

class Piece:
    def __init__(self, color, king=False):
        self.color = color
        self.king = king

    def clone(self):
        new_object = Piece(self.color, self.king)
        return new_object


class Square:
    def __init__(self, color, occupant=None):
        self.color = color  # color is either BLACK or WHITE
        self.occupant: Optional[Piece] = occupant  # occupant is a Piece object

    def clone(self):
        new_object = Square(self.color)
        if self.occupant is not None:
            assert isinstance(self.occupant, Piece)
            new_object.occupant = self.occupant.clone()

        return new_object
def main():
    game = Game()
    game.main()


if __name__ == "__main__":
    main()