import numpy as np
import copy

class Connect4Game:
    """
    Implementation of Connect-4 on a 5x5 board.
    Player 1 uses pieces with value 1, Player 2 uses pieces with value -1.
    """
    def __init__(self):
        # Initialize empty 5x5 board
        self.board = np.zeros((5, 5), dtype=int)
        self.current_player = 1  # Player 1 starts
        self.winner = None
        self.game_over = False
        
    def get_valid_moves(self):
        """Return columns where a piece can be played (not full)"""
        valid_moves = []
        for col in range(5):
            if self.board[0][col] == 0:  # If top position is empty
                valid_moves.append(col)
        return valid_moves
    
    def make_move(self, col):
        """
        Place a piece in the specified column.
        Returns True if move was valid, False otherwise.
        """
        if col not in self.get_valid_moves():
            return False
        
        # Find the lowest empty row in the column
        for row in range(4, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                break
        
        # Check for win
        if self._check_win():
            self.winner = self.current_player
            self.game_over = True
        # Check for draw
        elif len(self.get_valid_moves()) == 0:
            self.game_over = True
        
        # Switch player
        self.current_player *= -1
        return True
    
    def _check_win(self):
        """Check if the current player has won"""
        player = self.current_player
        
        # Check horizontal
        for row in range(5):
            for col in range(2):  # Only need to check starting positions that can fit 4
                if (self.board[row][col] == player and
                    self.board[row][col+1] == player and
                    self.board[row][col+2] == player and
                    self.board[row][col+3] == player):
                    return True
        
        # Check vertical
        for row in range(2):  # Only need to check starting positions that can fit 4
            for col in range(5):
                if (self.board[row][col] == player and
                    self.board[row+1][col] == player and
                    self.board[row+2][col] == player and
                    self.board[row+3][col] == player):
                    return True
        
        # Check diagonal (down-right)
        for row in range(2):
            for col in range(2):
                if (self.board[row][col] == player and
                    self.board[row+1][col+1] == player and
                    self.board[row+2][col+2] == player and
                    self.board[row+3][col+3] == player):
                    return True
        
        # Check diagonal (up-right)
        for row in range(3, 5):
            for col in range(2):
                if (self.board[row][col] == player and
                    self.board[row-1][col+1] == player and
                    self.board[row-2][col+2] == player and
                    self.board[row-3][col+3] == player):
                    return True
        
        return False
    
    def clone(self):
        """Create a deep copy of the current game state"""
        new_game = Connect4Game()
        new_game.board = copy.deepcopy(self.board)
        new_game.current_player = self.current_player
        new_game.winner = self.winner
        new_game.game_over = self.game_over
        return new_game
    
    def print_board(self):
        """Print the current board state"""
        symbols = {0: ".", 1: "X", -1: "O"}
        print("  0 1 2 3 4")  # Column numbers
        for row in range(5):
            print(f"{row} ", end="")
            for col in range(5):
                print(f"{symbols[self.board[row][col]]} ", end="")
            print()
        print()