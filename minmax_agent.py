import numpy as np
import time

class MinMaxAgent:
    """
    Agent that uses Min-Max algorithm with Alpha-Beta pruning to play Connect-4.
    """
    def __init__(self, evaluation_weights=None, max_depth=4):
        """
        Initialize the Min-Max agent.
        
        Args:
            evaluation_weights: Weights for the evaluation function features
            max_depth: Maximum search depth
        """
        # Default weights if none provided
        if evaluation_weights is None:
            self.evaluation_weights = {
                'open_lines': 1.0,
                'three_in_a_row': 2.0,
                'center_control': 0.5
            }
        else:
            self.evaluation_weights = evaluation_weights
            
        self.max_depth = max_depth
        self.nodes_evaluated = 0  # Counter for performance analysis
        
    def choose_move(self, game):
        """
        Choose the best move using Min-Max with Alpha-Beta pruning.
        
        Args:
            game: Current game state
            
        Returns:
            The column to play in
        """
        self.nodes_evaluated = 0
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:
            return None
        
        best_move = valid_moves[0]
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        # Try each move and select the one with the maximum value
        for move in valid_moves:
            # Create a copy of the game and make the move
            game_copy = game.clone()
            game_copy.make_move(move)
            
            # Get value from Min-Max
            value = self._min_value(game_copy, alpha, beta, 1)
            
            if value > best_value:
                best_value = value
                best_move = move
                
            alpha = max(alpha, best_value)
            
        return best_move
    
    def _max_value(self, game, alpha, beta, depth):
        """
        Get the maximum value (for the maximizing player).
        
        Args:
            game: Current game state
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            depth: Current search depth
            
        Returns:
            The maximum utility value
        """
        self.nodes_evaluated += 1
        
        # Terminal test
        if game.game_over:
            if game.winner == 1:  # AI wins
                return 1000
            elif game.winner == -1:  # Opponent wins
                return -1000
            else:  # Draw
                return 0
                
        # Depth limit reached
        if depth >= self.max_depth:
            return self._evaluate(game)
            
        max_value = -float('inf')
        valid_moves = game.get_valid_moves()
        
        for move in valid_moves:
            game_copy = game.clone()
            game_copy.make_move(move)
            
            max_value = max(max_value, self._min_value(game_copy, alpha, beta, depth + 1))
            
            # Pruning
            if max_value >= beta:
                return max_value
                
            alpha = max(alpha, max_value)
            
        return max_value
    
    def _min_value(self, game, alpha, beta, depth):
        """
        Get the minimum value (for the minimizing player).
        
        Args:
            game: Current game state
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            depth: Current search depth
            
        Returns:
            The minimum utility value
        """
        self.nodes_evaluated += 1
        
        # Terminal test
        if game.game_over:
            if game.winner == 1:  # AI wins
                return 1000
            elif game.winner == -1:  # Opponent wins
                return -1000
            else:  # Draw
                return 0
                
        # Depth limit reached
        if depth >= self.max_depth:
            return self._evaluate(game)
            
        min_value = float('inf')
        valid_moves = game.get_valid_moves()
        
        for move in valid_moves:
            game_copy = game.clone()
            game_copy.make_move(move)
            
            min_value = min(min_value, self._max_value(game_copy, alpha, beta, depth + 1))
            
            # Pruning
            if min_value <= alpha:
                return min_value
                
            beta = min(beta, min_value)
            
        return min_value
    
    def _evaluate(self, game):
        """
        Evaluate the board position using multiple heuristics.
        
        Args:
            game: Current game state
            
        Returns:
            The evaluation score
        """
        # For player 1 (maximizing player)
        player = 1
        opponent = -1
        
        # Feature 1: Open lines (rows, columns, diagonals) where player can still win
        open_lines_player = self._count_open_lines(game.board, player)
        open_lines_opponent = self._count_open_lines(game.board, opponent)
        open_lines_score = open_lines_player - open_lines_opponent
        
        # Feature 2: Three pieces in a row (near win)
        three_in_row_player = self._count_three_in_row(game.board, player)
        three_in_row_opponent = self._count_three_in_row(game.board, opponent)
        three_in_row_score = three_in_row_player - three_in_row_opponent
        
        # Feature 3: Control of the center
        center_control_player = self._count_center_control(game.board, player)
        center_control_opponent = self._count_center_control(game.board, opponent)
        center_control_score = center_control_player - center_control_opponent
        
        # Calculate final score using weights
        final_score = (
            self.evaluation_weights['open_lines'] * open_lines_score +
            self.evaluation_weights['three_in_a_row'] * three_in_row_score +
            self.evaluation_weights['center_control'] * center_control_score
        )
        
        return final_score
    
    def _count_open_lines(self, board, player):
        """Count lines where player can still get 4 in a row (no opponent pieces blocking)"""
        count = 0
        
        # Check horizontal lines
        for row in range(5):
            for col in range(2):  # Starting positions that can fit 4
                line = [board[row][col+i] for i in range(4)]
                if player not in line and -player not in line:  # Empty line
                    count += 1
                elif -player not in line and player in line:  # Line with only player pieces
                    count += 1
        
        # Check vertical lines
        for row in range(2):  # Starting positions that can fit 4
            for col in range(5):
                line = [board[row+i][col] for i in range(4)]
                if player not in line and -player not in line:  # Empty line
                    count += 1
                elif -player not in line and player in line:  # Line with only player pieces
                    count += 1
        
        # Check diagonal lines (down-right)
        for row in range(2):
            for col in range(2):
                line = [board[row+i][col+i] for i in range(4)]
                if player not in line and -player not in line:  # Empty line
                    count += 1
                elif -player not in line and player in line:  # Line with only player pieces
                    count += 1
        
        # Check diagonal lines (up-right)
        for row in range(3, 5):
            for col in range(2):
                line = [board[row-i][col+i] for i in range(4)]
                if player not in line and -player not in line:  # Empty line
                    count += 1
                elif -player not in line and player in line:  # Line with only player pieces
                    count += 1
        
        return count

    def _count_three_in_row(self, board, player):
        """Count instances of three pieces in a row with an empty space"""
        count = 0
        
        # Check horizontal sequences
        for row in range(5):
            for col in range(2):  # Starting positions that can fit 4
                line = [board[row][col+i] for i in range(4)]
                if line.count(player) == 3 and line.count(0) == 1:
                    count += 1
        
        # Check vertical sequences
        for row in range(2):  # Starting positions that can fit 4
            for col in range(5):
                line = [board[row+i][col] for i in range(4)]
                if line.count(player) == 3 and line.count(0) == 1:
                    count += 1
        
        # Check diagonal sequences (down-right)
        for row in range(2):
            for col in range(2):
                line = [board[row+i][col+i] for i in range(4)]
                if line.count(player) == 3 and line.count(0) == 1:
                    count += 1
        
        # Check diagonal sequences (up-right)
        for row in range(3, 5):
            for col in range(2):
                line = [board[row-i][col+i] for i in range(4)]
                if line.count(player) == 3 and line.count(0) == 1:
                    count += 1
        
        return count
    
    def _count_center_control(self, board, player):
        """Count pieces in center column and central area"""
        count = 0
        
        # Center column (higher weight)
        for row in range(5):
            if board[row][2] == player:  # Column 2 is the center in a 5x5 board
                count += 2
        
        # Central region (3x3 grid in the middle)
        for row in range(1, 4):
            for col in range(1, 4):
                if board[row][col] == player:
                    count += 1
        
        return count