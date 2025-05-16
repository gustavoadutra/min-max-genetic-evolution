import numpy as np
import matplotlib.pyplot as plt
import time
import random
import pickle
import os

from connect4game import Connect4Game
from minmax_agent import MinMaxAgent
from genetic_algorithm import GeneticAlgorithm

def play_human_vs_agent(agent, human_player=1):
    """
    Play a game between a human and the AI agent.
    
    Args:
        agent: The AI agent
        human_player: 1 if human plays first, -1 if AI plays first
    """
    game = Connect4Game()
    
    print("Playing Connect-4 on a 5x5 board")
    print("You need to connect 4 pieces in a row to win!")
    print("Player 1 (X) vs Player 2 (O)")
    
    while not game.game_over:
        # Print current board
        game.print_board()
        
        if game.current_player == human_player:
            # Human's turn
            valid_moves = game.get_valid_moves()
            print(f"Valid moves: {valid_moves}")
            
            while True:
                try:
                    move = int(input("Enter your move (column 0-4): "))
                    if move in valid_moves:
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter a number between 0 and 4.")
        else:
            # AI's turn
            print("AI is thinking...")
            start_time = time.time()
            move = agent.choose_move(game)
            end_time = time.time()
            print(f"AI chose column {move} in {end_time - start_time:.2f} seconds")
            print(f"Nodes evaluated: {agent.nodes_evaluated}")
        
        # Make the move
        game.make_move(move)
    
    # Game over
    game.print_board()
    
    if game.winner is None:
        print("The game ended in a draw!")
    elif game.winner == human_player:
        print("Congratulations! You won!")
    else:
        print("The AI won. Better luck next time!")

def play_agent_vs_agent(agent1, agent2, num_games=10, print_boards=False):
    """
    Play multiple games between two AI agents, alternating who goes first.
    
    Args:
        agent1: First AI agent
        agent2: Second AI agent
        num_games: Number of games to play
        print_boards: Whether to print board states during the game
        
    Returns:
        Statistics about the games
    """
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    for game_idx in range(num_games):
        game = Connect4Game()
        
        # Alternate which agent goes first
        agent1_plays_first = game_idx % 2 == 0
        
        if print_boards:
            print(f"\nGame {game_idx + 1}")
            print(f"{'Agent 1' if agent1_plays_first else 'Agent 2'} plays first")
            game.print_board()
        
        # Play until game over
        while not game.game_over:
            current_agent = None
            if game.current_player == 1:  # First player's turn
                current_agent = agent1 if agent1_plays_first else agent2
            else:  # Second player's turn
                current_agent = agent2 if agent1_plays_first else agent1
            
            move = current_agent.choose_move(game)
            game.make_move(move)
            
            if print_boards:
                game.print_board()
        
        # Record result based on who played as which player
        if game.winner is None:
            draws += 1
            if print_boards:
                print("Game ended in a draw")
        elif (game.winner == 1 and agent1_plays_first) or (game.winner == -1 and not agent1_plays_first):
            agent1_wins += 1
            if print_boards:
                print("Agent 1 wins")
        else:
            agent2_wins += 1
            if print_boards:
                print("Agent 2 wins")
    
    # Calculate statistics
    stats = {
        'agent1_wins': agent1_wins,
        'agent2_wins': agent2_wins,
        'draws': draws,
        'agent1_win_rate': agent1_wins / num_games,
        'agent2_win_rate': agent2_wins / num_games,
        'draw_rate': draws / num_games
    }
    
    return stats


def plot_board_heatmap(game, weights, title="Board Evaluation Heatmap"):
    """
    Create a heatmap showing the evaluation of each possible move.
    
    Args:
        game: Current game state
        weights: Weights for the evaluation function
        title: Title for the plot
    """
    # Create an agent with given weights
    agent = MinMaxAgent(evaluation_weights=weights, max_depth=4)
    
    # Get valid moves
    valid_moves = game.get_valid_moves()
    
    # Initialize values
    move_values = np.zeros(5)
    
    # For each valid move, evaluate the resulting board
    for move in valid_moves:
        game_copy = game.clone()
        game_copy.make_move(move)
        
        if game_copy.game_over:
            if game_copy.winner == game.current_player:
                value = 1000  # Win
            elif game_copy.winner == -game.current_player:
                value = -1000  # Loss
            else:
                value = 0  # Draw
        else:
            # Evaluate board from opponent's perspective (Min node)
            value = -agent._evaluate(game_copy)
        
        move_values[move] = value
    
    # Plot heatmap
    plt.figure(figsize=(8, 2))
    plt.imshow([move_values], cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Evaluation')
    plt.xticks(range(5), range(5))
    plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    plt.savefig('move_heatmap.png')
    plt.close()

def main():
    """Main function to run the system"""
    print("Connect-4 AI with Min-Max and Genetic Algorithm")
    print("==============================================")
    
    # Define parameters
    ga_params = {
        'population_size': 20,
        'num_generations': 20,
        'tournament_size': 3,
        'crossover_rate': 0.8,
        'mutation_rate': 0.2,
        'mutation_scale': 0.5
    }
    
    # Check if saved weights exist
    try:
        with open('best_weights.pkl', 'rb') as f:
            best_weights = pickle.load(f)
        print("Loaded saved weights:", best_weights)
        run_evolution = input("Do you want to run a new evolution? (y/n): ").lower() == 'y'
    except FileNotFoundError:
        best_weights = None
        run_evolution = True
    
    if run_evolution:
        # Run genetic algorithm
        ga = GeneticAlgorithm(**ga_params)
        best_weights = ga.run()
        
        # Save best weights
        with open('best_weights.pkl', 'wb') as f:
            pickle.dump(best_weights, f)
        
        # Plot fitness history
        ga.plot_fitness_history()
        
        # Compare with baseline
        print("\nComparing evolved agent with baseline...")
        comparison = ga.compare_with_baseline(best_weights)
        print(f"Games played: {comparison['games_played']}")
        print(f"Evolved agent wins: {comparison['evolved_wins']} ({comparison['evolved_win_rate']:.2%})")
        print(f"Baseline agent wins: {comparison['baseline_wins']} ({comparison['baseline_win_rate']:.2%})")
        print(f"Draws: {comparison['draws']} ({comparison['draw_rate']:.2%})")
    
    # Create agents
    evolved_agent = MinMaxAgent(evaluation_weights=best_weights, max_depth=3)
    baseline_agent = MinMaxAgent(max_depth=3)  # Uses default weights
    
    # Menu
    while True:
        print("\nMenu:")
        print("1. Play against evolved AI")
        print("2. Play against baseline AI")
        print("3. Watch evolved AI vs baseline AI")
        print("4. Compare agents (100 games)")
        print("5. View board evaluation heatmap")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == '1':
            human_player = 1 if input("Do you want to go first? (y/n): ").lower() == 'y' else -1
            play_human_vs_agent(evolved_agent, human_player)
        elif choice == '2':
            human_player = 1 if input("Do you want to go first? (y/n): ").lower() == 'y' else -1
            play_human_vs_agent(baseline_agent, human_player)
        elif choice == '3':
            stats = play_agent_vs_agent(evolved_agent, baseline_agent, num_games=1, print_boards=True)
        elif choice == '4':
            print("Running 100 games between evolved and baseline agents...")
            stats = play_agent_vs_agent(evolved_agent, baseline_agent, num_games=100)
            print(f"Evolved agent wins: {stats['agent1_wins']} ({stats['agent1_win_rate']:.2%})")
            print(f"Baseline agent wins: {stats['agent2_wins']} ({stats['agent2_win_rate']:.2%})")
            print(f"Draws: {stats['draws']} ({stats['draw_rate']:.2%})")
        elif choice == '5':
            # Create a new game and make some moves to get an interesting state
            game = Connect4Game()
            game.make_move(2)  # Center
            game.make_move(1)
            game.make_move(3)
            game.make_move(0)
            
            print("Current board:")
            game.print_board()
            plot_board_heatmap(game, best_weights, "Evolved Agent's Board Evaluation")
            print("Heatmap saved as 'move_heatmap.png'")
        elif choice == '6':
            print("Thanks for playing!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()