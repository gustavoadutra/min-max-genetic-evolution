import numpy as np
import random
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import copy

class GeneticAlgorithm:
    """
    Genetic Algorithm to evolve weights for the Connect-4 evaluation function.
    """
    def __init__(self, population_size=20, num_generations=30, tournament_size=3, 
                 crossover_rate=0.8, mutation_rate=0.2, mutation_scale=0.5):
        """
        Initialize the Genetic Algorithm.
        
        Args:
            population_size: Number of individuals in the population
            num_generations: Number of generations to evolve
            tournament_size: Number of individuals in each tournament selection
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            mutation_scale: Scale of mutation (how much to change the weights)
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        
        # Store fitness history for analysis
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        # Define the standard opponent (non-evolved agent)
        self.baseline_weights = {
            'open_lines': 1.0,
            'three_in_a_row': 2.0,
            'center_control': 0.5
        }
        
    def initialize_population(self):
        """
        Initialize a random population of weight vectors.
        
        Returns:
            A list of weight dictionaries
        """
        population = []
        
        for _ in range(self.population_size):
            # Create random weights between 0 and 3
            weights = {
                'open_lines': random.uniform(0, 3),
                'three_in_a_row': random.uniform(0, 3),
                'center_control': random.uniform(0, 3)
            }
            population.append(weights)
            
        return population
    
    def calculate_fitness(self, weights):
        """
        Calculate fitness by playing against standard agents.
        
        Args:
            weights: The weight dictionary for evaluation function
            
        Returns:
            Fitness score (number of wins and other performance metrics)
        """
        from connect4game import Connect4Game
        from minmax_agent import MinMaxAgent
        
        # Create agent with the given weights
        agent = MinMaxAgent(evaluation_weights=weights, max_depth=3)
        
        # Create baseline agent
        baseline_agent = MinMaxAgent(evaluation_weights=self.baseline_weights, max_depth=3)
        
        # Play multiple games against the baseline agent
        num_games = 6  # 3 as first player, 3 as second player
        wins = 0
        score = 0  # Score system: win=3, draw=1, loss=0
        
        for game_idx in range(num_games):
            game = Connect4Game()
            is_first_player = game_idx < (num_games // 2)
            
            # Play until game over
            while not game.game_over:
                if (is_first_player and game.current_player == 1) or \
                   (not is_first_player and game.current_player == -1):
                    # Evolved agent's turn
                    move = agent.choose_move(game)
                else:
                    # Baseline agent's turn
                    move = baseline_agent.choose_move(game)
                
                if move is not None:
                    game.make_move(move)
            
            # Calculate score based on outcome
            if game.winner is None:  # Draw
                score += 1
            elif (game.winner == 1 and is_first_player) or \
                 (game.winner == -1 and not is_first_player):
                wins += 1
                score += 3  # Win
        
        # Add a small regularization to avoid extreme weights
        regularization = -0.01 * (abs(weights['open_lines']) + 
                                 abs(weights['three_in_a_row']) + 
                                 abs(weights['center_control']))
        
        # Return combined fitness
        return score + regularization, wins
    
    def tournament_selection(self, population, fitness_scores):
        """
        Select an individual using tournament selection.
        
        Args:
            population: List of individuals
            fitness_scores: List of fitness scores
            
        Returns:
            Selected individual
        """
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i][0] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return copy.deepcopy(population[winner_idx])
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent (weight dictionary)
            parent2: Second parent (weight dictionary)
            
        Returns:
            Two children (weight dictionaries)
        """
        if random.random() < self.crossover_rate:
            # Uniform crossover
            child1 = {}
            child2 = {}
            
            for key in parent1.keys():
                if random.random() < 0.5:
                    child1[key] = parent1[key]
                    child2[key] = parent2[key]
                else:
                    child1[key] = parent2[key]
                    child2[key] = parent1[key]
            
            return child1, child2
        else:
            # No crossover, return copies of parents
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
    
    def mutate(self, individual):
        """
        Mutate an individual's weights.
        
        Args:
            individual: Weight dictionary
            
        Returns:
            Mutated individual
        """
        for key in individual.keys():
            if random.random() < self.mutation_rate:
                # Add a random value between -mutation_scale and +mutation_scale
                mutation = random.uniform(-self.mutation_scale, self.mutation_scale)
                individual[key] += mutation
                
                # Ensure weight doesn't go negative
                individual[key] = max(0.1, individual[key])
                
        return individual
    
    def evolve_population(self, population, fitness_scores):
        """
        Evolve the population using selection, crossover, and mutation.
        
        Args:
            population: Current population
            fitness_scores: List of fitness scores
            
        Returns:
            New population
        """
        new_population = []
        
        # Elitism: Keep the best individual
        best_idx = fitness_scores.index(max(fitness_scores, key=lambda x: x[0]))
        new_population.append(copy.deepcopy(population[best_idx]))
        
        # Generate rest of the population
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self.tournament_selection(population, fitness_scores)
            parent2 = self.tournament_selection(population, fitness_scores)
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
                
        return new_population  # This line was missing

    def run(self):
        """
        Run the genetic algorithm for the specified number of generations.
        
        Returns:
            The best weight set found
        """
        # Initialize population
        population = self.initialize_population()
        
        print(f"Starting genetic algorithm with population size {self.population_size}")
        
        # Evolve for specified number of generations
        for generation in range(self.num_generations):
            print(f"Generation {generation+1}/{self.num_generations}")
            
            # Calculate fitness for each individual in the population
            fitness_scores = []
            with ProcessPoolExecutor() as executor:
                # Use parallel processing to speed up fitness calculations
                fitness_results = list(executor.map(self.calculate_fitness, population))
                fitness_scores = fitness_results
            
            # Find best individual in this generation
            best_idx = fitness_scores.index(max(fitness_scores, key=lambda x: x[0]))
            best_fitness = fitness_scores[best_idx][0]
            best_wins = fitness_scores[best_idx][1]
            best_individual = population[best_idx]
            
            # Calculate average fitness
            avg_fitness = sum(score[0] for score in fitness_scores) / len(fitness_scores)
            
            # Store history
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            print(f"  Best fitness: {best_fitness:.2f}, Wins: {best_wins}, Weights: {best_individual}")
            print(f"  Avg fitness: {avg_fitness:.2f}")
            
            # Evolve population for next generation
            population = self.evolve_population(population, fitness_scores)
            
        # Return the best individual from the final generation
        final_fitness_scores = []
        for individual in population:
            score = self.calculate_fitness(individual)
            final_fitness_scores.append(score)
            
        best_idx = final_fitness_scores.index(max(final_fitness_scores, key=lambda x: x[0]))
        best_weights = population[best_idx]
        
        print("Evolution complete!")
        print(f"Best weights found: {best_weights}")
        
        # Plot fitness history
        self.plot_fitness_history()
        
        return best_weights
        
    def plot_fitness_history(self):
        """Plot the fitness history over generations"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.num_generations + 1), self.best_fitness_history, 'b-', label='Best Fitness')
        plt.plot(range(1, self.num_generations + 1), self.avg_fitness_history, 'r-', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True)
        plt.savefig('fitness_evolution.png')
        plt.show()

    def compare_with_baseline(self, best_weights, num_games=30):
        """
        Compare the evolved weights with the baseline in multiple games.
        
        Args:
            best_weights: The evolved weight set to test
            num_games: Number of games to play
            
        Returns:
            Dictionary with comparison statistics
        """
        from connect4game import Connect4Game
        from minmax_agent import MinMaxAgent
        
        print(f"\nComparing evolved agent with baseline over {num_games} games...")
        
        # Create agents
        evolved_agent = MinMaxAgent(evaluation_weights=best_weights, max_depth=3)
        baseline_agent = MinMaxAgent(evaluation_weights=self.baseline_weights, max_depth=3)
        
        # Statistics
        wins_evolved = 0
        wins_baseline = 0
        draws = 0
        
        # Play games
        for game_idx in range(num_games):
            game = Connect4Game()
            is_evolved_first = game_idx < (num_games // 2)
            
            print(f"Game {game_idx+1}/{num_games}: " + 
                  ("Evolved goes first" if is_evolved_first else "Baseline goes first"))
            
            # Play until game over
            while not game.game_over:
                if (is_evolved_first and game.current_player == 1) or \
                   (not is_evolved_first and game.current_player == -1):
                    # Evolved agent's turn
                    move = evolved_agent.choose_move(game)
                else:
                    # Baseline agent's turn
                    move = baseline_agent.choose_move(game)
                
                if move is not None:
                    game.make_move(move)
            
            # Record outcome
            if game.winner is None:
                draws += 1
                print("  Result: Draw")
            elif (game.winner == 1 and is_evolved_first) or \
                 (game.winner == -1 and not is_evolved_first):
                wins_evolved += 1
                print("  Result: Evolved agent wins")
            else:
                wins_baseline += 1
                print("  Result: Baseline agent wins")
        
        # Calculate statistics
        win_rate = wins_evolved / num_games
        baseline_win_rate = wins_baseline / num_games
        draw_rate = draws / num_games  # Add this to calculate draw rate

        # Print summary
        print("\nComparison Results:")
        print(f"  Evolved agent wins: {wins_evolved} ({wins_evolved/num_games:.2%})")
        print(f"  Baseline agent wins: {wins_baseline} ({wins_baseline/num_games:.2%})")
        print(f"  Draws: {draws} ({draws/num_games:.2%})")

        # Return results as dictionary with keys matching what main.py expects
        return {
            "games_played": num_games,
            "evolved_wins": wins_evolved,
            "evolved_win_rate": win_rate,
            "baseline_wins": wins_baseline, 
            "baseline_win_rate": baseline_win_rate,
            "draws": draws,
            "draw_rate": draw_rate
        }