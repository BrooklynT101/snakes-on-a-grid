__author__ = "Brooklyn Taylor"
__organization__ = "COSC343, University of Otago"
__email__ = "taybr713@student.otago.ac.nz"

import numpy as np
from typing import List
from typing import Tuple

agentName = "SinglePerceptronGA"
# trainingSchedule = [("self", 1), ("random", 1)]
# Start with random opponent only to keep things simple and fast.
trainingSchedule = [("random", 200)]
# average score after 200 games 3.76e+00
FORWARD = 0
LEFT = -1
RIGHT = 1
FOOD = -0.5
EMPTY = 0.0

# -----------------------------------------------------------------------------
# GA configuration (these values were suggested by CGPT to start with, will be adjusted later)
# -----------------------------------------------------------------------------
ELITISM_COUNT = 2               # Number of top snakes copied straight to next gen
TOURNAMENT_SIZE = 3             # We pick the best of this many snakes during selection
CROSSOVER_PROBABILITY = 0.7     # Chance we perform crossover; otherwise copy a parent
MUTATION_PROBABILITY = 0.05     # Chance that each gene will be mutated
MUTATION_STANDARD_DEVIATION = 0.10  # Size of the random nudge during mutation
WEIGHT_CLIP_LIMIT = 3.0         # Keep weights within a safe range after mutation

# -----------------------------------------------------------------------------
# Fitness evaluation modifiers
# -----------------------------------------------------------------------------
FOOD_REWARD = 0.5              # Reward for eating food
FRIEND_ATTACK_PENALTY = -0.3      # Penalty for biting a friendly snake
ENEMY_ATTACK_REWARD = 0.3       # Reward for biting an enemy snake
HEAD_CRASH_PENALTY = -0.2        # Penalty for crashing heads with another snake

# This is to track average fitness across generations so I can plot it later
fitness_history_csv_filename = "fitness_history.csv"
_average_fitness_per_generation: List[float] = []

# -----------------------------------------------------------------------------
# Snake class
# -----------------------------------------------------------------------------
class Snake:    
    """
    Very small neural network ("perceptron"):
    inputs:  49  (flattened 7x7 percept grid)
    outputs: 3   (scores for Left, Forward, Right)

    Chromosome layout (length = 150):
    - First 49*3 = 147 numbers are the weight matrix (49 rows, 3 columns) flattened
    - Last 3 numbers are the bias vector (one bias per output)

    Action selection:
    - We compute scores = x @ W + b
    - We choose the index of the largest score (0, 1, or 2)
    - We map indices to actions [-1, 0, 1] using the "actions" list provided by the engine
    """

    def __init__(self, nPercepts, actions):
        # The engine will pass:
        #   number_of_percepts = 49 (7x7 grid flattened)
        #   actions = [-1, 0, 1]
        self.nPercepts = nPercepts
        self.actions = actions

        # Define the small network shape explicitly for slef clarity
        self.number_of_inputs = 49               # 7x7 percepts flattened
        self.number_of_outputs = 3               # left, forward, right
        self.number_of_weights = self.number_of_inputs * self.number_of_outputs  # 49*3 = 147
        self.number_of_biases = self.number_of_outputs                           # 3
        self.chromosome_length = self.number_of_weights + self.number_of_biases  # 150

       # Create a random chromosome for initial behaviour (untrained) small random values to avoid extreme outputs at the start.
       # TODO: look into gaussian mutation
        self.chromosome = np.random.uniform(
            low=-0.5,
            high=0.5,
            size=self.chromosome_length
        ).astype(np.float32)

    
    # Helper method to decode the chromosome into weights and biases
    # Copilot helped with this implementation
    def _decode_chromosome_into_weights_and_biases(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Turn the flat chromosome into:
          - weight_matrix with shape (49, 3)
          - bias_vector with shape (3,)
        """
        # The first 147 numbers are weights
        weight_values = self.chromosome[0:self.number_of_weights]

        # The last 3 numbers are biases
        bias_values = self.chromosome[self.number_of_weights:self.chromosome_length]

        # Reshape weights into (49, 3) for a clean matrix multiply
        weight_matrix = weight_values.reshape(self.number_of_inputs, self.number_of_outputs)

        # Bias is already the correct shape (3,)
        bias_vector = bias_values

        return weight_matrix, bias_vector

    def AgentFunction(self, percepts):
        """
        This function is called once per turn with the snake's local 7x7 view.
        Must return exactly one of the allowed actions: LEFT (-1), FORWARD (0), RIGHT (+1).

        Steps:
          1) Flatten the (7x7) percept grid into a (49,) vector.
          2) Compute the 3 output scores using y = x @ W + b.
          3) Pick the index of the largest score.
          4) Return the corresponding action from self.actions ([-1, 0, 1]).
        """
        # 1) Flatten the percepts
        flat_percepts = percepts.flatten().astype(np.float32)  # Shape (49,)
        # 2) Compute the 3 output scores using y = x @ W + b.
        W, b = self._decode_chromosome_into_weights_and_biases()
        # output_scores = flat_percepts @ W + b  # Shape (3,)
        output_scores = flat_percepts @ W + b  # Shape (3,)
        # 3) Pick the index of the largest score.
        best_action_index = int(np.argmax(output_scores))
        # 4) Return the corresponding action from self.actions ([-1, 0, 1]).
        chosen_action = self.actions[best_action_index]
        return chosen_action

def evalFitness(population):

    N = len(population)

    # Fitness initialiser for all agents
    fitness = np.zeros((N))

    '''
     This loop iterates over your agents in the population - the purpose of this boiler plate
     code is to demonstrate how to fetch information from the population
     to score fitness of each agent
    '''
    for n, snake in enumerate(population):
        '''
         snake is an instance of Snake class that you implemented above, therefore you can access 
         any attributes (such as `self.chromosome').  Additionally, the object has the following
         attributes provided by the game engine; each a list of nTurns values
        
         snake.sizes - list of snake sizes over the game turns (0 means the snake is dead)
         snake.friend_attacks - turns when this snake has bitten another snake, not including
                                head crashes - 0 not bitten in that turn, 1 bitten friendly snake 
         snake.enemy_attacks - turns when this snake has bitten another snake, not including
                              head crashes - 0 not bitten in that turn, 1 bitten enemy snake
         snake.bitten - number of bites received in a given turn (it's possible to be bitten by
                        several snakes in one turn)
         snake.food - turns when food was eaten by the snake, not including biting other snake
                       (0 not eaten food, food eaten)
         snake.friend_crashes - turns when crashed heads with a friendly snake (0 no crash, 1 crash) 
         snake.enemy_crashes - turns when crashed heads with an enemy snake (0 no crash, 1 crash)
        '''
        meanSize = np.mean(snake.sizes)
        # The following two lines demonstrate how to 
        # extract other information from snake.sizes
        turnsAlive = np.sum(snake.sizes > 0)
        maxTurns = len(snake.sizes)

        '''
         This fitness function considers;
            - mean size of the snake during the game (the larger the better)
            - number of food items eaten (the more the better)
            - number of enemy snakes bitten (the more the better)
            - number of friendly snakes bitten (the fewer the better)
            - number of head crashes with snakes (the fewer the better) TODO: tweak this to consider friend vs enemy crashes
        '''
        f = meanSize \
            + FOOD_REWARD * np.sum(snake.food) \
            + ENEMY_ATTACK_REWARD * np.sum(snake.enemy_attacks) \
            + FRIEND_ATTACK_PENALTY * np.sum(snake.friend_attacks) \
            + HEAD_CRASH_PENALTY * (np.sum(snake.friend_crashes) + np.sum(snake.enemy_crashes))
        fitness[n] = f

    return fitness

# -----------------------------------------------------------------------------
# Genetic Algorithm functions
# -----------------------------------------------------------------------------

# Helper function to log average fitness - saving to a CSV file
def saveFitnessHistory(avg_fitness):
        '''
        Save the average fitness of the generation to a CSV file for later analysis.
        '''
        _average_fitness_per_generation.append(avg_fitness)
        try:
            with open(fitness_history_csv_filename, "a", encoding="utf-8") as file_handle:
                # Store "generation_index,average_fitness"
                generation_index = len(_average_fitness_per_generation) - 1
                file_handle.write(f"{generation_index},{avg_fitness}\n")
        except Exception:
            # If logging fails (e.g., no write permission), we just continue
            pass

# Factory method to create a child snake with a given chromosome
def create_child(parent_snake, chromosome):
    child = Snake(parent_snake.nPercepts, parent_snake.actions)
    child.chromosome = chromosome
    return child

# Tournament selection function
def tournament_selection(population, fitness, tournament_size):
    '''
    Select one parent using tournament selection.

    Steps:
      1) Randomly select 'tournament_size' individuals from the population.
      2) Choose the one with the highest fitness among them.

    Returns:
      The selected parent (Snake instance).
    '''
    # Pick distict indices for the tournament
    selected_indices = np.random.choice(len(population), size=tournament_size, replace=False)

    # Find the best fitness among the population
    best_index = selected_indices[0]
    best_fitness = fitness[best_index]

    for idx in selected_indices[1:]:
        if fitness[idx] > best_fitness:
            best_fitness = fitness[idx]
            best_index = idx

    return population[best_index]

def one_point_crossover(parent1, parent2):
    '''
    Perform one-point crossover between two parents to produce a child.
    Mixing the chromosomes of the parents to create a new chromosome for the child.

    TODO: define a crossover point to tweak the child's chromosome. Rather than random

    Steps:
      1) Randomly select a crossover point along the chromosome.
      2) Create a new child chromosome by combining genes from both parents.
    Returns:
      The child snake's new chromosome.
      Or a copy of one parent's if no crossover is performed.
    '''

    gene_count = parent1.chromosome_length
    
    # Decide whether to perform crossover or just copy a parent
    if np.random.rand() < CROSSOVER_PROBABILITY:
        # 1) Randomly select a crossover point along the chromosome.
        crossover_point = np.random.randint(1, gene_count)  # Avoid endpoints to ensure mixing

        # 2) Create a new child chromosome by combining genes from both parents.
        child_chromosome = np.zeros(gene_count, dtype=np.float32)
        child_chromosome[0:crossover_point] = parent1.chromosome[0:crossover_point]
        child_chromosome[crossover_point:gene_count] = parent2.chromosome[crossover_point:gene_count]
    else:
        # No crossover; just copy one parent randomly
        if np.random.rand() < 0.5:
            child_chromosome = np.copy(parent1.chromosome)
        else:
            child_chromosome = np.copy(parent2.chromosome)

    return child_chromosome

def mutate_in_place(genes):
    """
    For each gene in the chromosome:
      - With probability 'mutation_probability', add Gaussian noise (mean 0, std 'mutation_standard_deviation').
      - After mutation, clip values into [-weight_clip_limit, +weight_clip_limit] to keep things stable.
    Had Copilot help with the implementation of this function. Fixing some incorrect logic I had tried.
    """
    number_of_genes = genes.size
    for i in range(number_of_genes):
        if np.random.rand() < MUTATION_PROBABILITY:
            noise = np.random.normal(0, MUTATION_STANDARD_DEVIATION)
            genes[i] = genes[i] + noise
            
            if WEIGHT_CLIP_LIMIT is not None:
                if genes[i] > WEIGHT_CLIP_LIMIT:
                    genes[i] = WEIGHT_CLIP_LIMIT
                elif genes[i] < -WEIGHT_CLIP_LIMIT:
                    genes[i] = -WEIGHT_CLIP_LIMIT


def newGeneration(old_population):
    '''
    Build the next generation of snakes.

    Steps:
      1) Evaluate fitness of the old population.
      2) Keep a few best snakes unchanged (elitism).
      3) For the rest:
         - Select two parents by tournament selection.
         - Create a child using one-point crossover.
         - Mutate the child's genes a little bit.
      4) Call a function to log the average fitness.
      5) Return the new population and the average fitness of the old population.

    This function must return a tuple consisting of:
     - a list of the new_population of snakes that is of the same length as the old_population,
     - the average fitness of the old population
    '''
    population_size = len(old_population)

    # 1) Evaluate fitness of the old population.
    fitness_values = evalFitness(old_population)
    if population_size == 0:
        # Edge case: empty population, shouldnt happen but just in case
        print("Population size is zero.")
        return old_population, 0.0

    average_fitness = float(np.mean(fitness_values))

    # Record average fitness for analysis
    saveFitnessHistory(average_fitness)

    # 2) Sort snakes by fitness to find elitism candidates (descending order)
    sorted_indices = np.argsort(fitness_values)[::-1]

    # 3) Create new population list
    new_population = []

    # 3a) Copy top ELITISM_COUNT snakes unchanged
    # LLM Suggested fix for no aliasing issues
    elites_to_copy = min(ELITISM_COUNT, population_size)

    for i in range(elites_to_copy):
        elite = old_population[sorted_indices[i]]
        new_population.append(create_child(elite, elite.chromosome.copy()))

    # 3b) Create the rest of the new population by breeding 
    while len(new_population) < population_size:
        # Select two parents using tournament selection
        parent1 = tournament_selection(old_population, fitness_values, TOURNAMENT_SIZE)
        parent2 = tournament_selection(old_population, fitness_values, TOURNAMENT_SIZE)

        # Perform crossover to create a child chromosome
        child_chromosome = one_point_crossover(parent1, parent2)

        # Mutate the child's chromosome in place
        mutate_in_place(child_chromosome)

        # Create a new snake with the child's chromosome
        child_snake = create_child(parent1, child_chromosome)
        new_population.append(child_snake)
    return new_population, average_fitness
