__author__ = "Brooklyn Taylor"
__organization__ = "COSC343, University of Otago"
__email__ = "taybr713@student.otago.ac.nz"

import numpy as np
import os
from typing import List
from datetime import datetime

agentName = "HungrySnake"
trainingSchedule = [("self", 1), ("random", 1)]
# trainingSchedule = [("random", 200)]
FORWARD = 0
LEFT = -1
RIGHT = 1
FOOD = -0.5
EMPTY = 0.0

# This is to track average fitness across generations so I can plot it later
fitness_history_csv_filename = "fitness_history.csv"
_average_fitness_per_generation: List[float] = []

# -----------------------------------------------------------------------------
# Snake class
# -----------------------------------------------------------------------------
class Snake:

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

    # -------------------------------------------------------------------------
    # Helper to choose a local action toward the nearest food
    # -------------------------------------------------------------------------
    def _choose_action_towards_food(self, percepts_7x7: np.ndarray) -> int:
        """
        Very simple rule:
          - Find the nearest food cell (value == -0.5) in the 7x7 view.
          - The view is aligned to the snake's heading:
              forward = one row UP (toward smaller row index),
              left    = one column LEFT (toward smaller column index),
              right   = one column RIGHT (toward larger column index).
          - We return -1 for left, 0 for forward, 1 for right.

        If there is no food in view, we simply go forward (0).
        """
        # Head is at the center of the 7x7 grid
        head_row = 3
        head_col = 3

        # Find all coordinates where there is food (-0.5)
        food_positions = np.argwhere(percepts_7x7 == -0.5)

        if food_positions.size == 0:
            # No food seen in the local 7x7 view: just keep moving forward
            return FORWARD

        # Pick the nearest food by Manhattan distance in the local percept grid 
        distances = []
        for (r, c) in food_positions:
            distance = abs(r - head_row) + abs(c - head_col)
            distances.append(distance)
        nearest_index = int(np.argmin(distances))
        target_row, target_col = food_positions[nearest_index]

        # Decide which immediate turn best reduces the distance.
        # Priority:
        # 1) If the target is clearly to our left/right (bigger horizontal gap), turn that way.
        # 2) Otherwise, if the target is ahead (smaller row index), go forward.
        # 3) If the target is behind or same row, we still go forward (no backward option).
        row_diff = target_row - head_row  # negative means ahead (up)
        col_diff = target_col - head_col  # negative means to the left, positive to the right

        if abs(col_diff) > abs(row_diff):
            # Move horizontally toward the food
            if col_diff < 0:
                return LEFT  # -1 (turn left)
            elif col_diff > 0:
                return RIGHT  #  1 (turn right)
            else:
                # same column, fall back to forward decision below
                pass

        # Prefer going forward when target is ahead (row_diff < 0)
        if row_diff < 0:
            return FORWARD  # 0 (forward)

        # If food is behind or same row, we cannot go backward; keep going forward.
        return FORWARD  # 0 (forward)

    def AgentFunction(self, percepts):
        '''
        Replaced the placeholder with a very simple "go toward food" rule.

        - Percepts are a 7x7 Numpy array aligned to the snakes heading.
        - We look for food cells (value == -0.5) and choose an immediate action
          that reduces the distance to the nearest food:
            * -1 = turn left
            *  0 = forward
            *  1 = turn right
        - If no food is visible, we keep moving forward.
        '''
        return self._choose_action_towards_food(percepts)

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
         snake.foods - turns when food was eaten by the snake, not including biting other snake
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
         This fitness functions only considers the average snake size
        '''
        fitness[n] = meanSize

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

def newGeneration(old_population):

    '''
     This function must return a tuple consisting of:
     - a list of the new_population of snakes that is of the same length as the old_population,
     - the average fitness of the old population
    '''
    N = len(old_population)

    nPercepts = old_population[0].nPercepts
    actions = old_population[0].actions


    fitness = evalFitness(old_population)

    # Create new population list...
    new_population = list()
    for n in range(N):

        # Create a new snake
        new_snake = Snake(nPercepts, actions)

        '''
         Here you should modify the new snakes chromosome by selecting two parents (based on their
         fitness) and crossing their chromosome to overwrite new_snake.chromosome

         Consider implementing elitism, mutation and various other
         strategies for producing a new creature.

         .
         .
         .
        '''

        # Add the new snake to the new population
        new_population.append(new_snake)

    # At the end you need to compute the average fitness and return it along with your new population
    avg_fitness = np.mean(fitness)
    saveFitnessHistory(avg_fitness)
    return (new_population, avg_fitness)
