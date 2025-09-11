__author__ = "<your name>"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "<your e-mail>"

import numpy as np

agentName = "<my_agent>"
trainingSchedule = [("self", 1), ("random", 1)]

# This is the class for your snake/agent
class Snake:

    def __init__(self, nPercepts, actions):
        '''
         You should initialise self.chromosome member variable here (whatever you choose it
         to be - a list/vector/matrix of numbers - and initialise it with some random
         values)
        '''

        self.nPercepts = nPercepts
        self.actions = actions



    def AgentFunction(self, percepts):
        '''
         You should implement a neural network-based model here that translates from 'percepts' 
         to 'actions' with the weights and biases coming from 'self.chromosome', and the
         inputs to the network coming from the 'percepts' variable.
        
         Percepts are a 7x7 Numpy Matrix. 

         The return value must be an integer, a choice of one of possible actions 
         from [-1,0,1] corresponding to turning left, moving forward, and turning right.

         .
         .
         .
        ''' 

        index = 1 # Current code always chose the action to move forward
        return self.actions[index]

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

    return (new_population, avg_fitness)
