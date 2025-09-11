__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"
__date__ = "August 2025"

import numpy as np

agentName = "random"
trainingSchedule = None

class Snake:

    def __init__(self, nPercepts, actions):
        '''
         This agent doesn't evolve, and it doesn't have a chromosome.
         The pass statement is for no-op (replace it with your chromosome initialisation)
        '''
        self.actions = actions

    def AgentFunction(self, percepts):
        '''
          This agent ignores percepts and chooses random action.  Your agents should not
          perform random actions - your agents' actions should be deterministic from
          computation based on self.chromosome and percepts
        '''
        index = np.random.randint(low=0,high=len(self.actions))
        return self.actions[index]
