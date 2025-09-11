__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"
__date__ = "August 2025"
__version__ = 2.1

import importlib
import numpy as np
import traceback
import sys
import gzip, pickle
from datetime import datetime
import os
import signal
import time

maxTrainingEpochs = 500
maxActions = 3
numPlays = 5
fieldOfVision = 7
snakeStartingLength = 3
FOOD_CODE = -0.5

def alarm_handler(signum, frame):
    raise RuntimeError("Time out")

def percepts_global_to_agent_frame_of_reference(percepts,rotation):

    if rotation == 90:
        percepts = np.rot90(percepts, axes=[0, 1])
    elif rotation == 270:
        percepts = np.rot90(percepts, axes=[1, 0])
    elif rotation == 180:
        percepts = np.rot90(np.rot90(percepts,axes=[0,1]),axes=[0,1])

    return percepts

def actions_agent_to_global_shift(action, rotation):

    # 000
    # 020
    # 010   0       a=-1  [0,-1]; a=0 [-1,0]; a=1 [0,1]         0 (-1)  270   0 (1)  90

    # 000
    # 120
    # 000   90      a=-1  [-1,0]; a=0 [0,1]; a=1 [1,0]          90 (-1) 0     90 (1) 180

    # 010
    # 020
    # 000   180     a=-1  [0,1]; a=0 [1,0]; a=1 [0,-1]          180 (-1)

    # 000
    # 021
    # 000   270     a=-1  [1,0]; a=0 [0,-1]; a=1 [-1,0]

    if action == 1:
        if rotation == 0:
            return 0,1
        elif rotation == 90:
            return 1,0
        elif rotation == 180:
            return 0,-1
        else:
            return -1,0
    elif action == -1:
        if rotation == 0:
            return 0,-1
        elif rotation == 90:
            return -1,0
        elif rotation == 180:
            return 0,1
        else:
            return 1,0
    elif action == 0:
        if rotation == 0:
            return 1,0
        elif rotation == 90:
            return 0,1
        elif rotation == 180:
            return -1,0
        else:
            return 0,-1



# Class avatar is a wrapper for the agent with extra bits required
# for running the game
class Avatar:

    # Initialise avatar for an agent of a given player
    def __init__(self,agent,player):
        self.agent = agent
        self.player = player

    # Reset the avatar variables for a new game
    def reset_for_new_game(self,nTurns,snakeStartingLength):
        self.size = snakeStartingLength
        self.sizes = np.zeros((nTurns)).astype('uint32')
        self.friend_attacks = np.zeros((nTurns)).astype('int32')
        self.enemy_attacks = np.zeros((nTurns)).astype('int32')
        self.bitten = np.zeros((nTurns)).astype('int32')
        self.foods = np.zeros((nTurns)).astype('uint32')
        self.friend_crashes= np.zeros((nTurns)).astype('uint32')
        self.enemy_crashes= np.zeros((nTurns)).astype('uint32')
        self.dead = False
        self.body = []

    def update_size_stats(self,turn):
        if not self.dead:
            self.sizes[turn] = self.size

    # Execute AgentFunction that maps percepts to actions
    def action(self, turn, percepts):

        if self.player.game.in_tournament:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(1)

        try:
            self.player.exec.perceptFieldOfVision = self.player.fieldOfVision
            self.player.exec.perceptFrames = self.player.nFrames

            action = self.agent.AgentFunction(percepts)

        except Exception as e:
            if self.player.game.in_tournament:
                raise RuntimeError("Error! Failed to execute AgentFunction - %s" % str(e))
            else:
                print("Error! Failed to execute AgentFunction - %s" % str(e))
                traceback.print_exc()
                sys.exit(-1)

        if self.player.game.in_tournament:
            signal.alarm(0)

        if type(action) != int and type(action) != np.int64 and type(action) != np.int32 and type(action) != np.int8:
            if self.player.game.in_tournament:
                raise RuntimeError("Error! AgentFunction must return an integer")
            else:
                print("Error! AgentFunction must return an integer")
                traceback.print_exc()
                sys.exit(-1)

        if action not in [-1,0,1]:
            if self.player.game.in_tournament:
                raise RuntimeError("Error! The returned action must be an integer -1,0, or 1")
            else:
                print("Error! The returned action must be an integer -1,0, or 1, not %d" % action)
                traceback.print_exc()
                sys.exit(-1)

        return action

# Class player holds all the agents for a given player
class Player:

    def __init__(self, game, player, playerFile,emptyMode=False, jointname=False, atEpoch=None):

        self.game = game
        self.player = player
        this_scripts_folder = os.path.dirname(os.path.abspath(__file__))
        self.playerFileRaw = playerFile
        self.playerFile = os.path.join(this_scripts_folder, playerFile)
        self.nAgents = self.game.nAgents
        self.fitness = list()
        self.errorMsg = ""
        self.ready = False

        if emptyMode:
            return

        if not os.path.exists(self.playerFile):
            print("Error! Agent file '%s' not found" % self.playerFile)
            traceback.print_exc()
            sys.exit(-1)

        if len(playerFile) > 3 and playerFile[-3:].lower() == '.py':
            playerModule = playerFile[:-3]
        else:
            print("Error! Agent file %s needs a '.py' extension" % self.playerFile)
            traceback.print_exc()
            sys.exit(-1)

        # Import agent file as module
        if self.game.in_tournament:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(10)
        try:
            if self.game.in_tournament and playerModule != 'random_agent':
                self.exec = importlib.machinery.SourceFileLoader('my_agent', playerModule + '.py').load_module()
            else:
                self.exec = importlib.import_module(playerModule)

        except Exception as e:
            if self.game.in_tournament:
                signal.alarm(0)
                self.errorMsg = str(e)
                return
            else:
                print("Error! Failed to load '%s'" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        if self.game.in_tournament:
            signal.alarm(0)

        if hasattr(self.exec, 'agentName') and self.exec.agentName[0] != '<':
            self.name = self.exec.agentName
        else:
            if self.game.in_tournament and playerFile != 'random_agent.py':
                self.name = playerFile.split('/')[-2]
            else:
                self.name = playerFile

        if jointname and self.game.in_tournament:
           self.pname = playerFile.split('/')[-2]

        self.fieldOfVision = fieldOfVision

        self.nFrames = 1

        if not hasattr(self.exec,'trainingSchedule'):
            if self.game.in_tournament:
                signal.alarm(0)
                self.errorMsg = "Agent is missing the 'trainingSchedule' variable."
                return
            else:
                print("Error! Agent is missing the 'trainingSchedule' variable.")
                traceback.print_exc()
                sys.exit(-1)

        self.trainingSchedule = self.exec.trainingSchedule

        if self.trainingSchedule is not None and not isinstance(self.trainingSchedule,list):
            if self.game.in_tournament:
                signal.alarm(0)
                self.errorMsg = "Agent's 'trainingSchedule' should be a list of (str,int) tuples."
                return
            else:
                print("Error! Agent's 'trainingSchedule' should be a list of (str,int) tuples.")
                traceback.print_exc()
                sys.exit(-1)

        if isinstance(self.trainingSchedule, list):

            totTrainEpochs = 0

            for trainSession in self.trainingSchedule:
                if not isinstance(trainSession,tuple) or len(trainSession) < 2 or not (isinstance(trainSession[0],str)) or not isinstance(trainSession[1],int):
                    if self.game.in_tournament:
                        signal.alarm(0)
                        self.errorMsg = "Agent's 'trainingSchedule' should be a list containing (str,int) tuples."
                        return
                    else:
                        print("Error! Agent's 'trainingSchedule' should be a list containing (str,int) tuples.")
                        traceback.print_exc()
                        sys.exit(-1)

                if trainSession[1] < 0:
                    if self.game.in_tournament:
                        signal.alarm(0)
                        self.errorMsg = "Agent's 'trainingSchedule' should be a list of (str,int) tuples, where int corresponds to the number of train generations."
                        return
                    else:
                        print("Error! Agent's 'trainingSchedule' should be a list of (str,int) tuples, where int corresponds to the number of train generations.")
                        traceback.print_exc()
                        sys.exit(-1)

                    totTrainEpochs += trainSession[1]

            if totTrainEpochs > maxTrainingEpochs:
                if self.game.in_tournament:
                    signal.alarm(0)
                    self.errorMsg = "Agent's 'trainingSchedule' cannot specify more than %d training epochs in total." % maxTrainingEpochs
                    return
                else:
                    print("Error! Agent's 'trainingSchedule' cannot specify more than %d training epochs in total." % maxTrainingEpochs)
                    traceback.print_exc()
                    sys.exit(-1)

        if self.trainingSchedule is None or self.game.training == 'none':
            self.trained = True
        else:
            self.trained = False

        # Create the initial population of agents by creating
        # new instance of the agent using provided MyCreature class
        agentFile = self.playerFile

        if self.game.in_tournament and agentFile != 'random_agent':
            if self.game.training == 'pretrained':
                agentFileSave = "/".join(agentFile.split('/')[:-1] + ['my_agent'])
            else:
                agentFileSave = "/".join(agentFile.split('/')[:-1] + [agentFile.split('/')[-2]])
        else:
            agentFileSave = agentFile

        if atEpoch is None:
            self.savedAgent = ".".join(agentFileSave.split('.')[:-1]) + '.tar.gz'
        else:
            self.savedAgent = ".".join(agentFileSave.split('.')[:-1]) + f"_atgen{atEpoch:03d}.tar.gz"

        this_scripts_folder = os.path.dirname(os.path.abspath(__file__))
        self.savedAgent = os.path.join(this_scripts_folder, self.savedAgent)

        if self.game.training == 'none' or not os.path.exists(self.savedAgent) or (not self.game.in_tournament and os.path.getmtime(self.savedAgent) < os.path.getmtime(agentFile)):
            agents = list()
            for n in range(self.nAgents):
                if self.game.in_tournament:
                    signal.signal(signal.SIGALRM, alarm_handler)
                    signal.alarm(1)
                try:
                    agent = self.exec.Snake(nPercepts=self.fieldOfVision**2*self.nFrames, actions=[-1,0,1])
                except Exception as e:
                    if self.game.in_tournament:
                        signal.alarm(0)
                        self.errorMsg = str(e)
                        return
                    else:
                        print("Error! Failed to instantiate Snake() from '%s'" % self.playerFile)
                        traceback.print_exc()
                        sys.exit(-1)

                if self.game.in_tournament:
                    signal.alarm(0)
                agents.append(agent)
        else:
            with gzip.open(self.savedAgent,'r') as f:
                agents = pickle.load(f)
            self.trained = True

        # Convert list of agents to list of avatars
        try:
            self.agents_to_avatars(agents)
        except Exception as e:
            if self.game.in_tournament:
                signal.alarm(0)
                self.errorMsg = str(e)
                return
            else:
                print("Error! Failed to create a list of Snakes")
                traceback.print_exc()
                sys.exit(-1)

        self.ready = True

    # Convert list of agents to list of avatars
    def agents_to_avatars(self, agents):
        self.avatars = list()
        self.stats = list()

        for agent in agents:
            if type(agent) != self.exec.Snake:
                if self.game.in_tournament:
                    raise RuntimeError(
                        'Error! The new_population returned from newGeneration() must contain objects of Snake() type')
                else:
                    print("Error! The new_population returned form newGeneration() in '%s' must contain objects of Snake() type" %
                    self.playerFile)
                    traceback.print_exc()
                    sys.exit(-1)

            avatar = Avatar(agent,player=self)
            self.avatars.append(avatar)
            self.stats.append(dict())

    def avatar_to_agent_stats(self,avatar):
        agent = avatar.agent
        agent.sizes = avatar.sizes
        agent.friend_attacks = avatar.friend_attacks
        agent.enemy_attacks = avatar.enemy_attacks
        agent.bitten = avatar.bitten
        agent.foods = avatar.foods
        agent.friend_crashes= avatar.friend_crashes
        agent.enemy_crashes= avatar.enemy_crashes
        return agent

    # Get a new generation of agents
    def new_generation_agents(self,gen):

        # Record game stats in the agent objects
        old_population = list()
        for avatar in self.avatars:
            agent = self.avatar_to_agent_stats(avatar)
            old_population.append(agent)

        if self.playerFile != 'random_agent.py':
            msg = "  avg_fitness: "

            if self.game.in_tournament:
                self.game.train_report.append(msg)

            if self.game.verbose:
                sys.stdout.write(msg)
                sys.stdout.flush()

        # Get a new population of agents by calling
        # the provided newGeneration method
        if self.game.in_tournament:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(4)

        try:
            result = self.exec.newGeneration(old_population)
        except Exception as e:
            if self.game.in_tournament:
                raise RuntimeError('Error! Failed to execute newGeneration(), %s' % str(e))
            else:
                print("Error! Failed to execute newGeneration() from '%s', %s" % (self.playerFile, str(e)))
                traceback.print_exc()
                sys.exit(-1)

        if self.game.in_tournament:
            signal.alarm(0)

        if type(result) != tuple or len(result) != 2:
            if self.game.in_tournament:
                raise RuntimeError('Error! The returned value form newGeneration() must be a 2-item tuple')
            else:
                print("Error! The returned value form newGeneration() in '%s' must be a 2-item tuple" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        (new_population, fitness) = result

        if type(new_population) != list:
            if self.game.in_tournament:
                raise RuntimeError('Error! The new_population returned form newGeneration() must be a list')
            else:
                print("Error! The new_population returned form newGeneration() in '%s' must be a list" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        try:
            fitness = float(fitness)
        except Exception as e:
            if self.game.in_tournament:
                raise RuntimeError('Error! The fitness returned form newGeneration() must be float or int')
            else:
                print("Error! The new_population returned form newGeneration() in '%s' must be a float or int" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        if len(new_population) != len(old_population):
            if self.game.in_tournament:
                raise RuntimeError('Error! The new_population returned form newGeneration() must contain %d items' % self.nAgents)
            else:
                print("Error! The new_population returned form newGeneration() in '%s' must contain %d items" % (self.playerFile, self.nAgents))
                traceback.print_exc()
                sys.exit(-1)

        if self.playerFile != 'random_agent.py':
            msg = " %.2e" % fitness
            if self.game.in_tournament:
                self.game.train_report.append(msg)

            if self.game.verbose:
                sys.stdout.write(msg)
                sys.stdout.flush()

        self.fitness.append(fitness)

        # Convert agents to avatars
        self.agents_to_avatars(new_population)

    def evaluate_fitness(self):

        agents = []
        for avatar in self.avatars:
            agent = self.avatar_to_agent_stats(avatar)
            agents.append(agent)

        try:
            fitness = self.exec.evalFitness(agents)
        except:
            if self.game.in_tournament:
                raise RuntimeError("Error! Failed to execute evalFitness() from '%s'" % self.playerFile)
            else:
                print("Error! Failed to execute evalFitness() from '%s'" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        if isinstance(fitness,np.ndarray):
            fitness = fitness.tolist()

        if not isinstance(fitness, list):
            if self.game.in_tournament:
                raise RuntimeError("Error! Function evalFitness() from '%s' must return a list" % self.playerFile)
            else:
                print("Error! Function evalFitness() from '%s' must return a list" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        if len(fitness) != len(agents):
            if self.game.in_tournament:
                raise RuntimeError(
                    "Error! Length of the list returned by evalFitness() from '%s' is %d; expecting the length to be %d." % (
                    self.playerFile, len(fitness), len(agents)))
            else:
                print(
                    "Error! Length of the list returned by evalFitness() from '%s' is %d; expecting the length to be %d." % (
                    self.playerFile, len(fitness), len(agents)))
                traceback.print_exc()
                sys.exit(-1)

        self.fitness.append(np.mean(fitness))
        I = np.argsort(fitness)[::-1]
        self.avatars = np.array(self.avatars)[I].tolist()
        msg = "  avg_fitness:  %.2e\n\n" % np.mean(fitness)
        if self.game.in_tournament:
            self.game.train_report.append(msg)

        if self.game.verbose:
            sys.stdout.write(msg)
            sys.stdout.flush()


    def save_trained(self, atEpoch=None):

        if atEpoch is None:
            savedAgent = self.savedAgent
        else:
            savedAgent = ".".join(self.savedAgent.split(".")[:-2]) + f"_atgen{atEpoch:03d}.tar.gz"

        if atEpoch is None and self.game.verbose:
            sys.stdout.write("Saving last generation agents to %s..."  % self.savedAgent)
            sys.stdout.flush()
        agents = []
        for avatar in self.avatars:
            agents.append(avatar.agent)

        with gzip.open(savedAgent,'w') as f:
            pickle.dump(agents, f)

        if atEpoch is None and self.game.verbose:
            sys.stdout.write("done\n")
            sys.stdout.flush()


class SnakePlay:

    def __init__(self,game,showGame=None,saveGame=False):
        self.game = game
        self.map = np.zeros((self.game.gridSize, self.game.gridSize), dtype='float')
        self.showGame = showGame
        self.saveGame = saveGame

        if self.saveGame:
            self.vis_map = np.zeros((self.game.gridSize, self.game.gridSize, 3, self.game.nTurns+1), dtype='int8')
        elif self.showGame is not None:
            self.vis_map = np.zeros((self.game.gridSize, self.game.gridSize, 3, 1), dtype='int8')


    def vis_update(self,i,players,food):

        if not self.saveGame:
            self.vis_map *= 0
            i = 0

        for k,player in enumerate(players):
            for avatar in player.avatars:
                if avatar.dead:
                    continue

                j = 1
                if avatar.dead:
                    j = -1

                for y,x in avatar.body:
                    self.vis_map[y,x,k,i] = 1*j

                y,x = avatar.head
                self.vis_map[y,x,k,i] = 2*j

        for y,x in food:
            self.vis_map[y, x, 2, i] = 1

        return self.vis_map[:,:,:,i]

    def manhattan_distance(self, x1,y1,x2,y2):
        x = np.min([np.abs(x1-x2),np.abs(x2-x1)])
        y = np.min([np.abs(y1-y2),np.abs(y2-y1)])

        return x+y


    def play(self,players):

        nRegions = self.game.gridSize//self.game.regionSize
        regions = []
        for y in range(nRegions):
            for x in range(nRegions):
                regions.append((x,y))

        I = self.game.rnd_fixed_seed.permutation(len(regions))
        regions= np.array(regions)[I]
        regions = [tuple(r) for r in regions.tolist()]

        self.food = []
        self.snake = [[],[]]
        for i in range(self.game.nAgents):
            if len(regions) < 2:
                print(f"Error! Not enough regions available.")
                sys.exit(-1)

            yr,xr = regions[-1]
                
            while nRegions - yr - 1 == yr and nRegions - xr -1 == xr:
                regions = regions[:-1]
                yr,xr = regions[-1]

            for k,player in enumerate(players):

                if k>0:
                    yr = nRegions - yr - 1
                    xr = nRegions - xr - 1

                yh = yr*self.game.regionSize+self.game.regionSize//2
                xh = xr*self.game.regionSize+self.game.regionSize//2

                if k==0:
                    rotation = self.game.rnd_fixed_seed.choice([0,90,180,270])
                else:
                    rotation = rotation-180
                    if rotation < 0:
                        rotation += 360

                if rotation==0:
                    jy = 1
                    jx = 0
                elif rotation ==90:
                    jy = 0
                    jx = -1
                elif rotation==180:
                    jy = -1
                    jx = 0
                else:
                    jy = 0
                    jx = 1

                if k==0:
                    j = 1
                else:
                    j = -1

                avatar = player.avatars[i]
                avatar.reset_for_new_game(self.game.nTurns,self.game.snakeStartingLength)
                offset = self.game.regionSize//2-avatar.size
                if avatar.size % 2 == 1:
                    offset += 1

                for n,z in enumerate(range(offset, offset+avatar.size)):
                    y = (yh+jy*z)%self.game.gridSize
                    x = (xh+jx*z)%self.game.gridSize
                    if n == 0:
                        self.map[y,x] = (avatar.size)*j
                        avatar.head = (y, x)
                    else:
                        self.map[y,x] = j
                    #self.map[y,x] = (avatar.size-n)*j
                    avatar.body.append((y, x))
                    self.snake[k].append((y, x))

                avatar.rotation = rotation

                # Remove (yr,xr) from regions
                regions.remove((yr,xr))
        
        food_choices = []
        for y in range(self.game.gridSize):
            for x in range(self.game.gridSize):
                if self.map[y,x] == 0:
                    food_choices.append((y,x))
        
        I = self.game.rnd_fixed_seed.permutation(len(food_choices))
        food_choices = np.array(food_choices)[I].tolist()
        food_choices = [tuple(r) for r in food_choices]

        for n in range(self.game.nFoods):
            y, x = food_choices[-1]
            yr = self.game.gridSize - y - 1
            xr = self.game.gridSize - x - 1
            if (yr, xr) not in food_choices:
                food_choices = food_choices[:-1]
                continue

            self.food.append((y, x))
            self.map[y,x] = FOOD_CODE
            self.food.append((yr, xr))
            self.map[yr,xr] = FOOD_CODE

            food_choices.remove((y, x))
            food_choices.remove((yr, xr))

        if self.showGame is not None:
            vis_map = self.vis_update(0,players,self.food)
            self.game.vis.show(vis_map, turn=0, titleStr=self.showGame)
        elif self.saveGame:
            self.vis_update(0,players,self.food)

        # Play the game over a number of turns
        for turn in range(self.game.nTurns):


            food_eaten = []

            # Create new agent map based on actions
            #new_agent_map = np.ndarray((self.gridSize,self.gridSize), dtype=object)

            # 000
            # 020
            # 010   0       a=-1  [0,-1]; a=0 [-1,0]; a=1 [0,1]         0 (-1)  270   0 (1)  90

            # 000
            # 120
            # 000   90      a=-1  [-1,0]; a=0 [0,1]; a=1 [1,0]          90 (-1) 0     90 (1) 180

            # 010
            # 020
            # 000   180     a=-1  [0,1]; a=0 [1,0]; a=1 [0,-1]          180 (-1)

            # 000
            # 021
            # 000   270     a=-1  [1,0]; a=0 [0,-1]; a=1 [-1,0]


            # Get actions of the agents
            # Reset avatars for a new game
            for k, player in enumerate(players):

                for avatar in player.avatars:

                    if avatar.dead:
                        continue

                    gameDone = False

                    # Percepts
                    percepts = np.zeros((avatar.player.fieldOfVision,avatar.player.fieldOfVision)).astype('float')

                    pBHalf = avatar.player.fieldOfVision // 2

                    if k==0:
                        jk=1
                    else:
                        jk=-1

                    # Add nearby agents to percepts
                    for i,io in enumerate(range(-pBHalf,pBHalf+1)):
                        for j,jo in enumerate(range(-pBHalf,pBHalf+1)):
                            y = (avatar.head[0] + io)
                            if y < 0:
                                y += self.game.gridSize
                            elif y >= self.game.gridSize:
                                y -= self.game.gridSize

                            x = (avatar.head[1] + jo)
                            if x < 0:
                                x += self.game.gridSize
                            elif x >= self.game.gridSize:
                                x -= self.game.gridSize

                            percepts[i,j] = self.map[y,x]

                    percepts = percepts_global_to_agent_frame_of_reference(percepts,avatar.rotation)

                    # Get action from agent
                    try:
                        action = avatar.action(turn+1,percepts)
                    except Exception as e:
                        action = avatar.action(turn + 1, percepts)
                        if self.game.in_tournament:
                            self.game.game_scores[k] += [-500]
                            self.game.game_messages[k] = str(e)
                            self.game.game_play = False
                        else:
                            traceback.print_exc()
                            sys.exit(-1)

                    if not self.game.game_play:
                        break

                    y, x = actions_agent_to_global_shift(action,avatar.rotation)

                    #x = avatar.position[0]
                    #y = avatar.position[1]

                    # 000
                    # 020
                    # 010   0       a=-1  [0,-1]; a=0 [-1,0]; a=1 [0,1]         0 (-1)  270   0 (1)  90

                    # 000
                    # 120
                    # 000   90      a=-1  [-1,0]; a=0 [0,1]; a=1 [1,0]          90 (-1) 0     90 (1) 180

                    # 010
                    # 020
                    # 000   180     a=-1  [0,1]; a=0 [1,0]; a=1 [0,-1]          180 (-1)

                    # 000
                    # 021
                    # 000   270     a=-1  [1,0]; a=0 [0,-1]; a=1 [-1,0]

                    # Action 0 is move left
                    if avatar.rotation == 0 or avatar.rotation==180:
                        if action == -1:
                           yd = 0
                           xd = -1
                        elif action == 0:
                           yd = -1
                           xd = 0
                        elif action == 1:
                           yd = 0
                           xd = 1
                    else:
                        if action == -1:
                           yd = -1
                           xd = 0
                        elif action == 0:
                           yd = 0
                           xd = 1
                        elif action == 1:
                           yd = 1
                           xd = 0

                    if avatar.rotation >= 180:
                        xd *= -1
                        yd *= -1

                    avatar.rotation += action*90
                    if avatar.rotation < 0:
                        avatar.rotation += 360
                    avatar.rotation %= 360

                    y, x = avatar.head

                    y += yd
                    x += xd

                    if y < 0:
                        y += self.game.gridSize
                    if x < 0:
                        x += self.game.gridSize

                    y %= self.game.gridSize
                    x %= self.game.gridSize

                    if (y,x) == (33,22):
                        pass

                    avatar.new_head = (y,x)

                    if (y,x) in self.food:
                        avatar.foods[turn] = 1
                        avatar.new_size = avatar.size + 1
                        if (y,x) not in food_eaten:
                            food_eaten += [(y,x)]
                    elif (y,x) in self.snake[k]:
                        avatar.new_size = avatar.size + 1
                        avatar.friend_attacks[turn] = 1
                    elif (y,x) in self.snake[(k+1)%2]:
                        avatar.new_size = avatar.size + 1
                        avatar.enemy_attacks[turn] = 1
                    else:
                        avatar.new_size = avatar.size

                    new_body = [avatar.new_head]
                    for s in range(avatar.new_size-1):
                        try:
                            new_body.append(avatar.body[s])
                        except:
                            raise(Exception("Error while updating avatar body"))
                    avatar.new_body = new_body

            if not self.game.game_play:
                return None

            all_avatars = []
            for k, player in enumerate(players):
                for avatar in player.avatars:
                    if avatar.dead:
                        continue
                    
                    avatar.head = avatar.new_head
                    avatar.body = avatar.new_body
                    avatar.size = avatar.new_size

                    all_avatars += [avatar]

            for i in range(len(all_avatars)):

                head_collisions = None
                collider_sizes = None
                for j in range(i+1,len(all_avatars)):

                    if all_avatars[i].head == all_avatars[j].head:
                        if all_avatars[i].player.player == all_avatars[j].player.player:
                            all_avatars[i].friend_crashes[turn] = 1
                            all_avatars[j].friend_crashes[turn] = 1
                        else:
                            all_avatars[i].enemy_crashes[turn] = 1
                            all_avatars[j].enemy_crashes[turn] = 1
                        if head_collisions is None:
                            head_collisions = [all_avatars[i]]
                            collider_sizes = [all_avatars[i].size]
                        head_collisions.append(all_avatars[j])
                        collider_sizes.append(all_avatars[j].size)

                if head_collisions is not None:
                    collider_sizes = np.array(collider_sizes)
                    head_collisions = np.array(head_collisions)
                    I = np.argsort(collider_sizes)
                    collider_sizes = collider_sizes[I[::-1]]
                    head_collisions = head_collisions[I[::-1]]
                    collider_sizes -= collider_sizes[1]

                    for k in range(len(head_collisions)):
                        avatar = head_collisions[k]
                        if collider_sizes[k] <= 0:
                            avatar.dead = True
                        else:
                            avatar.body = avatar.body[:collider_sizes[k]]
                            avatar.size = collider_sizes[k]
                
            # Resolve eating
            for i in range(len(all_avatars)):
                if all_avatars[i].dead:
                    continue

                eaten_at = []
                for k in np.arange(len(all_avatars[i].body))[1:][::-1]:
                 
                   coord = all_avatars[i].body[k]

                   for j in range(len(all_avatars)):

                        if all_avatars[j].dead:
                            continue

                        if coord == all_avatars[j].head:
                            eaten_at.append(k)

                if len(eaten_at) > 0:
                    all_avatars[i].bitten[turn] = len(eaten_at)
                    k1 = eaten_at[-1]
                    for k in range(k1,len(all_avatars[i].body)):
                        if k not in eaten_at:
                            self.food.append(all_avatars[i].body[k])
                        all_avatars[i].size -= 1
                    all_avatars[i].body = all_avatars[i].body[:k1]
                    if all_avatars[i].size < 1:
                        raise ValueError("Avatar size should not be less than 1")

            food = []

            self.map *= 0
            for coord in self.food:
                if coord not in food and coord not in food_eaten:
                    food.append(coord)  
                    y,x = coord
                    self.map[y,x] = FOOD_CODE

            self.food = food
            gameDone = True

            for k, avatar in enumerate(all_avatars):

                if avatar.dead:
                    continue

                gameDone = False

                #if k==0:
                #    j=1
                #else:
                #    j=-1
                j = 1-2*avatar.player.player

                size = avatar.size
                avatar.sizes[turn] = size
                for s in range(len(avatar.body)):
                    y,x = avatar.body[s]

                    if s==0:
                        self.map[y,x] = size*j
                    else:
                        self.map[y,x] = j

            self.snake = [[],[]]
            for k, player in enumerate(players):
                for avatar in player.avatars:
                    for b in avatar.body:
                        self.snake[k].append(b)


            if gameDone:
                break

            if self.showGame is not None:
                i = 0

                if self.saveGame:
                    i = turn+1
                vis_map = self.vis_update(i,players,self.food)
                self.game.vis.show(vis_map, turn=turn+1, titleStr=self.showGame)
                #self.game.vis.show(self.map, self.food, heads1+colheads1, heads2+colheads2, turn=self.turn+1, titleStr=self.showGame, collisions=collisions)
            elif self.saveGame:
                self.vis_update(turn+1,players,self.food)
            #needUpdate = False
            for k, player in enumerate(players):
                for avatar in player.avatars:
                    if avatar.dead:
                        continue

            self.turn = turn

        if self.saveGame:
            if self.game.in_tournament:
                savePath = "/".join(self.game.players[0].playerFile.split('/')[:-1])
            else:
                this_scripts_folder = os.path.dirname(os.path.abspath(__file__))
                savePath = os.path.join(this_scripts_folder, "saved")
            if not os.path.isdir(savePath):
                os.makedirs(savePath, exist_ok=True)

            now = datetime.now()
            # Month abbreviation, day and year
            saveStr = now.strftime("%b-%d-%Y-%H-%M-%S")
            if len(players) == 1:
                saveStr += "-%s" % (players[0].name)
                name2 = None
            else:
                saveStr += "-%s-vs-%s" % (players[0].name, players[1].name)
                name2 = players[1].name

            if self.game.in_tournament:
                saveStr += '_%s' % self.game.training

            saveStr += ".pickle.gz"

            saveFile = os.path.join(savePath, saveStr)

            #print(f"Saving game to {saveFile}...", end='')
            #sys.stdout.flush()

            self.game.game_saves.append(saveFile)

            with gzip.open(saveFile, 'w') as f:
                pickle.dump((players[0].name, name2, self.vis_map), f)

            #print("done.")


        scores = []
        for k, player in enumerate(players):
            scores.append(0)
            for avatar in player.avatars:
                if avatar.dead:
                    continue

                scores[-1] += avatar.size

        if len(scores) == 1:
            return scores[0]
        else:
            return scores[0]-scores[1]



# Class that runs the entire game
class SnakeGame:

    # Initialises the game
    def __init__(self, gridSize, nTurns, nFoods, nAgents,                 snakeStartingLength=snakeStartingLength, regionSize=5, saveFinalGames=True,seed=None, tournament=False, verbose=True, training='trained'):

        if gridSize < regionSize:
            print(f"Error! Invalid setting {gridSize} for gridSize.  Must be at least {regionSize}" )
            sys.exit(-1)

        gridRegions = gridSize // regionSize
        gridRegions = gridRegions**2

        if gridRegions < nAgents*2:
            print(f"Error! Invalid setup with gridSize={gridSize} and nSnakes={nAgents} settings.")
            minGridSize = int(np.ceil(np.sqrt(nAgents*2)))*regionSize
            maxSnakes = gridRegions//2
            print(f"Either increase gridSize to {minGridSize} or reduce nSnakes to {maxSnakes}")

            sys.exit(-1)


        self.rnd = np.random.RandomState()
        self.gridSize = gridSize
        self.regionSize = regionSize
        self.nTurns = nTurns
        self.nActions = 3
        self.game_play = True
        self.in_tournament = tournament
        self.nFoods = nFoods
        self.nAgents = nAgents
        self.snakeStartingLength = snakeStartingLength
        self.saveFinalGames = saveFinalGames
        self.rnd_fixed_seed = np.random.RandomState(seed)
        self.verbose = verbose
        self.training = training

        if self.in_tournament:
            self.train_report = []
            self.game_report = []

    # Update the stats for the visualiser
    def update_vis_agents(self,players,creature_state):
        for p in range(2):
            for n in range(self.nAgents):
                i = n + p * self.nAgents
                avatar = players[p].avatars[n]

                creature_state[i, 0] = avatar.position[0]
                creature_state[i, 1] = avatar.position[1]
                creature_state[i, 2] = avatar.alive
                creature_state[i, 3] = p
                creature_state[i, 4] = avatar.size

    # Run the game
    def run(self,player1File, player2File,visResolution=(720,480), visSpeed='normal',savePath="saved",
            trainers=[("random_agent.py","random")],runs = [1,2,3,4,5], shows = [1,2,3,4,5],jointname=False):


        if visSpeed not in ['normal', 'fast','slow']:
            print(f"Error! Invalid setting {visSpeed} for visualisation speed.  Valid choices are 'slow','normal',fast'")
            sys.exit(-1)

        self.players = list()

        self.game_messages = ['', '']
        self.game_scores = [[],[]]
        self.game_saves = list()

        # Load player 1
        if player1File is not None:
            try:
                self.players.append(Player(self,len(self.players),player1File,jointname=jointname))
            except Exception as e:
                if self.in_tournament:
                    self.players.append(Player(self,0,player1File,self.nAgents,emptyMode=True))
                    self.game_messages[0] = "Error! Failed to create a player with the provided code"
                else:
                    print('Error! ' + str(e))
                    sys.exit(-1)

            if not self.players[0].ready:
                self.game_scores[0].append(-500)
                if self.players[0].errorMsg != "":
                    self.game_messages[0] = self.players[0].errorMsg

                self.game_play = False
            elif not self.players[0].trained:
                self.players[0] = self.train(self.players[0],visResolution,visSpeed,savePath,trainers)
                if self.players[0] is None:
                    self.game_scores[0].append(-500)
                    self.game_play = False

            # Load player 2
        if player2File is not None:
            try:
                self.players.append(Player(self,len(self.players),player2File,jointname=jointname))
            except Exception as e:
                if self.in_tournament:
                    self.players.append(Player(self,1,player2File,emptyMode=True))
                    self.game_messages[1] = "Error! Failed to create a player with the provided MyAgent.py code"
                else:
                    print('Error! ' + str(e))
                    sys.exit(-1)

            if not self.players[1].ready:
                self.game_scores[1].append(-500)
                if self.players[1].errorMsg != "":
                    self.game_messages[1] = self.players[1].errorMsg

                self.game_play = False
            elif not self.players[1].trained:
                self.players[1] = self.train(self.players[1],visResolution,visSpeed,savePath)
                if self.players[1] is None:
                    self.game_scores[1].append(-500)
                    self.game_play = False

        if not self.game_play:
            return





        if self.saveFinalGames:
            saves = shows
        else:
            saves = []

        self.play(self.players,runs,shows,saves,visResolution,visSpeed,savePath)


    def train(self,player,visResolution=(720,480), visSpeed='normal',savePath="saved",
              trainers=[("random","randomPlayer"), ("hunter","hunterPlayer")]):

        playerNumber = player.player
        trainingSchedule = player.trainingSchedule

        tot_gens = 0
        for op, gens in trainingSchedule:
            tot_gens += gens


        if tot_gens > maxTrainingEpochs:
            tot_gens = maxTrainingEpochs

        gens_count = 0

        for op, gens in trainingSchedule:

            if gens_count + gens > tot_gens:
                gens = tot_gens - gens_count

            if gens==0:
                break

            if op == 'random':
                opFile = 'random_agent.py'
            elif op == 'self':
                opFile = player.playerFileRaw
            else:
                opFile = op

            opponentNumber = (player.player + 1) % 2

            # Load opponent
            players = [player]

            if op == 'self':
                msg = "\nTraining %s against self for %d generations...\n" % (player.name, gens)
                if self.in_tournament:
                    self.train_report.append(msg)

                if self.verbose:
                    sys.stdout.write(msg)

                player.save_trained(atEpoch=gens_count)

                try:
                    opponent = Player(self, opponentNumber, playerFile=opFile, atEpoch=gens_count)


                    if os.path.exists(opponent.savedAgent):
                        os.remove(opponent.savedAgent)

                except Exception as e:
                    if self.in_tournament:
                        self.game_messages[playerNumber] = "Error! Failed to create opponent '%s' in training" % op
                        return None
                    else:
                        print('Error! ' + str(e))
                        sys.exit(-1)

                players.append(opponent)
            elif op is not None:
                try:
                    opponent = Player(self, opponentNumber, playerFile=opFile)
                except Exception as e:
                    if self.in_tournament:
                        self.game_messages[playerNumber] = "Error! Failed to create opponent '%s' in training" % op
                        return None
                    else:
                        print('Error! ' + str(e))
                        sys.exit(-1)

                if not opponent.ready:
                    self.game_scores[player.player].append(-500)
                    if player.errorMsg != "":
                        self.game_messages[player.player] = player.errorMsg
                    return None

                msg = "\nTraining %s against %s for %d generations...\n" % (player.name, op, gens)
                if self.in_tournament:
                    self.train_report.append(msg)

                if self.verbose:
                    sys.stdout.write(msg)
                players.append(opponent)
            else:
                msg = "\nTraining %s in single-player mode for %d generations...\n" % (player.name, gens)
                if self.in_tournament:
                    self.train_report.append(msg)

                if self.verbose:
                    sys.stdout.write(msg)
            msg = "------"
            if self.in_tournament:
                self.train_report.append(msg)

            if self.verbose:
                sys.stdout.write(msg)


            self.play(players,[], [], [], visResolution, visSpeed, savePath, trainGames=(gens,gens_count,tot_gens))

            #if opFile == player.playerFile:
            #    if self.game_scores[player.player] > self.game_scores[opponentNumber]:
            #        save_player = player
            #    else:
            #        save_player = opponent
            #else:
            #    save_player = player


            if not self.game_play:
                return None

            gens_count += gens
            if gens_count >= tot_gens:
                break

            #try:
            #    save_player.save_trained(train_temp)
            #except Exception as e:
            #    if self.in_tournament:
            #        self.game_messages[playerNumber] = "Error! Failed to save training results."
            #        return None
            #    else:
            #        traceback.print_exc()
            #        sys.exit(-1)

        try:
            player.save_trained()
        except Exception as e:
            if self.in_tournament:
                self.game_messages[playerNumber] = "Error! Failed to save training results."
                return None
            else:
                traceback.print_exc()
                sys.exit(-1)

        return player

    def play(self,players, run_games, show_games, save_games, visResolution=(720,480), visSpeed='normal',savePath="saved",trainGames=None):

        if len(show_games)>0:
            import vis_pygame as vis
            playerStrings = []
            for p in players:
                playerStrings += [p.name]

            if len(players) > 1 and hasattr(self.players[0],'pname') and  hasattr(self.players[1],'pname'):
                for p in players:
                    playerStrings += [p.pname]

            self.vis = vis.visualiser(speed=visSpeed,playerStrings=playerStrings,
                                  resolution=visResolution)

        if trainGames is None:
            nRuns = len(run_games)
        else:
            gens, gens_count, tot_gens = trainGames
            nRuns = gens

        # Play the game a number of times
        for game in range(1, nRuns + 1):
            if trainGames is None:
                if len(players)==1:
                    if game==1:
                        msg = "\nTournament (single-player mode) %s!" % (players[0].name)
                        if self.in_tournament:
                            self.game_report.append(msg)

                        if self.verbose:
                            sys.stdout.write(msg)
                else:
                    if game==1:
                        msg = "\nTournament %s vs. %s!!!" % (players[0].name, players[1].name)
                        if self.in_tournament:
                            self.game_report.append(msg)

                        if self.verbose:
                            sys.stdout.write(msg)
                msg = "\n    Game %d..." % (game)
                if self.in_tournament or hasattr(self,'vexcept'):
                    self.game_report.append(msg)

                if self.verbose or hasattr(self,'vexcept'):
                    sys.stdout.write("\n  Game %d..." % (game))

            else:
                msg = "\n  Gen %3d/%d..." % (game+gens_count, tot_gens)
                if self.in_tournament:
                    self.train_report.append(msg)

                if self.verbose:
                    sys.stdout.write(msg)

            if trainGames is None and game in show_games:
                showGame = "Snakes on a plane!"
                if len(self.players) > 1 and hasattr(self.players[0],'pname') and hasattr(self.players[1],'pname'):
                    showGame = '%s vs %s, game %d' % (self.players[0].pname, self.players[1].pname, game)
            else:
                showGame = None

            if trainGames is None and game in save_games:
                saveGame = True
            else:
                saveGame = False

            sgame = SnakePlay(self,showGame,saveGame)
            gameResult = sgame.play(players)

            if gameResult is None:
                if self.in_tournament:
                    return
                else:
                    print("Error! No game result!")
                    traceback.print_exc()
                    sys.exit(-1)

            if trainGames is None:
                score = gameResult
                if len(players) > 1:
                    if score>0:
                        if hasattr(self.players[0],'pname'):
                            msg = "won by %s with" % (players[0].pname)
                        else:
                            msg = "won by %s (orange) with" % (players[0].name)

                        if self.in_tournament:
                            self.game_report.append(msg)

                        if self.verbose or hasattr(self,'vexcept'):
                            sys.stdout.write(msg)
                    elif score<0:
                        if hasattr(self.players[1],'pname'):
                            msg = "won by %s with" % (players[1].pname)
                        else:
                            msg = "won by %s (purlpe) with" % (players[1].name)
                        if self.in_tournament:
                            self.game_report.append(msg)

                        if self.verbose or hasattr(self,'vexcept'):
                            sys.stdout.write(msg)
                            #score *= -1
                    else:
                        msg = "tied with"
                        if self.in_tournament:
                            self.game_report.append(msg)

                        if self.verbose or hasattr(self,'vexcept'):
                            sys.stdout.write(msg)

                msg = " score=%02d after %d turn" % (np.abs(score),sgame.turn+1)
                if sgame.turn!=0:
                    msg += "s"
                msg += "."

                if self.in_tournament:
                    self.game_report.append(msg)

                if self.verbose or hasattr(self,'vexcept'):
                    sys.stdout.write(msg)
                    sys.stdout.flush()

                self.game_scores[0].append(score)
                if len(players) > 1:
                    self.game_scores[1].append(-score)

            else:
                #for avatar in players[0].avatars:
                #    avatar.average_gen_stats()

                try:
                    if game + gens_count < tot_gens:
                        players[0].new_generation_agents(game+gens_count)
                    else:
                        players[0].evaluate_fitness()
                except Exception as e:
                    if self.in_tournament:
                        self.game_scores[p].append(-500)
                        self.game_messages[p] = str(e)
                        self.game_play = False
                    else:
                        traceback.print_exc()
                        sys.exit(-1)

                #vis_map = sgame.vis_map[:,:,:sgame.turn + 2]
                #vis_fh = sgame.vis_fh[:, :sgame.turn + 2]

        if len(show_games) > 0:
            time.sleep(5)
            del self.vis
            self.vis = None
            #time.sleep(2)

    # Play visualisation of a saved game
    @staticmethod
    def load(loadGame,visResolution=(720,480), visSpeed='normal'):
        import vis_pygame as vis

        if not os.path.isfile(loadGame):
            print("Error! Saved game file '%s' not found." % loadGame)
            sys.exit(-1)

        # Open the game file and read data
        try:
            with gzip.open(loadGame) as f:
              (player1Name,player2Name,vis_map) = pickle.load(f)
        except:
            print("Error! Failed to load %s." % loadGame)

        playerStrings = [player1Name]
        if player2Name is not None:
            playerStrings += [player2Name]

        # Create an instance of visualiser
        v = vis.visualiser(speed=visSpeed, playerStrings=playerStrings,resolution=visResolution)

        # Show visualisation
        titleStr = "Snakes on a plane! %s" % os.path.basename(loadGame)
        for t in range(vis_map.shape[3]):
            v.show(vis_map[:,:,:,t], turn=t, titleStr=titleStr)


def main(argv):
    # Load the defaults
    from settings import game_settings

    # Create a new game and run it
    g = SnakeGame(gridSize=game_settings['gridSize'],
                nTurns=game_settings['nTurns'],nFoods=2*game_settings['nSnakes'],
                nAgents=game_settings['nSnakes'],
                saveFinalGames=game_settings['saveTournamentGames'],
                seed=game_settings['seed'])

    if not 'player1' in game_settings or not 'player2' in game_settings:
        print("Error! Both players must be specified in settings.py.")
        sys.exit(-1)

    g.run(game_settings['player1'],
          game_settings['player2'],
          visResolution=game_settings['visResolution'],
          visSpeed=game_settings['visSpeed'])


if __name__ == "__main__":
   main(sys.argv[1:])

