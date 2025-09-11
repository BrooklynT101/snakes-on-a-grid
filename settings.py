__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"
__date__ = "August 2025s"

# You can manipulate these defaults to change the game parameters.

game_settings = {

   #File implementing the agent playing as player 1
   "player1": "random_agent.py",

   # File implementing the agent playing as player 2
   "player2": "random_agent.py",

   # Size of the game grid
   "gridSize": 50,

   # Number of snakes on a plane
   "nSnakes": 40,

   # Number of turns per game
   "nTurns": 100,

   # Speed of visualisation ('slow','normal','fast')
   "visSpeed": 'normal',

   # Visualisation resolution
   "visResolution": (720, 480),

   # Set to True to save tournament games
   "saveTournamentGames": False,

   "seed": 0   # seed for game choices, None for random seed
}

if __name__ == "__main__":
    from snakes import SnakeGame
    import sys

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
