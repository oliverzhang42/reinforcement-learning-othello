MODE = 't' #Mode, 't' for train, 'h' for play vs human, and 'm' for play vs machine
PATH = None #Path, specify the folder which you want to save/read weights.
LR = 0.001 #Learning rate, only matters if you're training.
RANDOM = True #Determines whether there will be randomness in your training.
SAVE_FREQUENCY = 50 #After how many iterations do you want to save?
WEIGHTS = [-1] #The list of weights to load. If training or vs human 
                   #make the list one length. If play vs machine, you HAVE TO make it length 2.
TOTAL_EPISODES = 10000 #What's the total number of episodes do you want to run? Only used if training



from absl import app

from OthelloController import OthelloController
from OthelloAgainstAI import OthelloSession
from OthelloArena import Arena


def main(argv):
    del argv # Unused.
        
    if MODE == 't':
        # Train
        if RANDOM:
            epsilon = 5
        else:
            epsilon = 10000000
        
        controller = OthelloController(PATH, 1, learning_rate = LR, epsilon = epsilon)
        
        if(WEIGHTS[0] != -1):
            controller.load([WEIGHTS[0]])
        
        controller.main(WEIGHTS[0], TOTAL_EPISODES, 1, SAVE_FREQUENCY)
        
    elif MODE == 'h':
        session = OthelloSession(PATH)
        session.play(WEIGHTS[0])
        
    elif MODE == 'm':
        arena = Arena(PATH)
        arena.play(WEIGHTS[0], WEIGHTS[1])
        

if __name__ == '__main__':
    app.run(main)
