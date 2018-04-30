from ReversiAI import ReversiController

TOTAL_EPISODES = 10000000

#path = "/Users/student36/reinforcement-learning-othello/"
path = "/home/oliver/git/othello/reinforcement-learning-othello/Weights_Folder1/"

controller = ReversiController(path, False, False, 1)
#controller.load([0])
controller.main(TOTAL_EPISODES)
