from ReversiAI import ReversiController

TOTAL_EPISODES = 10000000

#path = "/Users/student36/reinforcement-learning-othello/Weights_Folder3"
path = "/home/oliver/git/othello/reinforcement-learning-othello/Weights_Folder9/"

controller = ReversiController(path, False, False, population_size = 1,
                               learning_rate = 0.000003, epsilon = 5,
                               epsilon_increment = 0.001)
controller.load([0])
controller.population[0].depth = 3
controller.main(TOTAL_EPISODES)
