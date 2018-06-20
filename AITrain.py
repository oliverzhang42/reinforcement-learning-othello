from ReversiAI import ReversiController

TOTAL_EPISODES = 10000000

#path = "/Users/student36/reinforcement-learning-othello/Weights_Folder3"
path = "/home/oliver/git/othello/reinforcement-learning-othello/Weights_Folder2/"

controller = ReversiController(path, False, False, population_size = 1,
                               learning_rate = 0.00000003, epsilon = 10,
                               epsilon_increment = 0.0001)
controller.load([0])

controller.population[0].depth = 1
controller.main(TOTAL_EPISODES)
