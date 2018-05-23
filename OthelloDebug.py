from MishaCanvas import *
from tkinter import *
from reversi import reversiBoard
from ReversiAI import *
import AlphaBeta
import time
from othelloBoard import Board
from copy import deepcopy
import time

#path = "/Users/student36/reinforcement-learning-othello/Weights_Folder2/"
path = "/home/oliver/git/othello/reinforcement-learning-othello/Weights_Folder4/"

controller = ReversiController(path, False, False, 2, epsilon = 10000)
controller.load([35000, 0])
controller.population[1] = RandomPlayer()
controller.population[0].depth = 2

wins = [0,0]

for i in range(100):
    winner = controller.play_two_ai(0, 1)

    if(winner == -1):
        wins[1] += 1
    if(winner == 1):
        wins[0] += 1

    print("Wins by the first AI: " + str(wins[0]))
    print("Wins by the second AI: " + str(wins[1]))
