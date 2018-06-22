from MishaCanvas import *
from tkinter import *
from reversi import reversiBoard
from ReversiAI import *
import AlphaBeta
import time
from othelloBoard import Board
from copy import deepcopy
import time

#path = "/Users/student36/reinforcement-learning-othello/Weights_Folder5/"
path = "/home/oliver/git/othello/reinforcement-learning-othello/Folder/"

controller = ReversiController(path, False, False, 24, epsilon = 10000)
controller.load([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
#controller.population[1] = RandomPlayer()
controller.population[0].depth = 3
#controller.population[1].depth = 3

win = [[0 for i in range(24)] for j in range(24)]

#win[0][13] means 0 and 13 played and 0 won, 13 winning is win[13][0]

for i in range(24):
    for j in range(i+1, 24):
        winner = controller.play_two_ai(i,j)
        
        if(winner == 1):
            win[i][j] += 1
        if(winner == -1):
            win[j][i] += 1
        if(winner == 0):
            win[i][j] += 0.5
            win[j][i] += 0.5

        print("===========")
        print("AI1: " + str(i) + " AI2: " + str(j))
        print("Winner " + str(winner))
