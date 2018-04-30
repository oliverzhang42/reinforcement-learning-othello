from MishaCanvas import *
from tkinter import *
from reversi import reversiBoard
from ReversiAI import *
import AlphaBeta
import time
from othelloBoard import Board
from copy import deepcopy
import time

#path = "/Users/student36/reinforcement-learning-othello/Weights_Folder3/"
path = "/home/oliver/git/othello/reinforcement-learning-othello/Weights_Folder3/"

controller = ReversiController(path, True, True, 2, epsilon = 10000)
controller.load([53000, 0])
controller.population[0] = BasicPlayer()

state = [[1, 1, 1, -1, 1, 1, 0, 0],
         [-1, -1, -1, -1, 1, 0, 0, 0],
         [1, 0, -1, 1, -1, 1, 0, 0],
         [1, 1, 1, 1, -1, 1, 1, 1],
         [1, -1, 1, 1, -1, 0, 0, 0],
         [1, 1, 1, 1, -1, 0, 0, 0],
         [1, 0, 1, 1, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 0, 0, 0]]

board = reversiBoard(8)
board.reset()
board.board = state

print(controller.population[1].policy(state, -1))
