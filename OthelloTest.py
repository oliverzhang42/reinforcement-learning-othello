from MishaCanvas import *
from tkinter import *
from reversi import reversiBoard
from ReversiAI import *
import AlphaBeta
import time
from othelloBoard import Board
from copy import deepcopy
import time

#path = "/Users/student36/reinforcement-learning-othello/Weights_Folder1/"
path = "/home/oliver/git/othello/reinforcement-learning-othello/Weights_Folder1/"

controller = ReversiController(path, True, True, 2, epsilon = 10000)
controller.load([13500, 0])

controller.play_two_ai(0,0)
