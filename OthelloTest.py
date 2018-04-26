from MishaCanvas import *
from tkinter import *
from reversi import reversiBoard
from ReversiAI import *
import AlphaBeta
import time
from othelloBoard import Board
from copy import deepcopy
import time

controller = ReversiController(path, False, False, 2, epsilon = 10000)
controller.load([53000, 0])

