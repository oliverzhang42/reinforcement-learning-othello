from MishaCanvas import *
from tkinter import *
from reversi import reversiBoard
from ReversiAI import *
import AlphaBeta
import time
from othelloBoard import Board
from copy import deepcopy

root = Tk()
# Grid is 10x15, cell size 20 pixels, white to start with
#  It will pack itself into the root
mc = BasicMishaCanvas(root, 8, 8, cellsize = 100)

env = reversiBoard(8)
env.reset()
#env.reverse()
mc.setBoard(env.board)



#path = "/Users/student36/Desktop/ReinforcementLearning/Reversi1/"
path = "/home/oliver/git/othello/reinforcement-learning-othello/"

controller = ReversiController(path, False, False, 1, epsilon = 10000)
controller.load([40000])

alphabeta = AlphaBeta.AlphaBeta()

# Bind a function to a click to the canvas
def makeMove(event):
    global env
    x, y = mc.cell_coords(event.x, event.y)

    if(env.ValidMove(x,y,1)):
        observation, reward, done, info = env.step([x,y])

        if(done):
            print("Done!!!")
        else:
            board = Board()
            board.pieces = observation
            
            (value, move) = alphabeta._minmax_with_alpha_beta(board, 1, 5)

            env.step(move)

            mc.setBoard(env.board)
    else:
       print("That Move cannot be made, make another one.")
       #print(env.board)

def passMove(event):
    global env
    global alphabeta
    observation, reward, done, info = env.step([-1,-1])

    if(done):
        print("Done!!!")
    else:
        board = Board()
        board.pieces = observation
            
        (value, move) = alphabeta._minmax_with_alpha_beta(board, 1, 5)

        env.step(move)

        mc.setBoard(env.board)

mc.bind("<Button-1>", makeMove)
mc.bind("<Button-2>", passMove)
