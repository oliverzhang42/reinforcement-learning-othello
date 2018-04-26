from MishaCanvas import *
from tkinter import *
from reversi import reversiBoard
from ReversiAI import *
import AlphaBeta
import time
from othelloBoard import Board
from copy import deepcopy
import time

root = Tk()
# Grid is 8x8, cell size 100 pixels, white to start with
#  It will pack itself into the root
mc = BasicMishaCanvas(root, 8, 8, cellsize = 100)

env = reversiBoard(8)
env.reset()
mc.setBoard(env.board)



path = "/Users/student36/reinforcement-learning-othello/Weights_Folder3/"
#path = "/home/oliver/git/othello/reinforcement-learning-othello/Weights_Folder3/"

controller = ReversiController(path, False, False, 1, epsilon = 10000)
controller.load([53000])

def reverse(board):
    d = {1: -1, 0:0, -1:1}

    newboard = [[0 for i in range(8)] for j in range(8)]
    
    for i in range(8):
        for j in range(8):
            newboard[i][j] = d[newboard[i][j]]

    return newboard

# Bind a function to a click to the canvas
def makeMove(event):
    global env
    global controller
    x, y = mc.cell_coords(event.x, event.y)

    if(env.ValidMove(x,y,1)):
        observation, reward, done, info = env.step([x,y])

        if(done):
            print("Done!!!")
        else:
            mc.setBoard(observation)
            
            #time.sleep(1)
            
            move = controller.population[0].policy(observation, -1)
            
            observation, reward, done, info = env.step(move)

            mc.setBoard(env.board)
    else:
       print("That Move cannot be made, make another one.")
       #print(env.board)

def passMove(event):
    global env
    global controller
    observation, reward, done, info = env.step([-1,-1])

    if(done):
        print("Done!!!")
    else:
        move = controller.population[0].policy(observation, -1)

        observation, reward, done, info = env.step(move)

        mc.setBoard(observation)

mc.bind("<Button-1>", makeMove)
mc.bind("<Button-2>", passMove)
