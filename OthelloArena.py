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

#path = "/Users/student36/reinforcement-learning-othello/Weights_Folder1/"
path = "/home/oliver/git/othello/reinforcement-learning-othello/Weights_Folder1/"

controller = ReversiController(path, True, True, 2, epsilon = 10000)
controller.load([16000, 17000])
#controller.population[1] = BasicPlayer()

print(controller.population)

#controller.population[1] = RandomPlayer()

def fight(controller, index1, index2, toPlay):
    global mc

    observation = copy.deepcopy(controller.env.board)

    player = [controller.population[index1], controller.population[index2]]

    d = {1: 0, -1: 1}
    e = {0: 1, 1: -1}

    # Chose a move and take it

    move = player[d[toPlay]].policy(observation, toPlay)
    
    print(str(toPlay) + ", " + str(move))
    
    observation, reward, done, info = controller.env.step(move)

    mc.setBoard(observation)

def move1(event):
    global controller
    fight(controller, 0, 1, 1)

def move2(event):
    global controller
    fight(controller, 0, 1, -1)

mc.bind("<Left>", move1)
mc.bind("<Right>", move2)
mc.focus_set()
