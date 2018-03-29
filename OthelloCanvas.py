from MishaCanvas import *
from tkinter import *
from reversi import reversiBoard
from ReversiAI import *
import time
 
root = Tk()
# Grid is 10x15, cell size 20 pixels, white to start with
#  It will pack itself into the root
mc = BasicMishaCanvas(root, 8, 8, cellsize = 100)

env = reversiBoard(8)
env.reset()
mc.setBoard(env.board)

path = "/Users/student36/Desktop/ReinforcementLearning/Reversi1/"
#path = "/home/oliver/Desktop/Reversi3/"

controller = ReversiController(path, False, False, 1, epsilon = 10000)
controller.load([40000])

# Bind a function to a click to the canvas
def makeMove(event):
    global env
    x, y = mc.cell_coords(event.x, event.y)

    if(env.ValidMove(x,y,1)):
        observation, reward, done, info = env.step([x,y])

        if(done):
            print("Done!!!")
        else:
            move = controller.population[0].policy(observation)

            env.step(move)

            mc.setBoard(env.board)
    else:
       print("That Move cannot be made, make another one.")

def passMove(event):
    global env
    observation, reward, done, info = env.step([-1,-1])

    if(done):
        print("Done!!!")
    else:
        move = controller.population[0].policy(observation)

        env.step(move)

        mc.setBoard(env.board)

mc.bind("<Button-1>", makeMove)
mc.bind("<Button-2>", passMove)
