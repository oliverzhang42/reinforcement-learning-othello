from OthelloCanvas import *
from tkinter import *
from OthelloBoard import OthelloBoard
from OthelloController import *
import AlphaBeta
import time
from copy import deepcopy
import time

def reverse(board):
    d = {1: -1, 0:0, -1:1}

    newboard = [[0 for i in range(8)] for j in range(8)]
     
    for i in range(8):
        for j in range(8):
            newboard[i][j] = d[newboard[i][j]]

    return newboard
    
class OthelloSession(object):
    def __init__(self, path):
        self.path = path
        self.root = Tk()
        self.env = OthelloBoard(8)
        self.env.reset()
        
        self.mc = BasicOthelloCanvas(self.root, 8, 8, cellsize = 100)
        self.mc.setBoard(self.env.board)
        
        self.controller = OthelloController(path, 1, epsilon = 10000)
    
    def play(self, load_num):
        self.controller.load([load_num])
        self.controller.population[0].depth = 3
        
        def makeMove(event):
            x, y = self.mc.cell_coords(event.x, event.y)
        
            if(self.env.ValidMove(x,y,1)):
                observation, reward, done, info = self.env.step([x,y])
        
                if(done):
                    print("Done!!!")
                else:
                    self.mc.setBoard(observation)
                    self.mc.update() 
                    
                    move = self.controller.population[0].policy(observation, -1)
                    
                    observation, reward, done, info = self.env.step(move)
        
                    self.mc.setBoard(self.env.board)
                    self.mc.after(5000, self.mc.update)
            else:
               print("That Move cannot be made, make another one.")
        
        def passMove(event):
            observation, reward, done, info = self.env.step([-1,-1])
        
            if(done):
                print("Done!!!")
            else:
                move = self.controller.population[0].policy(observation, -1)
        
                observation, reward, done, info = self.env.step(move)
        
                self.mc.setBoard(observation)
        
        self.mc.bind("<Button-1>", makeMove)
        self.mc.bind("<Button-2>", passMove)
        self.mc.update()
        
        self.root.mainloop()