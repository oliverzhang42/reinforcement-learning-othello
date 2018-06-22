from OthelloCanvas import *
from tkinter import *
from OthelloBoard import OthelloBoard
from OthelloController import *
import AlphaBeta
import time
from copy import deepcopy
import time
import absl


class Arena(object):
    def __init__(self, path):
        self.root = Tk()
        self.mc = BasicOthelloCanvas(self.root, 8, 8, cellsize = 100)
        
        self.env = OthelloBoard(8)
        self.env.reset()
        self.mc.setBoard(self.env.board)
        
        self.controller = OthelloController(path, 2, epsilon = 10000)
        self.playing = True

    def move(self, index1, index2, toPlay):
        observation = copy.deepcopy(self.env.board)
        
        players = [self.controller.population[index1], self.controller.population[index2]]
        
        d = {1: 0, -1: 1}
        e = {0: 1, 1: -1}
        
        # Chose a move and take it
        
        move = players[d[toPlay]].policy(observation, toPlay)
            
        observation, reward, done, info = self.env.step(move)
        
        self.mc.setBoard(observation)
    
    def play(self, load_weights1, load_weights2):
        self.controller.load([load_weights1, load_weights2])
        moving = True
        
        self.mc.focus_set()
        
        while(True):
            t1 = time.time()
            self.move(0, 1, 1)
            t2 = time.time()

            time.sleep(max(2-t2+t1, 0))
            self.mc.update()
            self.mc.update_idletasks()
            t1 = time.time()
            self.move(0, 1, -1)
            t2 = time.time()
            
            time.sleep(max(2-t2+t1, 0))
            self.mc.update()
            self.mc.update_idletasks()
            
            