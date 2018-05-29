from MishaCanvas import *
from tkinter import *
from reversi import reversiBoard
from ReversiAI import *
import AlphaBeta
import time
from othelloBoard import Board
from copy import deepcopy
import time
import math

#path = "/Users/student36/reinforcement-learning-othello/Weights_Folder4/"
path = "/home/oliver/git/othello/reinforcement-learning-othello/Weights_Folder5/"

controller = ReversiController(path, True, True, 2, epsilon = 10000)
controller.load([7000, 0])

def process(array):
    new_array = []
    pieces = [0, 1, -1]
    
    for i in range(3):
        board = []
        for j in range(8):
            row = []
            for k in range(8):
                row.append(int(array[j][k] == pieces[i]))
            board.append(row)
        new_array.append(board)

    return new_array

def reverse(array):
    newarray = copy.deepcopy(array)
    d = {1:-1, 0:0, -1:1}
    for i in range(len(array)):
        for j in range(len(array[0])):
            newarray[i][j] = d[array[i][j]]
    return newarray

board = [[0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, -1],
       [0, 0, -1, 0, 0, 0, 1, 0],
       [1, 1, -1, -1, 1, 1, 1, -1],
       [0, 1, 1, 1, -1, 1, -1, -1],
       [-1, -1, -1, -1, 1, -1, -1, -1],
       [-1, -1, -1, -1, -1, -1, -1, -1],
       [1, -1, -1, -1, -1, -1, -1, -1]]

board2 = [[-1, -1, -1, -1, -1, -1, -1, -1],
       [-1, -1, 1, 1, 1, 1, 1, 1],
       [-1, 1, -1, -1, 1, -1, 1, 1],
       [-1, 1, -1, -1, -1, 1, 1, 1],
       [1, -1, -1, -1, -1, -1, 1, 1],
       [1, -1, 1, -1, -1, -1, -1, 1],
       [1, -1, -1, 1, -1, -1, 0, 1],
       [1, -1, -1, -1, -1, -1, 0, 1]]

board3 = [[0, -1, -1, -1, -1, -1, -1, 0],
          [-1, 0, -1, -1, 1, -1, 0, 0],
          [-1, 1, 1, -1, -1, 1, -1, -1],
          [-1, -1, 1, -1, -1, 1, -1, -1],
          [0, 0, 1, 1, 1, -1, 0, 0],
          [0, -1, 1, 1, 1, 0, -1, 0],
          [-1, 0, 0, 0, 0, 0, 0, -1],
          [0, 0, 0, 0, 0, 0, 0, 0]]

board4 = [[0, -1, -1, -1, -1, -1, -1, 0],
          [-1, 1, 1, 1, 1, -1, 0, 0],
          [-1, 1, 1, -1, -1, 1, -1, -1],
          [-1, -1, 1, -1, -1, 1, -1, -1],
          [0, 0, 1, 1, 1, -1, 0, 0],
          [0, -1, 1, 1, 1, 0, -1, 0],
          [-1, 0, 0, 0, 0, 0, 0, -1],
          [0, 0, 0, 0, 0, 0, 0, 0]]

board5 = [[0, -1, -1, -1, -1, -1, 0, 0],
          [-1, 0, -1, 1, 1, -1, 0, 0],
          [-1, -1, 1, 1, 1, -1, -1, -1],
          [-1, 1, -1, -1, 1, -1, -1, -1],
          [-1, 1, -1, -1, 1, -1, -1, -1],
          [-1, -1, -1, -1, 1, 1, -1, -1],
          [-1, 0, -1, 1, 1, 1, 0, -1],
          [0, 0, 1, 1, 0, 0, 0, 0]]

board6 = [[0, -1, -1, -1, -1, -1, 0, 0],
          [-1, 1, 1, 1, 1, -1, 0, 0],
          [-1, 1, 1, 1, 1, -1, -1, -1],
          [-1, 1, -1, -1, 1, -1, -1, -1],
          [-1, 1, -1, -1, 1, -1, -1, -1],
          [-1, -1, -1, -1, 1, 1, -1, -1],
          [-1, 0, -1, 1, 1, 1, 0, -1],
          [0, 0, 1, 1, 0, 0, 0, 0]]

board7 = [[-1, -1, -1, -1, -1, -1, 0, 0],
          [-1, -1, 1, 1, 1, -1, 0, 0],
          [-1, 1, -1, 1, 1, -1, -1, -1],
          [-1, 1, -1, -1, 1, -1, -1, -1],
          [-1, 1, -1, -1, 1, -1, -1, -1],
          [-1, -1, -1, -1, 1, 1, -1, -1],
          [-1, 0, -1, 1, 1, 1, 0, -1],
          [0, 0, 1, 1, 0, 0, 0, 0]]

board8 = [[0, -1, -1, -1, -1, -1, 1, 0],
          [-1, 0, -1, 1, 1, 1, 0, 0],
          [-1, -1, 1, 1, 1, -1, -1, -1],
          [-1, 1, -1, -1, 1, -1, -1, -1],
          [-1, 1, -1, -1, 1, -1, -1, -1],
          [-1, -1, -1, -1, 1, 1, -1, -1],
          [-1, 0, -1, 1, 1, 1, 0, -1],
          [0, 0, 1, 1, 0, 0, 0, 0]]

board9 = [[0, 0, 0, 0, 0, 0, 0, 0],
          [0, -1, 0, 0, 0, 0, 0, 0],
          [0, 1, -1, 1, 0, 0, 0, 0],
          [0, 0, 0, -1, 1, 0, 0, 0],
          [0, 0, 0, 1, -1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]]

#print(controller.population[0].model.predict(np.array([process(board)])))
#print(controller.population[0].model.predict(np.array([process(reverse(board))])))

tree = AlphaBeta.AlphaBeta(controller)

b = reversiBoard(8)
b.board = board5

b1 = reversiBoard(8)
b1.board = board7

b3 = reversiBoard(8)
b3.board = board8

print(controller.population[0].model.predict(np.array([process(board9)])))

