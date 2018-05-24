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

path = "/Users/student36/reinforcement-learning-othello/Weights_Folder4/"
#path = "/home/oliver/git/othello/reinforcement-learning-othello/Weights_Folder4/"

controller = ReversiController(path, True, True, 2, epsilon = 10000)
controller.load([35000, 0])

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

print(controller.population[0].model.predict(np.array([process(board)])))
print(controller.population[0].model.predict(np.array([process(reverse(board))])))

tree = AlphaBeta.AlphaBeta(controller)

b = Board()
b.pieces = board3

print(tree.alphabeta(b, 2, -math.inf, math.inf, 1, 0))
print(tree.alphabeta(b, 3, -math.inf, math.inf, 1, 0))
print(tree.alphabeta(b, 4, -math.inf, math.inf, 1, 0))


#print(tree.alphabeta(b, 5, -math.inf, math.inf, 1, 0))
#print(tree.alphabeta(b, 5, -math.inf, math.inf, 1, 0))


#controller.play_two_ai(0,0)
