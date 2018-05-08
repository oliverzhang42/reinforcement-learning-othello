from MishaCanvas import *
from tkinter import *
from reversi import reversiBoard
from ReversiAI import *
import AlphaBeta
import time
from othelloBoard import Board
from copy import deepcopy
import time

path = "/Users/student36/reinforcement-learning-othello/Weights_Folder2/"
#path = "/home/oliver/git/othello/reinforcement-learning-othello/Weights_Folder2/"

controller = ReversiController(path, True, True, 2, epsilon = 10000)
controller.load([32000, 0])

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

board2 = [[]]

print(controller.population[0].model.predict(np.array([process(board)])))
print(controller.population[0].model.predict(np.array([process(reverse(board))])))

tree = AlphaBeta.AlphaBeta(controller)

b = Board()
b.pieces = board

print(tree._minmax_with_alpha_beta(b, 1, 1, 0))
print(tree._minmax_with_alpha_beta(b, -1, 1, 0))


#controller.play_two_ai(0,0)
