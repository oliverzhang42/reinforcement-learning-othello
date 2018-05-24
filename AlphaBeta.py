#This is Alpha Beta Pruning for Othello, using
#https://github.com/hylbyj/Alpha-Beta-Pruning-for-Othello-Game/blob/master/othello.py
#ply is the depth

import math
import numpy as np
from othelloBoard import Board
from copy import deepcopy

path = "/Users/student36/reinforcement-learning-othello/"
#path = "/home/oliver/git/othello/reinforcement-learning-othello/"

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

def reverse(board):
    newBoard = [[0 for i in range(8)] for j in range(8)]
    d = {0:0, 1:-1, -1:1}
    for i in range(8):
        for j in range(8):
            newBoard[i][j] = d[board[i][j]]

    return newBoard


class AlphaBeta():
    def __init__(self, controller):
        self.controller = controller
    
    def policy(self, board, color, index):
#        if(color == -1):
#            board = reverse(board)
        processedBoard = process(board)
        processedBoard = np.array([processedBoard])
        return self.controller.population[index].model.predict(processedBoard)[0][0]

### This assumes passing is the end. In reality, passing gives an extra move
### to a player. I'll need to revise this
# Notes:
#
# First, the policy assumes color == 1. Just don't change it... (don't inclue
# the color == -1, board = reverse(board)
#
# Second, when first calling it, do color = 1!!!

    def alphabeta(self, board, depth, alpha, beta, color, index):
        if(depth == 0):
            return self.policy(board, color, index), None

        moves = board.get_legal_moves(color)
        
        if len(moves) == 0:
            return self.policy(board, color, index), None

        if(color == 1):
            v = -math.inf
            return_move = None

            for move in moves:
                newboard = deepcopy(board)
                newboard.execute_move(move,color)

                score, m = self.alphabeta(newboard, depth-1, alpha, beta, -1, index)
                
                if(score > v):
                    v = score
                    return_move = move
                alpha = max(alpha, v)
                
                if(beta <= alpha):
                    break
            return v, return_move
        else:
            v = math.inf
            return_move = None

            for move in moves:
                newboard = deepcopy(board)
                newboard.execute_move(move,color)

                score, m = self.alphabeta(newboard, depth - 1, alpha, beta, 1, index)

                if(score < v):
                    v = score
                    return_move = move
                beta = min(beta, v)

                if(beta <= alpha):
                    break
            return v, return_move
