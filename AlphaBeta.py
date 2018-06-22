import math
import numpy as np
from OthelloBoard import OthelloBoard
from copy import deepcopy

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
        processedBoard = process(board.board)
        processedBoard = np.array([processedBoard])
        return self.controller.population[index].model.predict(processedBoard)[0][0]

    def alphabeta(self, board, depth, alpha, beta, color, index):
        if(depth == 0):
            return self.policy(board, color, index), None

        moves = []
        if(color == 1):
            moves = board.move_generator()
        else:
            board.reverse()
            moves = board.move_generator()
            board.reverse()
        
        if len(moves) == 0:
            newboard = OthelloBoard(8)
            newboard.reset()
            newboard.board = deepcopy(board.board)
            score, m = self.alphabeta(newboard, depth-1, alpha, beta, -color, index)
                
            return score, m

        if(color == 1):
            return_move = None

            for move in moves:
                newboard = OthelloBoard(8)
                newboard.reset()
                newboard.board = deepcopy(board.board)
                newboard.MakeMove(move[0], move[1],color)

                score, m = self.alphabeta(newboard, depth-1, alpha, beta, -1, index)
                
                if(score > alpha):
                    alpha = score
                    return_move = move
                
                if(beta <= alpha):
                    break
            return alpha, return_move
        elif(color == -1):
            return_move = None

            for move in moves:
                newboard = OthelloBoard(8)
                newboard.reset()
                newboard.board = deepcopy(board.board)
                newboard.MakeMove(move[0], move[1],color)

                score, m = self.alphabeta(newboard, depth - 1, alpha, beta, 1, index)

                if(score < beta):
                    beta = score
                    return_move = move

                if(beta <= alpha):
                    break
            return beta, return_move
        else:
            print("Error in AlphaBeta, Color is not 1 or -1")
            print("Color is: " + color)
            raise(Exception("Color Error in Alpha Beta"))
