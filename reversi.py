#http://code.activestate.com/recipes/580698-reversi-othello/

import os, copy

class reversiBoard():
    pass_counter = 0
    
    # 8 directions
    dirx = [-1, 0, 1, -1, 1, -1, 0, 1]
    diry = [-1, -1, -1, 0, 0, 1, 1, 1]

    def __init__(self, n):
        self.n = n
        self.board = [[0 for i in range(8)] for j in range(8)]
        self.to_play = 1

        self.reset()

    def reset(self):
        n = self.n
        self.pass_counter = 0
        self.to_play = 1
        self.board = [[0 for i in range(8)] for j in range(8)]
        board = self.board
        
        if n % 2 == 0: # if board size is even
            z = (n - 2) // 2
            board[z][z] = -1
            board[n - 1 - z][z] = 1        
            board[z][n - 1 - z] = 1
            board[n - 1 - z][n - 1 - z] = -1

        return board
            
    def render(self):
        d = {1: 1, -1: 2, 0: 0}
        n = self.n
        board = self.board
        m = len(str(n - 1))
        for y in range(n):
            row = ''
            for x in range(n):
                row += str(d[board[y][x]])
                row += ' ' * m
            print(row + ' ' + str(y))
        print("")
        row = ''
        for x in range(n):
            row += str(x).zfill(m) + ' '
        print(row + '\n')

    def MakeMove(self, x, y, player): # assuming valid move
        n = self.n
        totctr = 0 # total number of opponent pieces taken

        board = self.board
        
        board[x][y] = player
        for d in range(8): # 8 directions
            ctr = 0
            for i in range(n):
                dx = x + self.dirx[d] * (i + 1)
                dy = y + self.diry[d] * (i + 1)
                if dx < 0 or dx > n - 1 or dy < 0 or dy > n - 1:
                    ctr = 0; break
                elif board[dx][dy] == player:
                    break
                elif board[dx][dy] == 0:
                    ctr = 0; break
                else:
                    ctr += 1
            for i in range(ctr):
                dx = x + self.dirx[d] * (i + 1)
                dy = y + self.diry[d] * (i + 1)
                board[dx][dy] = player
            totctr += ctr
        return (board, totctr)

    def TestMove(self, board, x, y, player): # assuming valid move
        n = self.n
        totctr = 0 # total number of opponent pieces taken
        
        board[x][y] = player
        for d in range(8): # 8 directions
            ctr = 0
            for i in range(n):
                dx = x + self.dirx[d] * (i + 1)
                dy = y + self.diry[d] * (i + 1)
                if dx < 0 or dx > n - 1 or dy < 0 or dy > n - 1:
                    ctr = 0; break
                elif board[dx][dy] == player:
                    break
                elif board[dx][dy] == 0:
                    ctr = 0; break
                else:
                    ctr += 1
            for i in range(ctr):
                dx = x + self.dirx[d] * (i + 1)
                dy = y + self.diry[d] * (i + 1)
                board[dx][dy] = player
            totctr += ctr
        return (board, totctr)

    def ValidMove(self, x, y, player):
        n = self.n

        board = self.board
        
        if x < 0 or x > n - 1 or y < 0 or y > n - 1:
            return False
        if board[x][y] != 0:
            return False
        (boardTemp, totctr) = self.TestMove(copy.deepcopy(board), x, y, player)
        if totctr == 0:
            return False
        return True

    # Here, Player is assumed to be 1
    def move_generator(self):
        possibleMoves = []
        
        board = self.board

        for i in range(len(board)):
            for j in range(len(board[i])):
                if(self.ValidMove(i,j,1)):
                    possibleMoves.append((i,j))

        return possibleMoves

    # Reverses the internal board
    def reverse(self):
        d = {1: -1, 0: 0, -1: 1}

        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                self.board[i][j] = d[self.board[i][j]]

    def findWinner(self):
        board = self.board
        positives = 0
        negatives = 0

        for i in range(len(board)):
            for j in range(len(board[i])):
                if(board[i][j] == -1):
                    negatives += 1
                elif(board[i][j] == 1):
                    positives += 1

        if(positives == negatives):
            return 0
        elif(positives > negatives):
            return 1 * self.to_play
        else:
            return -1 * self.to_play
    
    # 
    def step(self, action):
        #observation, reward, done, info
        x = action[0]
        y = action[1]
        reward = 0
        done = False

        if(x == -1 and y == -1):
            # Player decides to pass
            self.reverse()
            self.to_play *= 1
            self.pass_counter += 1
            
            if(self.pass_counter >= 2):
                # Two passes in a row
                done = True
                reward = self.findWinner()
                return self.board, reward, done, {}
            else:
                return self.board, reward, done, {}

        if(not self.ValidMove(x, y, 1)):
            print("Invalid Move!")
            done = True
            reward = -1*self.to_play # Gives a 1 if player 1 wins and vice versa
            return [], reward, done, {}
        else:
            # Always make a move like its the first person playing
            self.pass_counter = 0
            self.MakeMove(x,y,1)
            self.reverse()
            self.to_play *= -1
            return self.board, reward, done, {}
