import gym
from gym import spaces
import numpy as np

#https://github.com/nczempin/gym-tic-tac-toe/blob/master/gym_tic_tac_toe/envs/tic_tac_toe_env.py#L1

class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Discrete(512*512*2) # flattened

    #Step returns:
    #the new state, the reward (1 if you won, 0 if you didn't), and whether it is done
    def _step(self, action):
        done = False
        reward = 0

        p, square = action
        
       # p = p*2 - 1
        # check move legality
        board = self.state['board']
        proposed = board[square]
        om = self.state['on_move']
        if (proposed != 0):  # wrong player, not empty
            print("illegal move ", action, ". (square occupied): ", square)
            done = True
            reward = -1 * om  # player who did NOT make the illegal move
        if (p != om):  # wrong player, not empty
            print("illegal move  ", action, " not on move: ", p)
            done = True
            reward = -1 * om  # player who did NOT make the illegal move
        else:
            board[square] = p
            self.state['on_move'] = -p

        # check game over
        for i in range(3):
            # horizontals and verticals
            if ((board[i * 3] == p and board[i * 3 + 1] == p and board[i * 3 + 2] == p)
                or (board[i + 0] == p and board[i + 3] == p and board[i + 6] == p)):
                reward = p
                done = True
                break
        # diagonals
        if((board[0] == p and board[4] == p and board[8] == p)
            or (board[2] == p and board[4] == p and board[6] == p)):
                reward = p
                done = True

        # check draw
        if(board[0] != 0 and board[1] != 0 and board[2] != 0 and board[3] != 0
           board[4] != 0 and board[5] != 0 and board[6] != 0 and board[7] != 0
           board[8] != 0 and not done):
            done = True
                
        return self.state, reward, done, {}
    def _reset(self):
        self.state = {}
        self.state['board'] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.state['on_move'] = 1
        return self.state
    def _render(self, mode='human', close=False):
        if close:
            return
        print("on move: " , self.state['on_move'])
        for i in range (9):
            print (self.state['board'][i], end=" ")
            if(i % 3 == 2):
                print("")
        print()
    def move_generator(self):
        moves = []
        for i in range (9):
            
            if (self.state['board'][i] == 0):
                p = self.state['on_move']
                m = [p, i]
                moves.append(m)
        return moves
    def load(self,state):
        self.state = state
    def get_state(self,state):
        return self.state
    def _seed(self):
        return 10
