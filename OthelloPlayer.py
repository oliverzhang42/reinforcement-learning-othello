import keras
from keras.layers import BatchNormalization, Dense, Activation, Conv2D, Flatten
from keras.layers import Input, Add
from keras.optimizers import Adam
from keras import backend as K
import random
import time
import numpy as np
from OthelloBoard import * 
import copy
from AlphaBeta import AlphaBeta
import math

# How many neurons for each layer
LAYER_SIZE = 256

# Here, REWARD_DECAY is how much we care about the delayed reward compared to
# the immediate reward. REWARD_DECAY = 1 means we care about all reward the
# same, REWARD_DECAY = 0 means we don't care at all about the later rewards.
REWARD_DECAY = 0.99

# Size of the mini-batches used in training
BATCH_SIZE = 64

def reverse(array):
    newarray = copy.deepcopy(array)
    d = {1:-1, 0:0, -1:1}
    for i in range(len(array)):
        for j in range(len(array[0])):
            newarray[i][j] = d[array[i][j]]
    return newarray

def rotate_array(array):
    for i in range(len(array)):
        array[0] = rotate_90(array[0])
        array[1] = rotate_90(array[1])
        array[2] = rotate_90(array[2])

    return array

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

def rotate_90(array):
    # Array is a 8x8 array.
    # ccw rotation

    new_array = []

    for i in range(8):
        row = []
        for j in range(8):
            row.append(array[7-j][i])
        new_array.append(row)

    return new_array

class OthelloPlayer:
    def __init__(self, index, depth, parent = None, learning_rate = 0.00005, epsilon = 2,
                 epsilon_increment = 0.00005, debugging = False):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_increment = epsilon_increment
        self.experience = []
        self.debugging = debugging
        self.parent = parent
        self.index = index
        self.depth = depth

        self.create_model()

    def create_model(self):
        main_input = Input(shape = (3,8,8))

        c1 = Conv2D(LAYER_SIZE, (3,3), activation = 'relu', padding = 'same')(main_input)
        b1 = BatchNormalization()(c1)
        c2 = Conv2D(LAYER_SIZE, (3,3), activation = 'relu', padding = 'same')(b1)
        b2 = BatchNormalization()(c2)
        c3 = Conv2D(LAYER_SIZE, (3,3), activation = 'relu', padding = 'same')(b2)
        b3 = BatchNormalization()(c3)

        a3 = Add()([b3, b1])

        c4 = Conv2D(LAYER_SIZE, (3,3), activation = 'relu', padding = 'same')(a3)
        b4 = BatchNormalization()(c4)
        c5 = Conv2D(LAYER_SIZE, (3,3), activation = 'relu', padding = 'same')(b4)
        b5 = BatchNormalization()(c5)

        a5 = Add()([b5, a3])

        b6 = Conv2D(LAYER_SIZE, (3,3), activation = 'relu', padding = 'same')(a5)
        
        f1 = Flatten()(b6)
        d1 = Dense(LAYER_SIZE, activation = 'relu')(f1)
        d2 = Dense(1, activation = 'tanh')(d1)

        self.model = keras.models.Model(inputs = main_input, outputs = d2)

        self.model.compile(Adam(self.learning_rate), "mse")


    def add_to_history(self, state_array, reward):
        answers = []
        history = self.experience

        current_reward = reward

        processed_array = []

        for i in range(len(state_array)):
            processed_array.append(process(state_array[i]))

        state_array = processed_array
        
        for i in range(len(state_array)):
            current_array = state_array[len(state_array) - i - 1]
            
            history.append([current_array,
                                 current_reward])
            current_array = rotate_array(current_array)
            history.append([current_array,
                                 current_reward])
            current_array = rotate_array(current_array)
            history.append([current_array,
                                 current_reward])
            current_array = rotate_array(current_array)
            history.append([current_array,
                                 current_reward])
            current_reward *= REWARD_DECAY

    def wipe_history(self):
        self.experience = []

    def train_model(self, verbose):
        inputs = []
        answers = []
        history = self.experience
                  
        for i in range(BATCH_SIZE):
            lesson = random.choice(history)
            inputs.append(lesson[0])
            answers.append(lesson[1])

        inputs = np.array(inputs)
        answers = np.array(answers)
        
        self.model.fit(x = inputs, y = answers, verbose = 1)

    # Saves the model's weights.
    def save(self, s):
        self.model.save_weights(s)

    # Loads the weights of a previous model.
    def load(self, s):
        self.model.load_weights(s)
        #self.default_graph.finalize()

    def policy(self, observation, player):
        # Value is an array. The 0th element corresponds to (0,0), the 1st: (0,1)
        # the 8th: (1,0), etc.
        value = []

        if(player == -1):
            observation = reverse(observation)

        possible_moves = findMovesWithoutEnv(observation)

        if(len(possible_moves) == 0):
            # Passes
            return (-1, -1)

        if(self.debugging):
            print(possible_moves)
        
        decision_tree = AlphaBeta(self.parent)
        
        variation = random.random()
        
        if(variation < 1/self.epsilon):
            self.epsilon += self.epsilon_increment
            if(self.debugging):
                print("Random Move for player " + str(env.to_play))
            return random.choice(possible_moves)
        else:
            board = OthelloBoard(8)
            board.board = observation
            value, move = decision_tree.alphabeta(board, self.depth, -math.inf,
                                                  math.inf, 1, self.index)

            if(move == None):
                return (-1,-1)
            return move

class RandomPlayer(OthelloPlayer):
    def __init__(self):
        pass
    
    def add_to_history(self, state_array, reward):
        pass

    def wipe_history(self):
        pass
   
    def train_model(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def policy(self, observation, player):
        if(player == -1):
            observation = reverse(observation)
        
        possibleMoves = findMovesWithoutEnv(observation)

        if(len(possibleMoves) == 0):
            return (-1, -1)
  
        return random.choice(possibleMoves)

class BasicPlayer(RandomPlayer):
    def __init__(self, depth):
        self.weights = [[1000, 50, 100, 100, 100, 100, 50, 1000],
                   [50, -20, -10, -10, -10, -10, -20, 50],
                   [100, -10, 1, 1, 1, 1, -10, 100],
                   [100, -10, 1, 1, 1, 1, -10, 100],
                   [100, -10, 1, 1, 1, 1, -10, 100],
                   [100, -10, 1, 1, 1, 1, -10, 100],
                   [50, -20, -10, -10, -10, -10, -20, 50],
                   [1000, 50, 100, 100, 100, 100, 50, 1000]]

    def calculateScore(self, observation):
        score = 0
        for i in range(len(observation)):
            for j in range(len(observation[0])):
                score += observation[i][j] * self.weights[i][j]
        return score

    def policy(self, observation, player):
        if(player == -1):
            observation = reverse(observation)

        possibleMoves = findMovesWithoutEnv(observation)

        bestScore = -1000
        bestMove = (-1, -1)

        for move in possibleMoves:
            newobs = Board()
            newobs.pieces = copy.deepcopy(observation)
            newobs.execute_move(move, 1)
            tempScore = self.calculateScore(newobs.pieces)
            if(tempScore > bestScore):
                bestScore = tempScore
                bestMove = move
        
        return bestMove
