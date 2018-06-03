# So I'm switching my structure and Dividing the Reversi Class into two classes:
# ReversiPlayer will be a single AI.
# ReversiController will be a framework for the AI to exist in.

# Importing Stuff
import keras
from keras.layers import BatchNormalization, Dense, Activation, Conv2D, Flatten
from keras.layers import Input
from keras.optimizers import Adam
import random
import h5py
import time
import numpy as np
from reversi import * 
import copy
from othelloBoard import Board
from AlphaBeta import AlphaBeta
import math
from threading import Thread
from keras import backend as K
import tensorflow as tf

# Global Variables

# After Every SAVE_FREQUENCY episodes, we save the weights of the model in path.
SAVE_FREQUENCY = 50

# After Every WIPE_FREQUENCY episodes, we wipe the history of the two players.
WIPE_FREQUENCY = 2

# The number of total episodes to run.
TOTAL_EPISODES = 23000

# The size of each layer in the model. Currently Unused
LAYER_SIZE = 30

# The size of the othello board. (BOARD_SIZE by BOARD_SIZE)
BOARD_SIZE = 8

# Here, REWARD_DECAY is how much we care about the delayed reward compared to
# the immediate reward. REWARD_DECAY = 1 means we care about all reward the
# same, REWARD_DECAY = 0 means we don't care at all about the later rewards.
REWARD_DECAY = 0.99

BATCH_SIZE = 64

# Episodes before switching which model to train
EPISODES_BEFORE_SWITCH = 200

# Number of threads you want to run
THREAD_NUM = 8

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

class ReversiPlayer:
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

        self.session = tf.Session()
        K.set_session(self.session)

        self.create_model()
        
        self.model._make_predict_function()
        self.default_graph = tf.get_default_graph()

    def create_model(self):
        main_input = Input(shape = (3,8,8))

        c1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(main_input)
        b1 = BatchNormalization()(c1)
        c2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(b1)
        b2 = BatchNormalization()(c2)
        c3 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(b2)
        b3 = BatchNormalization()(c3)
        c4 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(b3)
        b4 = BatchNormalization()(c4)
        f1 = Flatten()(b4)
        d1 = Dense(256, activation = 'relu')(f1)
        d2 = Dense(1, activation = 'tanh')(d1)

        self.model = keras.models.Model(inputs = main_input, outputs = d2)
        
        #self.model = keras.models.Sequential()

        #self.model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same',
        #                      input_shape = (3,8,8)))
        #self.model.add(BatchNormalization())
        #self.model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
        #self.model.add(BatchNormalization())
        #self.model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
        #self.model.add(BatchNormalization())
        #self.model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
        #self.model.add(BatchNormalization())
        #self.model.add(Flatten())
        #self.model.add(Dense(256, activation = 'relu'))
        #self.model.add(Dense(1, activation = 'tanh'))

        self.model.compile(Adam(self.learning_rate), "mse")

        #self.model._make_predict_function()


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

        #print(history)

    def wipe_history(self):
        self.experience = []

        print("WIPE!")

    def train_model(self, verbose):
        inputs = []
        answers = []
        history = self.experience
                  
        for i in range(BATCH_SIZE):
            lesson = random.choice(history)
            inputs.append(lesson[0])
            answers.append(lesson[1])

        #print(model_num)

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
            board = reversiBoard(8)
            board.board = observation
            value, move = decision_tree.alphabeta(board, self.depth, -math.inf,
                                                  math.inf, 1, self.index)
            #print("%.15f" % value)
            #print(move)
            #rint("")
            if(move == None):
                return (-1,-1)
            return move

class RandomPlayer(ReversiPlayer):
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

class ReversiController:
    def __init__(self, path, display_img, debugging, population_size,
                 learning_rate = 0.0001, epsilon = 2, epsilon_increment = 0.001):
        self.display_img = display_img
        self.debugging = debugging
        self.path = path

        if(debugging):
            epsilon = 20000

        self.population = [ReversiPlayer(i, 3, self, learning_rate, epsilon,
                                         epsilon_increment, debugging)
                           for i in range(population_size)]

        self.population.append(RandomPlayer())

    def play_two_ai(self, index1, index2):
        return self.play_two_ai_training(index1, index2, False)

    def play_two_ai_training(self, index1, index2, training):
        switch = 0

        if(training):
            switch = random.randint(0, 58)
            #print("Switch: " + str(switch))

        # Random Player Index
        rpi = len(self.population) - 1
        
        move_player = [self.population[rpi], self.population[rpi]]
        learn_player = [self.population[index1], self.population[index2]]

        d = {1: 0, -1: 1}
        e = {0: 1, 1: -1}

        env = reversiBoard(BOARD_SIZE)
        observation = env.reset()

        # First array corresponds to the states faced by the first player
        # Same with second
        state_array = [[],[]]

        for t in range(200):
            
            if(t == switch):
                move_player = [self.population[index1], self.population[index2]]
                
            if(self.display_img):
                env.render()
            
            if(self.debugging):
                pass
                #time.sleep(5)

            # Chose a move and take it
            move = move_player[t % 2].policy(observation, e[t % 2])
            
            observation, reward, done, info = env.step(move)

            if(self.debugging):
                print(env.to_play)
                print("")
                print("Move")
                print(move)
                print("")

                print("Observation")
                print(observation)
                print("")

                time.sleep(3)
            
            if(not done and t >= switch):
                if(env.to_play == 1):
                    state_array[0].append(observation)
                elif(env.to_play == -1):
                    state_array[1].append(reverse(observation))

            # Check if done. We're only training once we finish the entire
            # episode. Here, the model which makes the last move has number
            # model_num, and the reward it has is reward

            if done:
                if(reward == 0):
                    print("Draw")
                    
                print("Episode finished after {} timesteps".format(t+1)) 

                if(self.debugging):
                    print("Winner: " + str(reward))
                #print("Winner: " + str(reward))


                if(len(state_array[0]) == 0):
                    pass

                learn_player[0].add_to_history(state_array[0], reward)
                learn_player[1].add_to_history(state_array[1], -reward)

                return reward
        return reward
        
    def main(self, total_episodes):
        # For some reason, I need this ... Not sure why ...
        K.get_session()
        
        #Number of training episodes
        for i in range(total_episodes):
            #One Round Robin Tournament
            for j in range(len(self.population) - 1): #The minus 1 is there for the randomPlayer
                for k in range(len(self.population) - 1):
#                    self.play_two_ai(j,k)
                    thread_array = []
                    for l in range(THREAD_NUM):
                        t = Thread(target = self.play_two_ai_training,
                                   args = (j,k, True))
                        t.start()
                        thread_array.append(t)

                    for t in thread_array:
                        t.join()

            #Everyone Trains
            for j in range(len(self.population) - 1):
                for k in range(THREAD_NUM):
                    self.population[j].train_model(self.debugging)

            if(i % SAVE_FREQUENCY == 0):
                print(i)
                self.save([i])

            if(i % WIPE_FREQUENCY == 0):
                for j in range(len(self.population) - 1):
                    self.population[j].wipe_history()
        
    def save(self, episode_numbers):
        for j in range(len(self.population) - 1):
            self.population[j].save(self.path + "Reversi_%d_%d" %
                                    (j, episode_numbers[j]))

    def load(self, episode_numbers):
        for j in range(len(self.population) - 1):
            self.population[j].load(self.path + "Reversi_%d_%d" %
                                    (j, episode_numbers[j]))
