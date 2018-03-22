# So I'm switching my structure and Dividing the Reversi Class into two classes:
# ReversiPlayer will be a single AI.
# ReversiController will be a framework for the AI to exist in.

# Importing Stuff
import keras
from keras.layers import BatchNormalization, Dense, Activation, Conv2D, Flatten
from keras.optimizers import Adam
import random
import h5py
import time
import numpy as np
from reversi import reversiBoard 
import copy

# Global Variables

# After Every SAVE_FREQUENCY episodes, we save the weights of the model in path.
SAVE_FREQUENCY = 100

# After Every WIPE_FREQUENCY episodes, we wipe the history of the two players.
WIPE_FREQUENCY = 10

# The number of total episodes to run.
TOTAL_EPISODES = 20000

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

class ReversiPlayer:
    def __init__(self, learning_rate, epsilon = 2, epsilon_increment = 0.001):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_increment = epsilon_increment
        self.experience = []

        self.create_model()

    def create_model(self):
        self.model = keras.models.Sequential()

        self.model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same',
                              input_shape = (1,8,8)))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(256, activation = 'relu'))
        self.model.add(Dense(1))

        self.model.compile(Adam(self.learning_rate), "mse")

    def add_to_history(self, state_array, reward):
        answers = []
        history = self.experience

        current_reward = reward
     
        for i in range(len(state_array)):
            current_array = state_array[len(state_array) - i - 1]
            
            history.append([[current_array],
                                 current_reward])
            current_array = rotate_90(current_array)
            history.append([[current_array],
                                 current_reward])
            current_array = rotate_90(current_array)
            history.append([[current_array],
                                 current_reward])
            current_array = rotate_90(current_array)
            history.append([[current_array],
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

        inputs = np.array(inputs)
        answers = np.array(answers)

        #print(model_num)
        #print(inputs)
        #print(answers)
        
        self.model.fit(x = inputs, y = answers, verbose = verbose)

    # Saves the model's weights.
    def save(self, s):
        self.model.save_weights(s)

    # Loads the weights of a previous model.
    def load(self, s):
        self.model.load_weights(s)

    def policy(self, observation):
        # Value is an array. The 0th element corresponds to (0,0), the 1st: (0,1)
        # the 8th: (1,0), etc.
        value = []

        possible_moves = self.env.move_generator()

        if(len(possible_moves) == 0):
            # Passes
            return (-1, -1)

        if(self.debugging):
            print(possible_moves)
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if((i,j) in possible_moves):
                    move = copy.deepcopy(observation)
                    # We always assume that we're player 1.
                    move[i][j] = 1
                    move = np.array([move])
                    move = np.array([move])
                    value.append(self.model.predict(move)[0][0])
                else:
                    value.append(-1)

        variation = random.random()
        
        if(variation < 1/self.epsilon):
            self.epsilon += self.epsilon_increment
            #print(self.epsilon)
            if(self.debugging):
                print("Random Move for player " + str(self.env.to_play))
            return random.choice(possible_moves)
        else:
            if(self.debugging):
                print(np.array(value).tolist())
            #print(value.index(max(value)))
            index = value.index(max(value))
            action = (index // 8,index % 8)
            return action

class ReversiController:
    def __init__(self, learning_rate, display_img, debugging, population_size,
                 epsilon = 2, epsilon_increment = 0.001):
        self.env = reversiBoard(BOARD_SIZE)
        self.display_img = display_img
        self.debugging = debugging

        self.population = [ReversiPlayer(learning_rate, epsilon, epsilon_increment)
                           for i in range(population_size)]

    def play_two_ai():

    def test():

    def main():

    def save():

    def load():
