#Note: Comments are wrong right now

import gym
import keras
from keras.layers import BatchNormalization, Dense, Activation, Conv2D
from keras.optimizers import Adam
import random
import h5py
import time
import numpy as np

# After Every SAVE_FREQUENCY episodes, we save the weights of the model in path.
SAVE_FREQUENCY = 100

# The number of total episodes to run.
TOTAL_EPISODES = 10000

# The size of each layer in the model.
LAYER_SIZE = 30

INPUT_SIZE = 9

# Here, REWARD_DECAY is how much we care about the delayed reward compared to
# the immediate reward. REWARD_DECAY = 1 means we care about all reward the
# same, REWARD_DECAY = 0 means we don't care at all about the later rewards.
REWARD_DECAY = 0.9

BATCH_SIZE = 64

# Episodes before switching which model to train
EPISODES_BEFORE_SWITCH = 200

class TicTacToe:
    def __init__(self, learning_rate, display_img, debugging, path):
        self.display_img = display_img
        self.debugging = debugging
        self.path = path

        # self.env is the implementation of the CartPole game. Code:
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        self.env = gym.make('TicTacToe-v0')

        # Epsilon sometimes randomizing the player's actions. Helps with
        # exploration of more possibilities.
        self.epsilon = 2

        self.create_model_1()
        self.create_model_0()

        self.experience = []


    # This Function Creates a Keras Model with three sections of:
    # a Batch Norm Layer, a Dense layer, and an Activation.
    # There a fourth section with no activation because the output
    # isn't limited in a 0-1 range.
    def create_model_1(self):
        self.model_1 = keras.models.Sequential()
        self.model_1.add(BatchNormalization(input_shape = (INPUT_SIZE,)))
        self.model_1.add(Dense(LAYER_SIZE))
        self.model_1.add(Activation("relu"))

        self.model_1.add(BatchNormalization())
        self.model_1.add(Dense(LAYER_SIZE)) 
        self.model_1.add(Activation("relu"))

        self.model_1.add(BatchNormalization())
        self.model_1.add(Dense(LAYER_SIZE)) 
        self.model_1.add(Activation("relu"))

        self.model_1.add(BatchNormalization())
        self.model_1.add(Dense(1))
        self.model_1.add(Activation("sigmoid"))

        self.model_1.compile(loss='mse', optimizer = Adam(learning_rate))
    
    def create_model_0(self):
        self.model_0 = keras.models.Sequential()
        self.model_0.add(BatchNormalization(input_shape = (INPUT_SIZE,)))
        self.model_0.add(Dense(LAYER_SIZE))
        self.model_0.add(Activation("relu"))

        self.model_0.add(BatchNormalization())
        self.model_0.add(Dense(LAYER_SIZE)) 
        self.model_0.add(Activation("relu"))

        self.model_0.add(BatchNormalization())
        self.model_0.add(Dense(LAYER_SIZE)) 
        self.model_0.add(Activation("relu"))

        self.model_0.add(BatchNormalization())
        self.model_0.add(Dense(1))
        self.model_0.add(Activation("sigmoid"))

        self.model_0.compile(loss='mse', optimizer = Adam(learning_rate))

    # Policy is how the model picks an action for a given situation and weights.
    # This is not training.

    # We introduce a bit of variation to encourage the model to try different
    # paths. This is called an epsilon greedy policy. Here's a good resource
    # for it:
    # https://jamesmccaffrey.wordpress.com/2017/11/30/the-epsilon-greedy-algorithm/

    # The model inputs the observation with either [1, 0] or [0, 1]
    # appended to its end. It will then output the predicted value of either
    # move.
    def policy(self, observation, model_number):
        value = []

        possible_moves = self.env.move_generator()

        if(len(possible_moves) == 0):
            raise Exception("There are no more possible moves that I can take!")

        for i in range(INPUT_SIZE):
            if(observation[i] == 1 or observation[i] == -1):
                value.append(-1)
            else:
                move = list(observation)
                move[i] = 1
                move = np.array([move])
                if(model_number == 1):
                    value.append(self.model_1.predict(move)[0])
                elif(model_number == 0):
                    value.append(self.model_0.predict(move)[0])
                else:
                    raise Exception("Model Number isn't 1 or 0!")

        variation = random.random()
        
        if(variation < 1/self.epsilon):
            self.epsilon += 0.1
            return random.choice(possible_moves)[1]
        else:
            #print(value)
            #print(value.index(max(value)))
            return value.index(max(value))

    def train_model(self, state_array, win_value, model_num, verbose):
        # This contains the final values of each state and action pair in the
        # episode. The model uses this to predict the values more accurately.
        answers = []

        current_reward = win_value
     
        for i in range(len(state_array)):
            answers = [current_reward] + answers
            current_reward *= REWARD_DECAY

        # Preprocessing the state array. We append the actions taken to every
        # state in the state array. This is how we get state-action pairs to
        # feed into the model.

        inputs = np.array(state_array)

        if(False):#self.debugging):
            print("Inputs of Model: Observations and the taken Action")
            print(inputs)
            print("")
            print("Targets of Model: Rewards of each Observation-Action pair")

        if(model_num == 1):
            self.model_1.fit(x = inputs, y = np.array(answers), verbose = verbose)
        elif(model_num == 0):
            self.model_0.fit(x = inputs, y = np.array(answers), verbose = verbose)

    # Saves the model's weights.
    def save(self, s, model_num):
        if(model_num == 1):
            self.model_1.save_weights(s)
        elif(model_num == 0):
            self.model_0.save_weights(s)
        else:
            raise Exception("Model_Number is not 1 or 0!")

    # Loads the weights of a previous model.
    def load(self, s, model_num):
        if(model_num == 1):
            self.model_1.load_weights(s)
        elif(model_num == 0):
            self.model_0.load_weights(s)
        else:
            raise Exception("Model_Number is not 1 or 0!")

    def test(self, model_num):
        observation = self.env.reset()
        
        for i in range(9):
            self.env._render()
            row = int(input("Which Row?"))
            col = int(input("Which Col?"))

            x = 3 * row + col

            observation, reward, done, info = self.env.step([1, x])
            self.env._render()

            board = list(observation['board'])

            # Chose a move and take it
            move = self.policy(board, model_num)

            observation, reward, done, info = self.env.step([-1, move])

    def main(self, path_for_m0 = "", path_for_m1 = ""):
        if(len(path_for_m1) != 0):
            self.load(path_for_m1, 1)

        if(len(path_for_m0) != 0):
            self.load(path_for_m0, 2)

        for i_episode in range(TOTAL_EPISODES):
            observation = self.env.reset()
            board = list(observation['board'])
            
            state_array = [[], []]
            action_array = []

            # Model Number which starts
            model_num = random.choice([0,1])

            for t in range(200):
                if(self.display_img):
                    self.env._render()
                    time.sleep(1)

                state_array[model_num].append(board)

                # Chose a move and take it
                move = self.policy(board, model_num)

                # print(move)

                action = [observation['on_move'], move]

                # print(action)

                observation, reward, done, info = self.env.step(action)
                board = list(observation['board'])
                
                # Check if done. We're only training once we finish the entire
                # episode. Here, the model which makes the last move has number
                # model_num, and the reward it has is reward
                     
                if done:
                    print("Episode finished after {} timesteps".format(t+1)) 
                    
                    self.train_model(state_array[model_num],
                                     reward, model_num, 1)
                    self.train_model(state_array[1-model_num],
                                     -reward, 1-model_num, 1)
                    
                    break

                # Switch
                model_num = 1 - model_num

            # After Every SAVE_FREQUENCY episodes, we save the weights of the
            # model in path.
            if(i_episode % SAVE_FREQUENCY == 0):	
                self.save(self.path + "/TicTacToe_W%d%d" % (i_episode, model_num),
                          model_num)
                self.save(self.path + "/TicTacToe_W%d%d" % (i_episode, 1-model_num),
                          1-model_num)

learning_rate = 0.01
display_img = True
debugging = True
path = "/home/oliver/Desktop"

x = TicTacToe(learning_rate, display_img, debugging, path)
x.load(path + "/TicTacToe_W35000", 0)
x.load(path + "/TicTacToe_W35001", 1)
#x.test(0)
x.main()
