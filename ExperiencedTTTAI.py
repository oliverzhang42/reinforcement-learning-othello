#Note: Comments are wrong right now
# Things to do:
# 1. Run program for a while.
# 2. Print out Value Function
# 3. Print out Loss
# 4. Test Behavior
# 5. 

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

WIPE_FREQUENCY = 10

# The number of total episodes to run.
TOTAL_EPISODES = 20000

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

def rotate_90(array):
    # Assume array is 3x3
    # ccw rotation
    order = [2, 5, 8, 1, 4, 7, 0, 3, 6]
    new_array = []

    for i in order:
        new_array.append(array[i])

    return new_array

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

        if(debugging):
            self.epsilon = 2000000

        self.create_model_1()
        self.create_model_0()

        self.experience_0 = []
        self.experience_1 = []


    # This Function Creates a Keras Model with three sections of:
    # a Batch Norm Layer, a Dense layer, and an Activation.
    # There a fourth section with no activation because the output
    # isn't limited in a 0-1 range.
    def create_model_1(self):
        self.model_1 = keras.models.Sequential()
        self.model_1.add(Dense(LAYER_SIZE, input_shape = (INPUT_SIZE,)))
        self.model_1.add(Activation("tanh"))

        self.model_1.add(Dense(LAYER_SIZE, input_shape = (INPUT_SIZE,)))
        self.model_1.add(Activation("tanh"))

        self.model_1.add(Dense(LAYER_SIZE, input_shape = (INPUT_SIZE,)))
        self.model_1.add(Activation("tanh"))
        
        self.model_1.add(Dense(1))

        self.model_1.compile(loss='mse', optimizer = Adam(learning_rate))
    
    def create_model_0(self):
        self.model_0 = keras.models.Sequential()
        self.model_0.add(Dense(LAYER_SIZE, input_shape = (INPUT_SIZE,)))
        self.model_0.add(Activation("tanh"))

        self.model_0.add(Dense(LAYER_SIZE, input_shape = (INPUT_SIZE,)))
        self.model_0.add(Activation("tanh"))

        self.model_0.add(Dense(LAYER_SIZE, input_shape = (INPUT_SIZE,)))
        self.model_0.add(Activation("tanh"))

        self.model_0.add(Dense(1))

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
        d = {0: -1, 1: 1}
        value = []

        possible_moves = self.env.move_generator()

        if(len(possible_moves) == 0):
            raise Exception("There are no more possible moves that I can take!")

        for i in range(INPUT_SIZE):
            if(observation[i] == 1 or observation[i] == -1):
                value.append(-1)
            else:
                move = list(observation)
                move[i] = d[model_number]
                move = np.array([move])
                if(model_number == 1):
                    value.append(self.model_1.predict(move)[0])
                elif(model_number == 0):
                    value.append(self.model_0.predict(move)[0])
                else:
                    raise Exception("Model Number isn't 1 or 0!")

        variation = random.random()
        
        if(variation < 1/self.epsilon):
            self.epsilon += 0.0001
            #print(self.epsilon)
            if(self.debugging):
                print("Random Move for player " + str(model_number))
            return random.choice(possible_moves)[1]
        else:
            if(self.debugging):
                print(np.array(value).tolist())
            #print(value.index(max(value)))
            return value.index(max(value))

    def add_to_history(self, state_array, win_value, model_num):
        answers = []

        if(model_num == 0):
            history = self.experience_0
        elif(model_num == 1):
            history = self.experience_1
        else:
            raise Exception("Model_Num is not 0 or 1!!!")

        current_reward = win_value
     
        for i in range(len(state_array)):
            current_array = state_array[len(state_array) - i - 1]
            
            history.append([current_array,
                                 current_reward])
            current_array = rotate_90(current_array)
            history.append([current_array,
                                 current_reward])
            current_array = rotate_90(current_array)
            history.append([current_array,
                                 current_reward])
            current_array = rotate_90(current_array)
            history.append([current_array,
                                 current_reward])
            current_reward *= REWARD_DECAY

        #print(model_num)
        #print(history)

    def wipe_history(self):
        self.experience_0 = []
        self.experience_1 = []

        print("WIPE!")

        # Preprocessing the state array. We append the actions taken to every
        # state in the state array. This is how we get state-action pairs to
        # feed into the model.

    def train_model(self, model_num, verbose):
        inputs = []
        answers = []

        if(model_num == 0):
            history = self.experience_0
        elif(model_num == 1):
            history = self.experience_1
        else:
            raise Exception("Model_Num is not 0 or 1!!!")
          
        for i in range(BATCH_SIZE):
            lesson = random.choice(history)
            inputs.append(lesson[0])
            answers.append(lesson[1])

        inputs = np.array(inputs)
        answers = np.array(answers)

        #print(model_num)
        #print(inputs)
        #print(answers)
        
        if(model_num == 1):
            self.model_1.fit(x = inputs, y = answers, verbose = verbose)
        elif(model_num == 0):
            self.model_0.fit(x = inputs, y = answers, verbose = verbose)

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

            if(done):
                print("End of Game")
                break

            # Chose a move and take it
            move = self.policy(board, model_num)

            observation, reward, done, info = self.env.step([-1, move])

            if(done):
                print("End of Game")
                break

    def main(self, path_for_m0 = "", path_for_m1 = ""):
        d = {0: -1, 1:1}
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

            self.env.state['on_move'] = d[model_num]

            if(self.debugging):
                print("Starting Model " + str(model_num))

            for t in range(200):
                if(self.display_img):
                    pass
                    #self.env._render()
                
                if(self.debugging):
                    self.env._render()
                    time.sleep(5)

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

                state_array[model_num].append(board)
                
                if done:
                    if(reward == 0):
                        print("Draw")
                        
                    print("Episode finished after {} timesteps".format(t+1)) 

                    #print(board)


                    self.add_to_history(state_array[0], -reward, 0)
                    self.add_to_history(state_array[1], reward, 1)

                    if(self.debugging):
                        self.train_model(0, 1)
                        self.train_model(1, 1)
                    else:
                        self.train_model(0, 0)
                        self.train_model(1, 0)
                    
                    break

                # Switch
                model_num = 1 - model_num

            if(i_episode % WIPE_FREQUENCY == 0):
                self.wipe_history()

            # After Every SAVE_FREQUENCY episodes, we save the weights of the
            # model in path.
            if(i_episode % SAVE_FREQUENCY == 0):	
                self.save(self.path + "/TicTacToe_W%d%d" % (i_episode, model_num),
                          model_num)
                self.save(self.path + "/TicTacToe_W%d%d" % (i_episode, 1-model_num),
                          1-model_num)

    def display(self):
        print("Printing Model 0:")
        for layer in self.model_0.layers:
            print("New Layer")
            print(layer.get_weights()) # list of numpy arrays
        
        print("Printing Model 1:")
        for layer in self.model_1.layers:
            print("New Layer")
            print(layer.get_weights()) # list of numpy arrays
        

learning_rate = 0.003
display_img = True
debugging = False #True
#path = "/Users/student36/Desktop/TicTacToe3"
path = "/home/oliver/Desktop/TicTacToe2"

x = TicTacToe(learning_rate, display_img, debugging, path)
x.load(path + "/TicTacToe_W199000", 0)
x.load(path + "/TicTacToe_W199001", 1)
#x.display()
x.test(0)

#x.main()
