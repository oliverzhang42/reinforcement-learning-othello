#Note: Comments are wrong right now
# Things to do:
# 1. Create the right type of model
# 2. Update Policy (Draws)
# 3. Find a way to have a "population" of reversi AI's?
#    Actually, it seemed that AlphaZero had only one AI playing itself.
# 4. Update all functions :D
# 5. Turns out that I actually need a way to test my AI. Hmmmm
# 6. Do I even need a history???

# Debugging:
# 1. Run program for a while.
# 2. Print out Value Function
# 3. Print out Loss
# 4. Test Behavior
# 5. 

import keras
from keras.layers import BatchNormalization, Dense, Activation, Conv2D
from keras.optimizers import Adam
import random
import h5py
import time
import numpy as np
from reversi import reversiBoard 


# After Every SAVE_FREQUENCY episodes, we save the weights of the model in path.
SAVE_FREQUENCY = 100

WIPE_FREQUENCY = 10

# The number of total episodes to run.
TOTAL_EPISODES = 20000

# The size of each layer in the model.
LAYER_SIZE = 30

BOARD_SIZE = 8

# Here, REWARD_DECAY is how much we care about the delayed reward compared to
# the immediate reward. REWARD_DECAY = 1 means we care about all reward the
# same, REWARD_DECAY = 0 means we don't care at all about the later rewards.
REWARD_DECAY = 0.99

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

class Reversi:
    def __init__(self, learning_rate, display_img, debugging, path):
        self.display_img = display_img
        self.debugging = debugging
        self.path = path

        # self.env is the implementation of the CartPole game. Code:
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        self.env = reversiBoard(BOARD_SIZE)

        # Epsilon sometimes randomizing the player's actions. Helps with
        # exploration of more possibilities.
        
        self.epsilon = 2

        if(debugging):
            self.epsilon = 2000000

        self.create_model()

        self.experience = []


    # This Function Creates a Keras Model with three sections of:
    # a Batch Norm Layer, a Dense layer, and an Activation.
    # There a fourth section with no activation because the output
    # isn't limited in a 0-1 range.
    def create_model(self):
        self.model = keras.models.Sequential()
        self.model.add(Dense(LAYER_SIZE, input_shape = (INPUT_SIZE,)))
        self.model.add(Activation("tanh"))

        self.model.add(Dense(LAYER_SIZE, input_shape = (INPUT_SIZE,)))
        self.model.add(Activation("tanh"))

        self.model.add(Dense(LAYER_SIZE, input_shape = (INPUT_SIZE,)))
        self.model.add(Activation("tanh"))

        self.model.add(Dense(1))

        self.model.compile(loss='mse', optimizer = Adam(learning_rate))

    # Policy is how the model picks an action for a given situation and weights.
    # This is not training.

    # We introduce a bit of variation to encourage the model to try different
    # paths. This is called an epsilon greedy policy. Here's a good resource
    # for it:
    # https://jamesmccaffrey.wordpress.com/2017/11/30/the-epsilon-greedy-algorithm/

    # The model inputs the observation with either [1, 0] or [0, 1]
    # appended to its end. It will then output the predicted value of either
    # move.
    def policy(self, observation):
        # Value is an array. The 0th element corresponds to (0,0), the 1st: (0,1)
        # the 8th: (1,0), etc.
        value = []

        possible_moves = self.env.move_generator()

        if(len(possible_moves) == 0):
            # Passes
            return (-1, -1)

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if(observation[i][j] == 1 or observation[i][j] == -1):
                    value.append(-1)
                else:
                    move = list(observation)
                    # We always assume that we're player 1.
                    move[i] = 1
                    move = np.array([move])
                    value.append(self.model.predict(move)[0])

        variation = random.random()
        
        if(variation < 1/self.epsilon):
            self.epsilon += 0.0001
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

    def add_to_history(self, state_array, reward):
        answers = []
        history = self.experience

        current_reward = reward
     
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
        self.experience = []

        print("WIPE!")

        # Preprocessing the state array. We append the actions taken to every
        # state in the state array. This is how we get state-action pairs to
        # feed into the model.

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

    # Have not updated yet
    def test(self):
        observation = self.env.reset()
        
        for i in range(9):
            self.env.render()
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

    def main(self, weight_path):
        d = {1: 0, -1: 1}
        
        if(len(weight_path) != 0):
            self.load(weight_path)

        for i_episode in range(TOTAL_EPISODES):
            observation = self.env.reset()
            board = list(observation['board'])

            # First array corresponds to the states faced by the first player
            # Same with second
            state_array = [[],[]]

            for t in range(200):
                if(self.display_img):
                    pass
                    #self.env.render()
                
                if(self.debugging):
                    self.env.render()
                    time.sleep(5)

                # Chose a move and take it
                move = self.policy(board)

                state_array[d[self.env.to_play]].append(board)

                # print(move)

                observation, reward, done, info = self.env.step(move)
                board = list(observation['board'])
                
                # Check if done. We're only training once we finish the entire
                # episode. Here, the model which makes the last move has number
                # model_num, and the reward it has is reward

                if done:
                    if(reward == 0):
                        print("Draw")
                        
                    print("Episode finished after {} timesteps".format(t+1)) 

                    #print(board)


                    self.add_to_history(state_array[0], reward)
                    self.add_to_history(state_array[1], -reward)

                    if(self.debugging):
                        self.train_model(0, 1)
                        self.train_model(1, 1)
                    else:
                        self.train_model(0, 0)
                        self.train_model(1, 0)
                    
                    break

            if(i_episode % WIPE_FREQUENCY == 0):
                self.wipe_history()

            # After Every SAVE_FREQUENCY episodes, we save the weights of the
            # model in path.
            if(i_episode % SAVE_FREQUENCY == 0):	
                self.save(self.path + "/TicTacToe_W%d%d" % (i_episode))

    def display(self):
        pass
        

learning_rate = 0.003
display_img = True
debugging = False #True
path = "/Users/student36/Desktop/TicTacToe1"
#path = "/home/oliver/Desktop/TicTacToe2"

x = TicTacToe(learning_rate, display_img, debugging, path)
x.load(path + "/TicTacToe_W199000.dms", 0)
x.load(path + "/TicTacToe_W199001.dms", 1)
#x.display()
x.test(0)

#x.main()
