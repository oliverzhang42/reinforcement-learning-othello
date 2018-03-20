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
from keras.layers import BatchNormalization, Dense, Activation, Conv2D, Flatten
from keras.optimizers import Adam
import random
import h5py
import time
import numpy as np
from reversi import reversiBoard 
import copy

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
    # Array is a 8x8 array.
    # ccw rotation

    new_array = []

    for i in range(8):
        row = []
        for j in range(8):
            row.append(array[7-j][i])
        new_array.append(row)

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

        self.model.compile(Adam(learning_rate), "mse")

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

    def main(self, weight_path = ""):
        d = {1: 0, -1: 1}
        
        if(len(weight_path) != 0):
            self.load(weight_path)

        for i_episode in range(TOTAL_EPISODES):
            board = self.env.reset()

            # First array corresponds to the states faced by the first player
            # Same with second
            state_array = [[],[]]

            for t in range(200):
                if(self.display_img):
                    pass
                    #self.env.render()
                
                if(self.debugging):
                    self.env.render()
                    #time.sleep(5)

                # Chose a move and take it
                move = self.policy(board)

                player = self.env.to_play

                
                observation, reward, done, info = self.env.step(move)

                if(self.debugging):
                    print("Move")
                    print(move)
                    print("")

                    print("Observation")
                    print(observation)
                    print("")

                state_array[d[self.env.to_play]].append(observation)
                
                # Check if done. We're only training once we finish the entire
                # episode. Here, the model which makes the last move has number
                # model_num, and the reward it has is reward

                if done:
                    if(reward == 0):
                        print("Draw")
                        
                    print("Episode finished after {} timesteps".format(t+1)) 

                    #print(board)
                    if(self.debugging):
                        print("State Array")
                        #print(state_array)
                        print("")

                    if(len(state_array[0]) == 0):
                        pass


                    self.add_to_history(state_array[0], reward)
                    self.add_to_history(state_array[1], -reward)

                    if(self.debugging):
                        self.train_model(1)
                        self.train_model(1)
                    else:
                        self.train_model(0)
                        self.train_model(0)
                    
                    break

            if(i_episode % WIPE_FREQUENCY == 0):
                self.wipe_history()

            # After Every SAVE_FREQUENCY episodes, we save the weights of the
            # model in path.
            if(i_episode % SAVE_FREQUENCY == 0):	
                self.save(self.path + "ReversiW%d" % (i_episode))

    def display(self):
        pass
        

learning_rate = 0.003
display_img = True
debugging = False
#path = "/Users/student36/Desktop"
path = "/home/oliver/Desktop/"

x = Reversi(learning_rate, display_img, debugging, path)
#x.load(path + "/TicTacToe_W199000.dms", 0)
#x.load(path + "/TicTacToe_W199001.dms", 1)
#x.display()
#x.test(0)
x.main()
