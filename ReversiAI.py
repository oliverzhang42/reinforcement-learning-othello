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
TOTAL_EPISODES = 23000

# The size of each layer in the model. Currently Unused
LAYER_SIZE = 30

# The size of the othello board. (BOARD_SIZE by BOARD_SIZE)
BOARD_SIZE = 8

# Here, REWARD_DECAY is how much we care about the delayed reward compared to
# the immediate reward. REWARD_DECAY = 1 means we care about all reward the
# same, REWARD_DECAY = 0 means we don't care at all about the later rewards.
REWARD_DECAY = 0.95

BATCH_SIZE = 64

# Episodes before switching which model to train
EPISODES_BEFORE_SWITCH = 200

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
    def __init__(self, learning_rate = 0.00005, epsilon = 2,
                 epsilon_increment = 0.00005, debugging = False):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_increment = epsilon_increment
        self.experience = []
        self.debugging = debugging

        self.create_model()

    def create_model(self):
        self.model = keras.models.Sequential()

        self.model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same',
                              input_shape = (3,8,8)))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(256, activation = 'relu'))
        self.model.add(Dense(1, activation = 'tanh'))

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

    def policy(self, observation, env):
        # Value is an array. The 0th element corresponds to (0,0), the 1st: (0,1)
        # the 8th: (1,0), etc.
        value = []

        possible_moves = env.move_generator()

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
                    move = process(move)
                    move = np.array([move])
                    value.append(self.model.predict(move)[0][0])
                else:
                    value.append(-1)

        variation = random.random()
        
        if(variation < 1/self.epsilon):
            self.epsilon += self.epsilon_increment
            if(self.debugging):
                print("Random Move for player " + str(env.to_play))
            return random.choice(possible_moves)
        else:
            if(self.debugging):
                print(np.array(value).tolist())
            index = value.index(max(value))
            action = (index // 8,index % 8)
            return action

class ReversiController:
    def __init__(self, path, display_img, debugging, population_size,
                 learning_rate = 0.0001, epsilon = 2, epsilon_increment = 0.001):
        self.env = reversiBoard(BOARD_SIZE)
        self.display_img = display_img
        self.debugging = debugging
        self.path = path

        if(debugging):
            epsilon = 20000

        self.population = [ReversiPlayer(learning_rate, epsilon,
                                         epsilon_increment, debugging)
                           for i in range(population_size)]

    def play_two_ai(self, index1, index2):
        player = [self.population[index1], self.population[index2]]

        d = {1: 0, -1: 1}

        observation = self.env.reset()

        # First array corresponds to the states faced by the first player
        # Same with second
        state_array = [[],[]]

        for t in range(200):
            if(self.display_img):
                self.env.render()
            
            if(self.debugging):
                self.env.render()
                #time.sleep(5)

            # Chose a move and take it
            move = player[t % 2].policy(observation, self.env)
            
            observation, reward, done, info = self.env.step(move)

            if(self.debugging):
                print("Move")
                print(move)
                print("")

                print("Observation")
                print(observation)
                print("")
            
            if(not done):
                state_array[d[self.env.to_play]].append(observation)

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

                player[0].add_to_history(state_array[0], reward)
                player[1].add_to_history(state_array[1], -reward)

                break
        
    def test(self):
        observation = self.env.reset()
        
        for i in range(64):
            board = copy.deepcopy(observation)
            
            self.env.render()
            
            # Chose a move and take it
            move = self.population[0].policy(board, self.env)

            observation, reward, done, info = self.env.step(move)

            print(done)

            if(done):
                print("End of Game")
                break

            self.env.render()

            row = int(input("Which Row?"))
            col = int(input("Which Col?"))

            action = (row, col)

            observation, reward, done, info = self.env.step(action)


            if(done):
                print("End of Game")
                break
        

    def main(self, total_episodes):
        
        #Number of training episodes
        for i in range(total_episodes):
            #One Round Robin Tournament
            for j in range(len(self.population)):
                for k in range(len(self.population)):
                    self.play_two_ai(j,k)

            #Everyone Trains
            for j in range(len(self.population)):
                self.population[j].train_model(self.debugging)

            if(i % SAVE_FREQUENCY == 0):
                print(i)
                self.save([i])

            if(i % WIPE_FREQUENCY == 0):
                for j in range(len(self.population)):
                    self.population[j].wipe_history()
        
    def save(self, episode_numbers):
        for j in range(len(self.population)):
            self.population[j].save(self.path + "Reversi_%d_%d" %
                                    (j, episode_numbers[j]))

    def load(self, episode_numbers):
        for j in range(len(self.population)):
            self.population[j].load(self.path + "Reversi_%d_%d" %
                                    (j, episode_numbers[j]))

#path = "/Users/student36/Desktop/Reversi1/"
path = "/home/oliver/Desktop/Reversi3/"

x = ReversiController(path, False, False, 1)

#for i in range(99):
#    x.load([(i + 1) * 100, 19900])
#    x.play_two_ai(0,1)

x.load([19900])
#x.test()
x.main(TOTAL_EPISODES)
