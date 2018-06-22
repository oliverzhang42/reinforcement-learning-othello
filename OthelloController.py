import random
import time
import numpy as np
from OthelloBoard import * 
import copy
from OthelloBoard import OthelloBoard
from threading import Thread
from keras import backend as K
from OthelloPlayer import reverse, OthelloPlayer, RandomPlayer, BasicPlayer

# Global Variables

# After Every WIPE_FREQUENCY episodes, we wipe the history of the two players.
WIPE_FREQUENCY = 2


class OthelloController:
    def __init__(self, path, population_size, debugging = False, display_img = False,
                 learning_rate = 0.0001, epsilon = 2, epsilon_increment = 0.001):
        self.display_img = display_img
        self.debugging = debugging
        self.path = path

        if(debugging):
            epsilon = 20000

        self.population = [OthelloPlayer(i, 3, self, learning_rate, epsilon,
                                         epsilon_increment, debugging)
                           for i in range(population_size)]

        self.population.append(RandomPlayer())

    def play_two_ai(self, index1, index2):
        return self.play_two_ai_training(index1, index2, False)

    def play_two_ai_training(self, index1, index2, training):
        switch = 0

        if(training):
            switch = random.randint(0, 58)

        # Random Player Index
        rpi = len(self.population) - 1
        
        move_player = [self.population[rpi], self.population[rpi]]
        learn_player = [self.population[index1], self.population[index2]]

        d = {1: 0, -1: 1}
        e = {0: 1, 1: -1}

        env = OthelloBoard(8)
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

                if(len(state_array[0]) == 0):
                    pass

                learn_player[0].add_to_history(state_array[0], reward)
                learn_player[1].add_to_history(state_array[1], -reward)

                return reward
        return reward
        
    def main(self, starting_episode, total_episodes, thread_num, save_frequency,
             random_multithreading = False):
        # For some reason, I need this when multithreading ... Not sure why ...
        K.get_session()
        
        for player in self.population:
            player.depth = 1
        
        #Number of training episodes
        for i in range(starting_episode, starting_episode + total_episodes):
            #One Round Robin Tournament
            for j in range(len(self.population) - 1): #The minus 1 is there for the randomPlayer
                for k in range(len(self.population) - 1):
                    if(random_multithreading):
                        thread_array = []
                        for l in range(thread_num):
                            t = Thread(target = self.play_two_ai_training,
                                       args = (j,k, True))
                            t.start()
                            thread_array.append(t)

                        for t in thread_array:
                            t.join()
                    else:
                        self.play_two_ai(j,k)
                    
            #Everyone Trains
            for j in range(len(self.population) - 1):
                if(random_multithreading):
                    for k in range(2*thread_num):
                        self.population[j].train_model(self.debugging)
                else:
                    self.population[j].train_model(self.debugging)

            if(i % save_frequency == 0):
                print(i)
                self.save([i])

            if(i % WIPE_FREQUENCY == 0):
                for j in range(len(self.population) - 1):
                    self.population[j].wipe_history()
        
    def save(self, episode_numbers):
        for j in range(len(self.population) - 1):
            self.population[j].save(self.path + "Reversi_%d" %
                                    (episode_numbers[j]))

    def load(self, episode_numbers):
        for j in range(len(self.population) - 1):
            self.population[j].load(self.path + "Reversi_%d" %
                                    (episode_numbers[j]))
