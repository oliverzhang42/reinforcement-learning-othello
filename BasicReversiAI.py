import random
from reversi import reversiBoard 
import numpy as np
env = reversiBoard(8)
env.reset()
env.render()

#'''
for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        enables = env.move_generator()
        print(enables)
        # if nothing to do ,select pass
        if len(enables)==0:
            action = (-1, -1)
        # random select (update learning method here)
        else:
            action = random.choice(enables)
        observation, reward, done, info = env.step(action)
        #print(action)
        #print(observation)
        env.render()
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print(reward)
            break
#'''

