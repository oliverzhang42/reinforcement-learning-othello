import sys
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string("mode", None, "Mode, 't' for train, 'h' for play vs human, and 'm' for play vs machine")
flags.DEFINE_string("path", None, "Path, specify the folder which you want to save/read weights.")
flags.DEFINE_float("lr", 0.000001, "Learning rate, only matters if you're training. Since the neural net is pretty deep, it has to be small.")
flags.DEFINE_boolean("random", True, "Determines whether there will be randomness in your training.")
flags.DEFINE_integer("save_frequency", 50, "After how many iterations do you want to save?")
flags.DEFINE_list("load_weight", [-1], "The list of weights to load. If training or vs human\
make the list one length. If play vs machine, make it length 2.")
flags.DEFINE_integer("total_episodes", 10000, "What's the total number of episodes do you want to run? Only used if training")

flags.mark_flag_as_required("mode")
flags.mark_flag_as_required("path")

from OthelloController import OthelloController
from OthelloAgainstAI import OthelloSession
from OthelloArena import Arena


def main(argv):
    del argv # Unused.
    
    FLAGS.load_weight[0] = FLAGS.load_weight[0].replace('[','')
    FLAGS.load_weight[0] = FLAGS.load_weight[0].replace(']','')    
    
    if FLAGS.mode == 't':
        # Train
        if FLAGS.random:
            epsilon = 5
        else:
            epsilon = 10000000
        
        controller = OthelloController(FLAGS.path, 1, learning_rate = FLAGS.lr, epsilon = epsilon)
        
        if(FLAGS.load_weight[0] != -1):
            controller.load([int(FLAGS.load_weight[0])])
        
        controller.main(int(FLAGS.load_weight[0]), FLAGS.total_episodes, FLAGS.save_frequency)
        
    elif FLAGS.mode == 'h':
        session = OthelloSession(FLAGS.path)
        session.play(int(FLAGS.load_weight[0]))
        
    elif FLAGS.mode == 'm':
        FLAGS.load_weight[1] = FLAGS.load_weight[1].replace('[','')
        FLAGS.load_weight[1] = FLAGS.load_weight[1].replace(']','')
        
        arena = Arena(FLAGS.path)
        arena.play(int(FLAGS.load_weight[0]), int(FLAGS.load_weight[1]))
        

if __name__ == '__main__':
    app.run(main)
