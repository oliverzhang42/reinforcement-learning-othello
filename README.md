### Reinforcement Learning Othello

This is a project on reinforcement learning. It employs Monte Carlo learning to tackle the game of Othello. Essentially, it plays games against itself and records those games. Then after each game, it sees which player won and uses that information to get better.

In more technical terms, the model is doing state-value-approximation. Each state is a different board state and the approximator function is a six-layerd Convolutional Neural Network with resnet set up. When playing against a human or another AI, it also applies a three-layered Alpha-Beta search. 

## Requirements

The library keras is required. It can be installed at https://keras.io/#installation. The package absl was also used for the command line interface, but it isn't necessary as long as you only run the script, but if you want to install it go here: https://github.com/abseil/abseil-py.

## How to run?

There are two files that you can interface with. OthelloInterface is a file that uses the command line to interface. Help can be found using the "--helpshort" tag, and you input arguments by using "python3 OthelloInterface.py --Var1 value1 --Var2 value2".

If you aren't as fluent with the command line, OthelloScript is a script that you can run. Simply modify the variables at the top and run the script.

## Class Structure:

My framework can be imagined as a simple layered tower. 

# Ground Floor: OthelloBoard

At the lowest level, there is OthelloBoard.py with the OthelloBoard class. This code is adapted from http://code.activestate.com/recipes/580698-reversi-othello/. Many thanks to them for enabling this project to happen. 

# Second Level: AlphaBeta

At the second level we have AlphaBeta.py with the AlphaBeta class. This class is meant to perform the AlphaBeta algorithm, and that's it.

# Third Level: OthelloPlayer

At the third level we have OthelloPlayer.py with the OthelloPlayer class. This class encapsulates an individual player. Each player is based around a neural network and a history. Policy() describes what the neural network thinks is the best move. Train_model() randomly samples the history and trains the network. Finally, Wipe_history() and add_to_history() manipulate the history.

# Fourth Level: OthelloController

At the fourth level we have OthelloController.py with the OthelloController class. If the OthelloPlayer class are players at a tournament, the OthelloController class is like the tournament host. It manages the playing of two players in play_two_ai() and also the arranging of matches in main(). Note: OthelloController was designed for only one learning player in mind. The reason why it has a population array instead of a single resident is that the other players can be RandomPlayers or BasicPlayers, simpler functions which don't require .load() or .save().

# Fifth Level: OthelloArena and OthelloAgainstAI

At the second to last level, we have OthelloArena.py and OthelloAgainstAI.py. OthelloArena constructs an OthelloController with two AI and plays them against each other. OthelloAgainstAI constructs an OthelloController with one AI and launches an interface allowing you to play it.

# Sixth Level: OthelloInterface and OthelloScript

Finally, we have the highest level, namely OthelloInterface and OthelloScript. These take your inputs and runs OthelloArena.py or OthelloAgainstAI.py. 
