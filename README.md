Turkish Checkers Game Engine Developed for Bachelors Thesis.

==========================================================================

This project implements a Turkish Checkers game engine, including all AI-related code, which was developed entirely by the author. The graphical user interface (GUI) and some game environment components were adapted from an existing Pygame Checkers project.

Some of the game environment parts are taken from GitHub User: everestwitman
https://github.com/everestwitman/Pygame-Checkers/blob/master/checkers.py

Parts taken (detailed information can be found in my thesis work appendix 1): 
GUI (Minor changes are made).
few parts from Class Graphic (Major changes made).
few parts from Class Game (Major changes made).
few parts from Class Board (Major changes made).

Even though the most of the namings can be same/similar most of the code inside is changed due to the games are not the same and it was not compatible with Turkish Checkers Game piece number, movement and other specific rules.
It is only used as starting point but major changes are made to be able to implement the Turkish Checkers Game engine and its AI players/Hybrid solution.


==========================================================================

This repo contains many files, each file having different parts of the thesis covered, for example: 

neural_network_updated_working.py | is for the developed engine against human player, which is the most basic version that humans can interact with.

ab_vs_cnn_ab.py | is for the self-play data creation using CNN model trained with randomly generated dataset. It uses random_model.keras to do CNN evaluations. It generates the files named x_data2.npy and y_data2.npy which is later used to improve the dataset to train the CNN model.

ab_vs_cnn_experiment.py | is for the self-play experiment with randomly generated dataset used to train CNN model. It uses random_model.keras or normal_keras to play against classical alpha-beta pruning algorithm, results of this games not saved but printed at the end of the game loop ( depends on the decided epochs, it will be printed to the console after epochs end)

cnn_ab_vs_cnn_ab.py | is the last version of the hybrid solution that play against other hybrid solution. same as ab_vs_cnn_experiment.py the game results are not saved as additional file but printed to the console.

train_checker_model.ipynb | is the file that first CNN model is trained which uses randomly generated datasets named train_x.npy, train_y.np, test_x.npy, test_y.npy

train_second_checker_model.ipynb |  is the file that second CNN model is trained which uses self-play generated datasets named x_data2.npy, y_data2.npy

generate_random_training_data.py | is the file generates random training data and saves them as files named train_x.npy, train_y.np, test_x.npy, test_y.npy

board.jpg | is the image for game board.

train_x.npy, train_y.np, test_x.npy, test_y.npy | contains randomly generated dataset, in the dataset.zip.

x_data2.npy, y_data2.npy | contains the self-play first hybrid solution against alpha-beta dataset, which is located in the dataset.zip.

random_model.keras | is the trained CNN with random data.

normal_model.keras | is the trained CNN with self-play.
