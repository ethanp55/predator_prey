This repo contains the code we used for the second case study we performed, which we call "Ad Hoc Teamwork" in our paper.  This domain can be thought of as a "predator-prey" domain where a team of four predators must work together to surround a prey.  The agents in this system (the predators and prey) move on a grid environment where they can only move up, down, left, or right.  A more thorough explanation can be found in our paper.

The repo is broken into 5 key directories:
    
1 - aat: This directory holds the code for the assumptions and checkers we used, along with a training.py file that generates AlegAATr's training data.  The actual training data and trained KNN models are stored in the training_data subdirectory.

2 - agents: This directory contains a python file for each agent we used, including AlegAATr.

3 - environment: The code for the grid environment is stored in this directory.  Specifically, the environment state is contained in state.py, the overall environment is contained in pursuit.py, and code for running the environment can be found in runner.py.  The runner takes as input a list of predators, the desired grid dimensions, and the number of epochs to run.

4 - tests: The code we used for experiments, our statistical tests, and simulations for estimating baseline performances is found in this directory.

5 - utils: Finally, some "helper" code is found in the utils directory.  This basically just contains some helpers that made our simulations/experiments a little easier to write.  Also, the code for the a-star path planning algorithm is contained in this directory (since a few agents use this algorithm to guide their decision-making).