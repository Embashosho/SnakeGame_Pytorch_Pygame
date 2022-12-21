# SnakeGame_Pytorch_Pygame
A 2D snake game where the snake agent learns to predict the best moves to improve its score using reinforcement learning. 


In this Python Reinforcement Learning project we teach an AI to play Snake! We build everything from scratch using Pygame and PyTorch and some other basic python libraries
firstly, we setup the environment and implement the Snake game.
then we implement the agent that controls the game.
Lastly, we implement the neural network to predict the moves and train it.

The snakegame is terminated immidiately it touches any of the boundaries or when it touches any part of its body.
the lenght increases by  a unit size anytimes it touches the food afterwhich another food is respawned at a random place in the game.
Additionally, the game is quickly reset and started after terminating the previous game, this is done to continously train the agent.

