# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 08:47:53 2022

@author: A
"""

import torch
import random
import numpy as np

from snakegame import SnakeGameAI, Direction, Point
from collections import deque
from model import Linear_Qnet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    
    def __init__(self):
        
        self.n_game = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9    #discount rate
        self.memory = deque(maxlen= MAX_MEMORY)  #pop left is called if number in deque exceeds MAX_MEMORY
        self.model = Linear_Qnet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma= self.gamma)
        
        #model,trainer
        
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        
        state = [
            
            #Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)) ,
            
            
            #Danger Right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_r and game.is_collision(point_u)) ,
            
            
            #Danger Left
            (dir_r and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_d)) ,
            
            #move directions
            dir_d,
            dir_l,
            dir_u,
            dir_r,
            
            
            #food direction
            game.food.x < game.head.x, #food is left
            game.food.x > game.head.x, #food is right
            game.food.y < game.head.y, #food is up
            game.food.y > game.head.y,  #food is down
            
            ]
        
        return np.array(state, dtype= int)
    
    
    
    def get_state(self, game):
        
        pass
    
    
    def remember(self, state, action, reward, next_state, done ):
        
        self.memory.append((state, action, reward, next_state, done)) #popleft is max memory is reached 
        
        
    
    def train_long_memory(self):
        
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples 
        
        else:
            mini_sample = self.memory
            
            
        states,actions,rewards,next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    
    
    def train_short_memory(self, state,action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        #random moves: tradeoff between exploration and exploitation
        self.epsilon = 80 - self.n_game
        final_move = [0, 0, 0]
        
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move]= 1
        else:
            state0 = torch.tensor(state, dtype=torch.int)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move]= 1
            
        return final_move
        
        
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent= Agent()
    game = SnakeGameAI()
    
    while True:
        
        #get old state
        state_old = agent.get_state(game)
        
        #get move
        final_move = agent.get_action(state_old)
        
        #perfom move and get new state
        reward, done, score = game.play_step(final_move) 
        state_new = agent.get_state(game)
        
        
        #train_short_memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        
        #remember
        agent.remember(state, action, reward, next_state, done)
        
        if done:
            #train_long_memory or called replay, plot result
            
            game.reset()
            agent.n_game += 1
            agent.train_long_memory()
            
            
            #check highscore
            if score > record:
                
                record = score
                agent.model.save()
                
            print('Game', agent.n_game, 'Score', score, 'Record:', record)
            
            #Plot
        
            plot_scores.append(score)
            total_score += score
            mean_score = total_score/ agent.n_game
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            
            
        


if __name__ == '__main__':
    
    train()
    