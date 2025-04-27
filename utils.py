import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch
import numpy as np
from env4 import GameState

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.maxaction = maxaction

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.maxaction #aslinya tuh tanh
        return a


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Q_Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q

def evaluate_policy(channel_gain,state, env, agent, turns = 3):
    env = GameState(5,3)   
    total_scores = 0
    total_data_rate = 0
    total_power = 0
    total_EE=0
   
    for j in range(turns):
        #s, info = env.ini()
        done = False
        MAX_STEPS = 1  # Batas maksimum langkah per episode
        step_count = 0
        a=np.zeros(5)
        while not done:
            step_count += 1
            print(step_count)
            
            # Take deterministic actions at test time
            a = agent.select_action(state, deterministic=True) #aslinya True
            
            print(a)
            
            next_loc= env.generate_positions() #lokasi untuk s_t
            next_channel_gain=env.generate_channel_gain(next_loc) #channel gain untuk s_t
            s_next, r, dw, tr, info,EE,rate = env.step(a,channel_gain,next_channel_gain)
            
            if step_count==MAX_STEPS:
                tr=True
            done = (dw or tr)

            total_scores += r
            state = s_next
            channel_gain=next_channel_gain
    return (total_scores/turns)

#Just ignore this function~
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
