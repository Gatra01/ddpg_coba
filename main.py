import numpy as np
import matplotlib.pyplot as plt
from env4 import GameState
from ddpg import *
from collections import deque
import torch.nn as nn
import os, shutil
import argparse
from datetime import datetime
from utils import str2bool,evaluate_policy

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default = 15000, help='Max training steps') #aslinya 5e6
parser.add_argument('--save_interval', type=int, default=2500, help='Model saving interval, in steps.') #aslinya 1e5
parser.add_argument('--eval_interval', type=int, default=500, help='Model evaluating interval, in steps.') #aslinya 2e3

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=400, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=2e-3, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-3, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size of training')
parser.add_argument('--random_steps', type=int, default=2000, help='random steps before trianing')
parser.add_argument('--noise', type=float, default=0.1, help='exploring noise') #aslinya 0.1
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device

def main():
    EnvName = ['Power Allocation','LunarLanderContinuous-v2','Humanoid-v4','HalfCheetah-v4','BipedalWalker-v3','BipedalWalkerHardcore-v3']
    #BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3']
    BrifEnvName = ['6G', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3']
    
    # Build Env
    env = GameState(5,3)
    eval_env = GameState(5,3)
    opt.state_dim = env.observation_space
    opt.action_dim = env.action_space
    opt.max_action = env.p_max   #remark: action space【-max,max】
    #print(f'Env:{EnvName[opt.EnvIdex]}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
    #      f'max_a:{opt.max_action}  min_a:{env.action_space.low[0]}  max_e_steps:{env._max_episode_steps}')

    #variable tambahan 
    iterasi = 200
    total_episode = -(-opt.Max_train_steps//iterasi)
    sepertiga_eps=total_episode//3

    
    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Build SummaryWriter to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BrifEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)


    # Build DRL model
    if not os.path.exists('model'): os.mkdir('model')
    agent = DDPG_agent(**vars(opt)) # var: transfer argparse to dictionary
    if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)

    if opt.render:
        
        while True:
            loc = env.generate_positions()
            channel_gain=env.generate_channel_gain(loc)
            state_eval1,inf=env.reset(channel_gain)
            state_eval1 = np.array(state_eval1, dtype=np.float32)
            score = evaluate_policy(channel_gain,state_eval1,env, agent, turns=1)
            
            print('EnvName:', BrifEnvName[opt.EnvIdex], 'score:', score, )
    else:
        total_steps = 0
        lr_steps = 0
        while total_steps < opt.Max_train_steps: # ini loop episode. Jadi total episode adalah Max_train_steps/200
            if lr_steps==sepertiga_eps :
                opt.a_lr=0.3 * opt.a_lr
                opt.c_lr=0.3 * opt.c_lr
                lr_steps=0
            loc= env.generate_positions() #lokasi untuk s_t
            channel_gain=env.generate_channel_gain(loc) #channel gain untuk s_t
            s,info= env.reset(channel_gain, seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False
            langkah = 0
            '''Interact & trian'''
            while not done:  
                langkah +=1
                lr_steps+=1
                if total_steps <= opt.random_steps: #aslinya < aja, ide pengubahan ini tuh supaya selec action di train dulu.
                    a = env.sample_valid_power()
                else: 
                    a = agent.select_action(s, deterministic=False)
                writer.add_scalar("Power node 1", a[0], total_steps)
                writer.add_scalar("Power node 2", a[1], total_steps)
                writer.add_scalar("Power node 3", a[2], total_steps)
                writer.add_scalar("Power node 4", a[3], total_steps)
                writer.add_scalar("Power node 5", a[4], total_steps)
                writer.add_scalar("Total power", a[0]+a[1]+a[2]+a[3]+a[4], total_steps)
                next_loc= env.generate_positions() #lokasi untuk s_t
                next_channel_gain=env.generate_channel_gain(next_loc) #channel gain untuk s_t
                s_next, r, dw, tr, info,EE,rate= env.step(a,channel_gain,next_channel_gain) # dw: dead&win; tr: truncated
                writer.add_scalar("Energi Efisiensi", EE, total_steps)
                writer.add_scalar("Reward iterasi", r, total_steps)
                writer.add_scalar("data rate 1", rate[0], total_steps)
                writer.add_scalar("data rate 2", rate[1], total_steps)
                writer.add_scalar("data rate 3", rate[2], total_steps)
                writer.add_scalar("data rate 4", rate[3], total_steps)
                writer.add_scalar("data rate 5", rate[4], total_steps)
                loc= env.generate_positions()
                channel_gain=env.generate_channel_gain(loc)
                if langkah == iterasi :
                    tr= True
                   
                    
                done = (dw or tr)

                agent.replay_buffer.add(np.array(s, dtype=np.float32), a, r, np.array(s_next, dtype=np.float32), dw)
                s = s_next
                channel_gain=next_channel_gain
                total_steps += 1

                '''train'''
                if total_steps >= opt.random_steps:
                    a_loss, c_loss = agent.train()
                    writer.add_scalar("Loss/Actor", a_loss, total_steps)
                    writer.add_scalar("Loss/Critic", c_loss, total_steps)
                    # print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps/1000)}k, actor_loss:{a_loss}')
                    # print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps/1000)}k, c_loss:{c_loss}')
        
                '''record & log'''
                if total_steps % opt.eval_interval == 0:
                    state_eval,inf=eval_env.reset(channel_gain)
                    state_eval = np.array(state_eval, dtype=np.float32)
                    ep_r = evaluate_policy(channel_gain,state_eval,eval_env, agent, turns=3)
                    if opt.write: 
                        writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                        #writer.add_scalar("Loss/Actor", a_loss.item(), total_steps)
                        #writer.add_scalar("Loss/Critic", q_loss.item(), total_steps)
                        #writer.add_scalar('avg_ee', avg_ee, global_step=total_steps)
                    print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps/1000)}k, Episode Reward:{ep_r}')


                '''save model'''
                if total_steps % opt.save_interval == 0:
                    agent.save(BrifEnvName[opt.EnvIdex], int(total_steps/1000))
                s = s_next
                channel_gain=next_channel_gain
        print("The end")

#%load_ext tensorboard
#%tensorboard --logdir runs
if __name__ == '__main__':
    main()
