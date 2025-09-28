
from torch.distributions import Normal
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np 
import torch
import time
import copy

# Custom class
from SAC.ReplayBuffer import ReplayBuffer
from SAC.ActorCritic import Actor , Critic



class Agent():
    def __init__(self,args,env,hidden_layer_num_list=[64,64]):

        # Hyperparameter
        self.evaluate_freq_steps = args.evaluate_freq_steps
        self.max_train_steps = args.max_train_steps
        self.num_actions = args.num_actions
        self.batch_size = args.batch_size
        self.num_states = args.num_states
        self.mem_min = args.mem_min
        self.gamma = args.gamma
        self.alpha = 0.2
        self.set_var = args.var
        self.var = self.set_var
        self.tau = args.tau
        self.lr = args.lr
        self.d = args.d

        # Variable
        self.total_steps = 0
        self.training_count = 0
        self.evaluate_count = 0

        # other
        self.env = env
        self.action_max = env.action_space.high[0]
        self.replay_buffer = ReplayBuffer(args)
        

        # Actor-Critic
        self.actor = Actor(args,hidden_layer_num_list.copy())
        self.critic1 = Critic(args,hidden_layer_num_list.copy())
        self.critic2 = Critic(args,hidden_layer_num_list.copy())
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.optimizer_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=self.lr, eps=1e-5)
        self.optimizer_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=self.lr, eps=1e-5)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)

        print(self.actor)
        print(self.critic1)
        print(self.critic2)
        print("-----------")

    def choose_action(self,state):

        state = torch.tensor(state, dtype=torch.float)

        s = torch.unsqueeze(state,0)
        with torch.no_grad():
            a = self.actor.sample(s)[0]

        return a.cpu().numpy().flatten() 

    def evaluate_action(self,state):

        state = torch.tensor(state, dtype=torch.float)

        s = torch.unsqueeze(state,0)
        with torch.no_grad():
            a = self.actor.sample(s)[0]

        return a.cpu().numpy().flatten() 

    def evaluate_policy(self, env , render = False):
        times = 10
        evaluate_reward = 0
        for i in range(times):
            s, info = env.reset()
            
            done = False
            episode_reward = 0
            while True:
                a = self.evaluate_action(s)  # We use the deterministic policy during the evaluating
            
                s_, r, done, truncted, _ = env.step(a)

               
                episode_reward += r
                s = s_
                #print(episode_reward)
                if truncted or done:
                    break
            evaluate_reward += episode_reward

        return evaluate_reward / times

    def var_decay(self, total_steps):
        new_var = self.set_var * (1 - total_steps / self.max_train_steps)
        self.var = new_var + 10e-10
        
    def train(self):
        time_start = time.time()
        episode_reward_list = []
        episode_count_list = []
        episode_count = 0
        
        # Training Loop
        while self.total_steps < self.max_train_steps:
            s = self.env.reset()[0]            
            while True:
                a = self.choose_action(s)
                s_, r, done , truncated , _ = self.env.step(a)
                done = done or truncated


                # storage data
                self.replay_buffer.store(s, a, [r], s_, done)
                
                # update state
                s = s_

                if self.replay_buffer.count >= self.mem_min:
                    self.training_count += 1
                    self.update()

                if self.total_steps % self.evaluate_freq_steps == 0:
                    self.evaluate_count += 1
                    evaluate_reward = self.evaluate_policy(self.env)
                    episode_reward_list.append(evaluate_reward)
                    episode_count_list.append(episode_count)
                    time_end = time.time()
                    h = int((time_end - time_start) // 3600)
                    m = int(((time_end - time_start) % 3600) // 60)
                    second = int((time_end - time_start) % 60)
                    print("---------")
                    print("Time : %02d:%02d:%02d"%(h,m,second))
                    print("Training episode : %d\tStep : %d / %d"%(episode_count,self.total_steps,self.max_train_steps))
                    print("Evaluate count : %d\tEvaluate reward : %0.2f"%(self.evaluate_count,evaluate_reward))

                self.total_steps += 1
                if done or truncated:
                    break
            episode_count += 1
        # Plot the training curve
        plt.plot(episode_count_list, episode_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Curve")
        plt.show()
            
    def update(self):
        s, a, r, s_, done = self.replay_buffer.numpy_to_tensor()  # Get training data .type is tensor    

        index = np.random.choice(len(r),self.batch_size,replace=False)

        # Get minibatch
        minibatch_s = s[index]
        minibatch_a = a[index]
        minibatch_r = r[index]
        minibatch_s_ = s_[index]
        minibatch_done = done[index]
        
        # Get target value (Maximum Entropy)
        with torch.no_grad():
            next_action , next_log_prob = self.actor.sample(minibatch_s_)
            next_value1 = self.critic1_target(minibatch_s_,next_action)
            next_value2 = self.critic2_target(minibatch_s_,next_action)
            target_value = torch.min(next_value1,next_value2) - self.alpha * next_log_prob
            
        # Update Critic 1
        value1 = self.critic1(minibatch_s , minibatch_a)
        critic1_loss = F.mse_loss(value1 , target_value)
        self.optimizer_critic1.zero_grad()
        critic1_loss.backward()
        self.optimizer_critic1.step()
        
        # Update Critic 2
        value2 = self.critic2(minibatch_s,minibatch_a)
        critic1_loss = F.mse_loss(value2 , target_value)
        self.optimizer_critic2.zero_grad()
        critic1_loss.backward()
        self.optimizer_critic2.step()
        
        # update Actor
        action , log_prob = self.actor.sample(minibatch_s)
        value1 = self.critic1(minibatch_s,action)
        value2 = self.critic2(minibatch_s,action)
        min_value = torch.min(value1,value2)
        actor_loss =  (min_value - self.alpha * log_prob).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        
        # Update target networks
        if self.total_steps % self.d == 0 :         
            self.soft_update(self.critic1_target,self.critic1, self.tau)
            self.soft_update(self.critic2_target,self.critic2, self.tau)
            self.soft_update(self.actor_target, self.actor, self.tau)    

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
    
        