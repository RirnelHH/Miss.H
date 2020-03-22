import numpy as np
import torch
from torch import nn,optim
from torch.distributions import Normal
import os

class PgAgent(nn.Module):
    def __init__(self, EnvActionSpace, EnvObsSpace, NumberLayer, EmbeddingNum):
        self.network = []
        self.parameters = []
        num_inp = EnvObsSpace.shape[0]
        for i in range(NumberLayer):
            self.network.append(nn.Linear(num_inp,EmbeddingNum))
            self.parameters.append([self.network[-1].weight,self.network.bias])
            num_inp = EmbeddingNum
            self.append(nn.ReLU())
        self.mean_layer = nn.Linear(num_inp,EnvActionSpace.shape[0])
        self.log_std_layer = nn.Linear(num_inp,EnvActionSpace.shape[0])
        self.parameters.append([self.mean_layer.weight,self.mean_layer.bias])
        self.parameters.append([self.log_std_layer.weight,self.log_std_layer.bias])
        self.tanh = nn.Tanh()


        self.action_scalar = torch.tensor((env.action_space.high - env.action_space.low)/2,dtype=torch.double)
        self.action_bias = torch.tensor((env.action_space.high + env.action_space.low)/2,dtype=torch.double)
        self.optimizer = optim.Adam(self.parameters,lr = 1e-4)
    
    def forward(self,x:np.ndarray):
        x = torch.from_numpy(x).float().unsqueeze(0)
        for layer in self.network:
            x = layer(x)
        mean = self.mean_layer(x).unsqueeze(0)
        log_std = self.log_std_layer(x).unsqueeze(0)
        log_std = torch.clamp(log_std,-20,2)
        return mean, log_std
        

    def select_action(self,x:np.ndarray):
        mean,log_std = self.forward(x)
        std = log_std.exp()

        m = Normal(mean,std)
        y = m.rsample()
        action = self.tanh(y)
        log_prob = m.log_prob(y)
        log_prob -= torch.log((1-action).pow(2) + epsilon)
        log_prob = log_prob.sum(1,keepdim = True)

        action = action * self.action_scalar + self.action_bias
        mean = mean * self.action_scalar + self.action_bias
        return action,log_prob,mean

class PgModel(object):
    def __init__(self,env_name:str, Agent:PgAgent, lr:np.float64):
        self.env = gym.make(env_name)
        self.Agent = Agent

        self.optim = optim.Adam(self.Agent.parameters,lr = lr)

    def select_action(self,x:np.ndarray):
        return self.Agent.select_action(x)
    

    def train(self,episode_i:np.int64,gamma:np.float64):

        for i in range(episode_i):
            step = 0
            done = False
            obs,actions,rewards = [env.reset()],[],[]
            with torch.no_grad():
                while True:
                    action,log_prob,M_mean = self.select_action(obs[-1])
                    actions.append(action)
                    ob,reward,done,_ = env.step(action)

                    rewards.append(reward)
                    obs.append(ob)
                    step += 1

                    if done:
                        obs.pop()
                        break

            running_fp = 0
            for i in reversed(range(step)):
                running_fp = gamma*running_fp + rewards[i]
                rewards[i] = running_fp
            
            ##Normaliation
            mean = np.mean(rewards)
            std = np.mean(rewards)

            self.optim.zero_grad()
            for i in range(step):
                temp_reward = (reward[i] - mean) / std
                action,log_prob,_ = self.select_action(obs[i])
                loss = temp_reward * log_prob
                loss.backward()

            self.optim.step()
        
    def save(self,path:str):
        assert os.path.exists(path) == False
        torch.save(self.Agent,path)

    def load(self,path:str):
        assert os.path.exists(path) == False
        self.Agent = torch.load(path)
            






        