import numpy as np
import torch
from torch import nn,optim
from torch.distributions import Normal

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
        return action,log_prob,mean



        