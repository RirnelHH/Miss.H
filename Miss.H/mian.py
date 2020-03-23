import numpy as np
import sys
from ops.Behavior_clone import *
from ops.Preference_learning import *
import os
from torch.utils.data import DataLoader

Game_name = "Humanoid-v2"
env = gym.make(Game_name)

work_space_path = os.getcwd()+"/Miss.H"
AgentPath = "models"
AgentPath = os.path.join(work_space_path,AgentPath)
AgentFile = os.listdir(AgentPath)
AgentFile = sorted(AgentFile,key = lambda x : int(x.split('_')[-3]))
Agents = [RandomAgent(env.action_space)]

for i in AgentFile:
    abs_path = os.path.join(AgentPath,i)
    Agents.append(Agent(abs_path,env.action_space,env.observation_space.shape[0],256))
    print(Agents[-1].m_name)

Data = Dataset(env,number_agent=len(Agents))
for i in range(10):
    Data.push(Agents)

model = Model(include_action = True, ob_dim = env.observation_space.shape[0],action_dim = env.action_space.shape[0])
model.train(Data,epoch = 100)## log : 出现loss大概率为0，可能是steo不一致导致的。