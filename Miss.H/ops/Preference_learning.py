import torch
from torch import nn
import numpy as np
import gym
from tqdm import tqdm
import os
from . import model as md

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Agent(object):
    def __init__(self, policy_path, sample_space,num_inp,embedding_dims):
        self.agent = md.GaussianPolicy(num_inp,sample_space.shape[0],embedding_dims,sample_space)
        self.agent.load_state_dict(torch.load(policy_path))
        self.sample_space = sample_space
        self.m_name = os.path.splitext(os.path.split(policy_path)[1])[0]


    def select_action(self,state):
        _,_,action = self.agent.sample(state)
        action = action.detach().numpy()
        return action

class RandomAgent(object):
    def __init__(self,action_space):
        self.action_space = action_space
        self.m_name = "Random Agent"

    def select_action(self,state):
        return self.action_space.sample()

"""
class GTDataset(object):
    def __init__(self,env):
        self.env = env
        self.trajs = []

    def gen_traj(self,agent, min_length):
        L_obs, L_action, L_reward = [self.env.reset()],[],[]
        while True:
            action = agent(obs[-1])
            state, reward, done,_ =self.env.step(action)
            L_obs.append(state)
            L_action.append(action)
            L_reward.append(reward)

            if done:
                if len(L_obs) > min_length + 1:
                    L_obs.pop()
                    break
                else:
                    L_obs.pop()
                    state = self.env.reset()
                    L_obs.append(state)

        return (np.stack(L_obs, axis = 0), np.concatenate(L_obs, axis = 0), np.array(L_reward))



    def prebuild(self,agents,min_length):
        assert len(agents) <= 0
        trajs = []
        for agent in tqdm(agents):
            traj = self.gen_traj(agent,min_length)
            tqdm.write("model : ", agent.m_name)
            trajs.append(traj)
        obs, actions, rewards = zip(*self.trajs)
        self.trajs = (np.concatenate(obs,axis = 0), np.concatenate(actions,axis = 0), np.concatenate(rewards, axis = 0))
        print(self.trajs[0].shape, self.trajs[1].shape, self.trajs[2].shape)

    def sample(self,num_sample,step = 40,include_action = False):
        obs,actions,rewards = self.trajs
        D = []

        for i in range(num_sample):
            x_ptr = np.random.choice(len(obs) - step)
            y_prt = np.random.choice(len(obs) - step)

            if include_action:
                D.append((np.concatenate([obs[x_ptr:x_ptr+step],actions[x_ptr:x_ptr+step]],axis=1),
                          np.concatenate([obs[y_prt:y_prt+step],actions[y_prt:y_prt+step]],axis=1),
                          0 if np.sum(rewards[x_ptr:x_ptr+step]) < np.sum(rewards[y_prt:y_prt+step]))
                        )

            else:
                D.append((obs[x_ptr:x_ptr+step],
                          obs[y_prt,y_prt+step],
                          0 if np.sum(rewards[x_ptr:x_ptr+step]) < np.sum(rewards[y_prt:y_prt+step]))
                        )

        return D
"""
class IRL_function(nn.Module):
    def __init__(self,inp_dim,num_layer = 2,embedding_dims=256):
        super(IRL_function,self).__init__()
        self.fc1 = nn.Linear(inp_dim,embedding_dims)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dims,embedding_dims)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(embedding_dims,1)
       
    def forward(self,x):
        if torch.cuda.is_available():
            f_x = torch.cuda.FloatTensor(x)
        else:
            f_x = torch.FloatTensor(x)
        f_x = self.fc1(f_x)
        f_x = self.relu1(f_x)
        f_x = self.fc2(f_x)
        f_x = self.relu2(f_x)
        r_x = self.fc3(f_x)

        return r_x


class Model(nn.Module):
    def __init__(self, include_action, ob_dim, action_dim, batch_size = 64,num_layer = 2, embedding_dims = 256, step = 40):
        super(Model,self).__init__()
        self.include_action = include_action
        self.inp_dim = ob_dim + action_dim if self.include_action else ob_dim


        self.step = step
        self.batch_size = batch_size

        self.reward_function = IRL_function(self.inp_dim,num_layer=num_layer,embedding_dims=embedding_dims)
        self.optim = torch.optim.Adam(self.reward_function.parameters(),lr = 1e-4)

    def train(self, dataset, epoch = 10,debug=True):
        for i in tqdm(range(epoch)):
            D = dataset.batch(batch_size = 5)
            loss = torch.FloatTensor()
            for d in D:
                r_x = torch.sum(self.reward_function(d[0]),axis = 0)
                r_y = torch.sum(self.reward_function(d[1]),axis = 0)
                logits = torch.cat([r_x,r_y],axis = 0).unsqueeze(axis=0)
                #lable = torch.FloatTensor([1.,0.] if d[-1] == 1 else [0.,1.])
                lable = torch.tensor([d[-1]],dtype=torch.int64)
                #T_loss = torch.nn.functional.cross_entropy(logits,lable)
                T_loss = torch.cat([loss,torch.nn.functional.cross_entropy(logits,lable).unsqueeze(0)],axis =0)

            T_label = 0
            if d[-1] == 1:
                if r_x > r_y:
                    T_label += 1
            else:
                if r_x < r_y:
                    T_label += 1
            loss = torch.mean((-1)*T_loss)
            print(f"epoch is : {i} , loss is {loss.item()}, logits is {logits}, lable is {lable}, T_loss :{T_loss.item()}")
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()







## ranked_mode : GT：选的step的reward  GTtraj:轨迹reward  No_step  Min_gap即x,y的reward相差要到达一定gap才能做对比
class Dataset(torch.utils.data.Dataset):
    def __init__(self,env,min_length = 1, step = None, capital = 5000,ranked_mode = 'GT', number_agent = 2,min_gap = None):
        super(Dataset,self).__init__()
        self.env = env
        self.memory = []
        self.capital = capital
        self.position = []
        self.ranked_mode = ranked_mode
        self.min_length = min_length
        self.step = step
        self.number_agent = number_agent
        for i in range(number_agent):
            self.memory.append([])
        self.position = np.zeros(number_agent,dtype=np.int64)
        self.min_gap = min_gap
        self.batch_size = 64

    def gen_traj(self,agent,min_length):
        if agent is None:
            agent = RandomAgent(self.env.action_space)
        done = False
        obs, actions,rewards = [self.env.reset()],[],[]
        step = 0
        while True:
            action = agent.select_action(obs[-1])
            state,reward,done,_ = self.env.step(action)
            obs.append(state)
            rewards.append(reward)
            actions.append(action)
            step += 1

            if done is True:
                if step >= min_length:
                    obs.pop()
                    break
                
                else:
                    obs.pop()
                    obs.append(self.env.reset())

        np_obs = np.stack(obs,axis = 0)
        np_action = np.stack(actions,axis=0)
        np_rewards = np.array(rewards)
        return (np_obs,np_action,np_rewards)

            

    def push(self,agents:list,random_agent = False):
        assert len(agents) > 0
        if random_agent:
            agents.append(None)

        if self.ranked_mode is "GT_No_Step":
            min_length = 1
        else:
            min_length = self.min_length
        
        trajs = []
        step = 0
        for agent in tqdm(agents):
            traj_len = 0
            while traj_len < min_length:
                temp = self.gen_traj(agent,min_length)
                traj_len += temp[0].shape[0]
                if self.position[step] < self.capital:
                    self.memory[step].append(temp)
                    self.position[step] += 1
                else:
                    index = np.random.choice(self.position,1)
                    self.memory[step].pop(index)
                    self.memory[step].append(temp)
            step += 1

    def sample(self):
        agent_index = np.random.choice(self.number_agent,2)
        data_index = np.random.choice(self.capital,2)
        data_index %= self.position[agent_index]
        ### data_index 
        if self.min_gap is not None:
            while np.abs(np.sum(self.memory[agent_index[0]][data_index[0]][-1]) - np.sum(self.memory[agent_index[1]][data_index[1]][-1])) < self.min_gap:
                agent_index = np.random.choice(self.number_agent,2)
                data_index = np.random.choice(self.capital,2)
                data_index %= self.position

        x = self.memory[agent_index[0]][data_index[0]]
        y = self.memory[agent_index[1]][data_index[1]] 
            

        if self.step is None or self.ranked_mode is "GT_No_step":
            r_x = np.sum(x[-1])
            r_y = np.sum(y[-1])
            data = (np.concatenate([x[0],x[1]],axis = 1),
                    np.concatenate([y[0],y[1]],axis = 1),
                    0 if r_x > r_y else 1)
        else: 
            time_x = np.random(x[0].shape[0]-step,1)
            time_y = np.random(y[0].shape[0] -step,1)
            if ranked_mode is "GTTraj":
                r_x = np.sum(x[-1][time_x:time_x+step])
                r_y = np.sum(y[-1][time_y:time_y+step])
            else:
                r_x = np.sum(x[-1])
                r_y = np.sum(y[-1])
            data = (np.concatenate([x[0],x[1]],axis = 1)[time_x:time_x+step],
                    np.concatenate([y[0],y[1]],axis = 1)[time_y:time_y+step],
                    0 if r_x > r_y else 1
            )
        return agent_index,data_index,data

    def batch(self,batch_size:int):
        batch_data = []
        for i in range(batch_size):
            _,_,data = self.sample()
            batch_data.append(data)
        return batch_data

    def __len__(self):
        return len(self.memory)
if __name__ == "__main__":
    env = gym.make("Humanoid-v2")
    dataset = Dataset(env)
    agent = RandomAgent(env.action_space)
    agent1 = RandomAgent(env.action_space)
    dataset.push([agent,agent1])
    dataset.__getitem__(1)
    model = Model(include_action = True,batch_size = 1,ob_dim = env.observation_space.shape[0],action_dim = env.action_space.shape[0])
    model.train(dataset)
