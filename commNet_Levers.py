"""*CommNet on Levers:
   *
   * IMPLEMENTATION DETAIL:  Parameters are shared accross each f unit accross agents (same for all agents).
   *                         Parameters are not shared accross communication steps. Therefore each 
   *                         communication step has its own instance of Streamlayer (the shared f unit).
   *                         Each f unit (Stream layer) has structure layer > relu > layer > relu > layer
   *                         according to release code.
   *     
   *                  """
import numpy as np 
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random
import itertools
import math

class Action_Decoder(nn.Module):
    """Takes as input the hidden state (after CommNet) and outputs a distribution over actions"""
    def __init__(self, in_size, nr_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_size, nr_actions)
    def forward(self, x):
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x

class Lut_Encoder(nn.Module):
    """Lookup table encoder: Maps the input state (the agent ID which is a number
    between 1 and 500) to a 128d vector. This is vector is fixed."""
    def __init__(self, lut_size = 500, lut_dim = 128):
        super().__init__()
        self.lookup_table = nn.Embedding(lut_size, lut_dim)

    def look(self, index):
        """index: a tensor with the indexes (agent numbers)
        returns: a tensor with an extra dimension of size lut_dim 
        in place of the indexes """
        return self.lookup_table(index)

class Stream_Layer(nn.Module):
    """The f unit from the paper """
    def __init__(self,n_levers, batch_size,  hidden_size = 64, out_size =64, activation = 'relu'):
        super(Stream_Layer, self).__init__()
        self.batch_size = batch_size
        self.n_levers = n_levers
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.in_size = 3*self.out_size
        
        self.fc1 = nn.Linear(self.in_size, hidden_size*3)
        self.fc2 = nn.Linear(self.hidden_size*3, self.hidden_size*3)
        self.fc3 = nn.Linear(self.hidden_size*3, self.hidden_size)

    def forward(self, h_i, c_i, h_0 = None):
        """h_c_in: The concatenated input vector of (h_0, c_i, h_i). 
        Returns: The resulting hidden state from the forward pass of each (h_0, c_i, h_i) input"""
        h_c_in = torch.cat((h_i,c_i, h_0), dim = 1)

        h_i_next = self.fc1(h_c_in)
        h_i_next = F.relu(h_i_next)
        h_i_next = self.fc2(h_i_next)
        h_i_next = F.relu(h_i_next)
        h_i_next = self.fc3(h_i_next)

        return h_i_next
    def comm(self, h_out):
        """Takes as input the hidden state outputs of each agent and retuns the average
        of hidden states excluding the agent's own hidden state.
        
        h_out: A tensor array with h vectors along dimension 0 """
        true_mask = np.ones((self.n_levers,), dtype=bool)
        sum_masks = []
        h_out = h_out.reshape(self.batch_size,self.n_levers,-1)

        for cntr in range(self.n_levers):
            hldr = np.array(true_mask)
            hldr[cntr] = False
            sum_masks.append(hldr)

        #Sums the communication output from all other agents)
        sum_masks = torch.from_numpy(np.array(sum_masks))
        c_i_next_list = []
        for batch in range(self.batch_size):
            temp = h_out[batch][sum_masks[0]]
            c_i_next_temp = [torch.sum(h_out[batch][sum_masks[i]], dim = 0) for i in range(self.n_levers)]
            c_i_next_list.append(torch.stack(c_i_next_temp))

        c_i_next = torch.cat(c_i_next_list, dim=0)
        num_other_agents =self.n_levers - 1
        return torch.div(c_i_next,num_other_agents)

def levers(opt):

    def calculate_reward(results, num_range ):
        """Calculates the number of distinct numbers from 0 to 4. 
           Gives the same result as reward function in realeased code.

       results: A numpy array or tensor array of resulting number choices"""
        numlist = np.arange(num_range)
        results = results.reshape(opt['batch_size'], opt['nlevers']) 
        r = torch.zeros(opt['batch_size'])
        for batch in range(opt['batch_size']):
            num_arr = np.sort(results[batch].numpy())
            a = num_arr[0]
            k = num_arr[1:]
            for b in num_arr[1:]:
                if a !=b:
                    r[batch] +=1
                    a = b
        r = r.repeat(5,1)
        r= torch.transpose(r,0,1)
        r = r.reshape(-1)* torch.tensor(1/(num_range-1))
        return r

    #CommNet structure:
    f1 = Stream_Layer(opt['nlevers'],opt['batch_size'], opt['hdim'], opt['hdim'])
    f2 = Stream_Layer(opt['nlevers'],opt['batch_size'], opt['hdim'], opt['hdim'])
    f3 = Stream_Layer(opt['nlevers'],opt['batch_size'], opt['hdim'], opt['hdim'])
    lt = Lut_Encoder(lut_size = opt['nagents'], lut_dim=opt['hdim'])
    ad = Action_Decoder(f3.out_size, opt['nlevers'])

    parameters = list(f1.parameters()) +list(f2.parameters()) + \
        list(f3.parameters()) + list(ad.parameters()) 

    optimizer = optim.SGD(parameters, lr=0.001)

    avreward = 0
    totreward = 0

    baseline = torch.tensor(0)
    for ep in range(opt['maxiter']):
        optimizer.zero_grad()
        #Get a tensor of agent ids of size (opt['nlevers']*opt['batch_size'], )
        ids_list = [np.random.choice(np.arange(opt['nagents']), size = opt['nlevers'], replace=False) \
        for _ in range(opt['batch_size'])]
        agent_ids = np.hstack(ids_list)
        agent_ids = torch.from_numpy(agent_ids)

        #Comm step 1:
        h0 = lt.look(agent_ids) #Size (opt['nlevers']*opt['batch_size'], 1)
        c0 = torch.zeros(opt['nlevers'] * opt['batch_size'], f1.out_size)
        h1 = f1.forward(h0,c0,h0)
        c1 = f1.comm(h1)

        #Comm step 2:
        h2 = f2.forward(h1,c1,h0)
        c2 = f2.comm(h2)
        h3 = f3.forward(h2,c2,h0)

        action_probabilities = ad(h3)
        action_indexes = torch.multinomial(action_probabilities , 1)

        reward = calculate_reward(action_indexes, opt['nlevers'])
        totreward += torch.mean(reward)
        if ep%100 == 0:
            avreward = totreward/100
            totreward = 0

        #PG with baseline (according to released code):
        baseline = 0.99*baseline + 0.01 * torch.mean(reward) 
        m = torch.distributions.Categorical(action_probabilities)
        new_loss = -m.log_prob(action_indexes.squeeze()) * (reward- baseline)
        new_loss = new_loss.sum()
        
        new_loss.backward()
        
        step = (ep - 1 + opt['anneal'])/opt['anneal']
        dt = max(opt['lr']*(1/step), .00001*opt['lr']) #dt is new lr
        for g in optimizer.param_groups: g['lr'] = dt

        if opt['clip'] != 0:
            torch.nn.utils.clip_grad_norm_(parameters,opt['clip'])
                
        optimizer.step()

        if ep % 100 == 0:
            print("Episode: {}/{}  Loss: {} dt: {}  Average reward: {}".format(ep,opt['maxiter'], new_loss.detach().item(), dt, avreward))
            
if __name__ == "__main__":
    opt = {
        'nlevers' : 5, 
        'nagents' : 500,
        'maxiter' : 150000,
        'batch_size' : 64,
        'lr' : 10,
        'anneal' : 1e6,
        'clip': 0.01,
        'hdim' : 64
        #nlayer : 2,
        #recurrent : False,
        #comm : True
    }

    levers(opt)
        
        
            










    

               
        




    

    














