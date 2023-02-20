import math
import torch
from torch import nn
import torch.nn.functional as F

class DNM_Linear_M3(nn.Module):
    def __init__(self, input_size, out_size, M=5, device='cpu'):
        super(DNM_Linear_M3, self).__init__()

        Synapse_W = torch.rand([out_size, M, input_size]).to(device)#.cuda() # [size_out, M, size_in]
        Synapse_q = torch.rand([out_size, M, input_size]).to(device)#.cuda()
        Dendritic_W2 = torch.rand([input_size]).to(device)#.cuda() # size_out, M, size_in]
        torch.nn.init.constant_(Synapse_q, 0.1)
        k = torch.rand(1).to(device)
        qs = torch.rand(1).to(device)

        self.params = nn.ParameterDict({'Synapse_W': nn.Parameter(Synapse_W)})
        self.params.update({'Synapse_q': nn.Parameter(Synapse_q)})
        self.params.update({'Dendritic_W2': nn.Parameter(Dendritic_W2)})
        self.params.update({'k': nn.Parameter(k)})
        self.params.update({'qs': nn.Parameter(qs)})
        self.input_size = input_size

        self.activate_layer = nn.ReLU() #nn.LeakyReLU(0.1)

    def forward(self, x):
        # Synapse
        out_size, M, _ = self.params['Synapse_W'].shape
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 2)
        x = x.repeat(1, out_size, M, 1)
        x = torch.mul(x, self.params['Synapse_W']) - self.params['Synapse_q']
        x = self.activate_layer(x)

        x = torch.mul(x, torch.tanh(self.params['Dendritic_W2']))

        # Dendritic
        x = torch.sum(x, 3) #prod 
        x = self.activate_layer(x)

        # Membrane
        x = torch.sum(x, 2)

        # Soma
        x = self.params['k'] * (x - self.params['qs'])

        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)



class DNM_multiple(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, M=5):
        super(DNM_multiple, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.DNM_Linear1 = DNM_Linear_M3(input_size, hidden_size, M)
        self.DNM_Linear2 = DNM_Linear_M3(hidden_size, out_size, M)
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.DNM_Linear1(x)
        out = self.DNM_Linear2(x)
        return out
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class DNM_multiple_3(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, M=5):
        super(DNM_multiple_3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.DNM_Linear_i = DNM_Linear_M3(input_size, hidden_size, M)
        self.DNM_Linear_h = DNM_Linear_M3(hidden_size, hidden_size, M)
        self.DNM_Linear_o = DNM_Linear_M3(hidden_size, out_size, M)
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.DNM_Linear_i(x)
        x = self.DNM_Linear_h(x)
        out = self.DNM_Linear_o(x)
        return out
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = x.float()
        x = self.l1(x)
        x = torch.relu(self.l2(x))
        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

# EA
#import math
import numpy as np

# 定义DNM
class DNM:
    def __init__(self, w, q, M=5, qs=0.1, k=0.5):
        self.M = M
        self.qs = qs
        self.k = k
        self.w = w
        self.q = q 
    
    def run(self, data):
        data = data.T
        _, j = data.shape
        result = np.zeros([j, 1]) # [297. 1]
        for h in range(j):
            train_in2 = np.tile(data[:,h], (self.M, 1)).T
            y = 1.0/(1+np.exp(-self.k*(self.w*train_in2-self.q)))
            z = np.prod(y, 1)
            v = sum(z)
            result[h] = 1.0/(1+np.exp(-self.k*(v-self.qs)))
        return result


