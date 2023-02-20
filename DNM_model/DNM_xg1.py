import math
import torch
from torch import nn
import torch.nn.functional as F

class Soma(nn.Module):
    def __init__(self, k, qs):
        super(Soma, self).__init__()
        self.params = nn.ParameterDict({'k': nn.Parameter(k)})
        self.params.update({'qs': nn.Parameter(qs)})

    def forward(self, x):
        y = 1 / (1 + torch.exp(-self.params['k'] * (x - self.params['qs'])))
        return y

class Soma_old(nn.Module):
    def __init__(self, k, qs):
        super(Soma_old, self).__init__()
        self.k = k
        self.qs = qs

    def forward(self, x):
        y = 1 / (1 + torch.exp(-self.k * (x - self.qs)))
        return y

class Membrane(nn.Module):
    def __init__(self):
        super(Membrane, self).__init__()

    def forward(self, x):
        x = torch.sum(x, 1)
        return x

class Membrane_w(nn.Module):
    def __init__(self, w):
        super(Membrane_w, self).__init__()
        self.params = nn.ParameterDict({'w': nn.Parameter(w)})

    def forward(self, x):
        x = torch.mul(x, self.params['w'])
        x = torch.sum(x, 1)
        return x

class Dendritic(nn.Module):
    def __init__(self):
        super(Dendritic, self).__init__()

    def forward(self, x):
        x = torch.prod(x, 2)
        return x

class Dendritic2(nn.Module):
    def __init__(self):
        super(Dendritic2, self).__init__()

    def forward(self, x):
        x = torch.sum(x, 3)
        return x

class Dendritic4(nn.Module):
    def __init__(self, b):
        super(Dendritic4, self).__init__()
        self.params = nn.ParameterDict({'b': nn.Parameter(b)})

    def forward(self, x):
        x = x+self.params["b"]
        x = torch.prod(x, 2)
        return x

class Synapse(nn.Module):

    def __init__(self, w, q, k):
        super(Synapse, self).__init__()
        self.params = nn.ParameterDict({'w': nn.Parameter(w)})
        self.params.update({'q': nn.Parameter(q)})
        self.params.update({'k': nn.Parameter(k)})

    def forward(self, x):
        num, _ = self.params['w'].shape
        x = torch.unsqueeze(x, 1)
        x = x.repeat((1, num, 1))
        y = 1 / (1 + torch.exp(
            torch.mul(-self.params['k'], (torch.mul(x, self.params['w']) - self.params['q']))))

        return y

class Synapse_old(nn.Module):

    def __init__(self, w, q, k):
        super(Synapse_old, self).__init__()
        self.params = nn.ParameterDict({'w': nn.Parameter(w)})
        self.params.update({'q': nn.Parameter(q)})
        self.k = k

    def forward(self, x):
        num, _ = self.params['w'].shape
        x = torch.unsqueeze(x, 1)
        x = x.repeat((1, num, 1))
        y = 1 / (1 + torch.exp(
            torch.mul(-self.k, (torch.mul(x, self.params['w']) - self.params['q']))))

        return y

class Synapse2(nn.Module):

    def __init__(self, w, q, k):
        super(Synapse2, self).__init__()
        self.params = nn.ParameterDict({'w': nn.Parameter(w)})
        self.params.update({'q': nn.Parameter(q)})
        self.k = k

    def forward(self, x):
        size_out, num, _ = self.params['w'].shape
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 1)
        x = x.repeat((1, size_out, num, 1))
        y = 1 / (1 + torch.exp(
            torch.multiply(-self.k, (torch.multiply(x, self.params['w']) - self.params['q']))))

        return y

class DNM(nn.Module):
    def __init__(self, w, q, k, qs):
        super(DNM, self).__init__()
        self.model = nn.Sequential(
            Synapse_old(w, q, k),
            Dendritic(),
            Membrane(),
            Soma_old(k, qs)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class BASE_DNM(nn.Module):
    def __init__(self, dim, M, kv=5, qv=0.3):#, device=torch.device('cuda:0')):
        
        w = torch.rand([M, dim])#.to(device)
        q = torch.rand([M, dim])#.to(device)
        #k = torch.tensor(kv)
        #qs = torch.tensor(qv)
        k = torch.rand(1)
        qs = torch.rand(1)

        super(BASE_DNM, self).__init__()
        self.model = nn.Sequential(
            Synapse(w, q, k),
            Dendritic(),
            Membrane(),
            Soma(k, qs)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class BASE_DNM3(nn.Module):
    def __init__(self, dim, M, kv=5, qv=0.3):#, device=torch.device('cuda:0')):
        
        w = torch.rand([M, dim])#.to(device)
        q = torch.rand([M, dim])#.to(device)
        #k = torch.tensor(kv)
        #qs = torch.tensor(qv)
        k = torch.rand(1)
        k_soma = torch.rand(1)
        qs = torch.rand(1)

        super(BASE_DNM3, self).__init__()
        self.model = nn.Sequential(
            Synapse(w, q, k),
            Dendritic(),
            Membrane(),
            Soma(k_soma, qs)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class BASE_DNM4(nn.Module):
    def __init__(self, dim, M, kv=5, qv=0.3):#, device=torch.device('cuda:0')):
        
        w = torch.rand([M, dim])#.to(device)
        q = torch.rand([M, dim])#.to(device)
        b = torch.rand([M, dim])
        #k = torch.tensor(kv)
        #qs = torch.tensor(qv)
        k = torch.rand(1)
        k_soma = torch.rand(1)
        qs = torch.rand(1)

        super(BASE_DNM4, self).__init__()
        self.model = nn.Sequential(
            Synapse(w, q, k),
            Dendritic4(b),
            Membrane(),
            Soma(k_soma, qs)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class BASE_DNM2(nn.Module):
    def __init__(self, dim, M, kv=5, qv=0.3):#, device=torch.device('cuda:0')):
        
        w = torch.rand([M, dim])#.to(device)
        q = torch.rand([M, dim])#.to(device)
        #k = torch.tensor(kv)#.to(device)
        #qs = torch.tensor(qv)#.to(device)
        k = torch.rand(1)
        qs = torch.rand(1)

        w_d = torch.rand([M, dim])
        w_m = torch.rand([M])

        super(BASE_DNM2, self).__init__()
        self.model = nn.Sequential(
            Synapse(w, q, k),
            Dendritic(),
            Membrane_w(w_m),
            Soma(k, qs)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class M_DNM(nn.Module):
    def __init__(self, input_size, M, k=0.5, qs=0.1):#, device=torch.device('cuda:0')):
        super(M_DNM, self).__init__()

        self.dnm1 = BASE_DNM3(input_size, M, k, qs)#, device)
        self.dnm2 = BASE_DNM3(input_size+1, M, k, qs)#, device)
        self.dnm3 = BASE_DNM3(input_size+2, M, k, qs)#, device)

        #q = torch.rand([out_size, M, input_size])#.cuda()
        #torch.nn.init.constant_(q, 0.1)
        #k = torch.tensor(k).cuda()
        #qs = torch.tensor(qs).cuda()


    def forward(self, x):

        o1 = self.dnm1(x)
        x = torch.cat((x,torch.unsqueeze(o1, 1)),1)
        o2 = self.dnm2(x)
        x = torch.cat((x,torch.unsqueeze(o2, 1)),1)
        o3 = self.dnm3(x)
        #x = torch.cat((x,o3),1)

        return o3

class M_DNM2(nn.Module):
    def __init__(self, input_size, M, k=0.5, qs=0.1):#, device=torch.device('cuda:0')):
        super(M_DNM2, self).__init__()

        self.dnm1 = BASE_DNM(input_size//2, M, k, qs)#, device)
        self.dnm2 = BASE_DNM(input_size//2, M, k, qs)#, device)
        self.dnm3 = BASE_DNM(2, M, k, qs)#, device)

    def forward(self, x):
        x1, x2 = x.chunk(2,dim=1)
        o1 = self.dnm1(x1)
        o2 = self.dnm2(x2)
        o = torch.cat((torch.unsqueeze(o1, 1),torch.unsqueeze(o2, 1)),1)
        o3 = self.dnm3(o)
        #x = torch.cat((x,o3),1)

        return o3

class DNM_Linear(nn.Module):
    def __init__(self, input_size, out_size, M, k=0.5, qs=0.1):
        super(DNM_Linear, self).__init__()

        DNM_W = torch.rand([out_size, M, input_size])#.cuda() # [size_out, M, size_in]
        dendritic_W = torch.rand([input_size])#.cuda() # size_out, M, size_in]
        membrane_W = torch.rand([M])#.cuda() # size_out, M, size_in]
        q = torch.rand([out_size, M, input_size])#.cuda()
        torch.nn.init.constant_(q, 0.1)
        #k = torch.tensor(k).cuda()
        #qs = torch.tensor(qs).cuda()

        self.params = nn.ParameterDict({'DNM_W': nn.Parameter(DNM_W)})
        self.params.update({'q': nn.Parameter(q)})
        self.params.update({'dendritic_W': nn.Parameter(dendritic_W)})
        self.params.update({'membrane_W': nn.Parameter(membrane_W)})
        self.k = k
        self.qs = qs

    def forward(self, x):
        # Synapse
        out_size, M, _ = self.params['DNM_W'].shape
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 2)
        x = x.repeat(1, out_size, M, 1)
        #x = torch.sigmoid(torch.mul(-self.k, (torch.mul(x, self.params['DNM_W']) - self.params['q'])))
        x = torch.mul(x, self.params['DNM_W'])
        x = F.relu(0.5 * (x - self.params['q']))

        # Dendritic
        #x = torch.mul(x, self.params['dendritic_W'])
        #x = x * self.params['dendritic_W']
        x = torch.sum(x, 3)
        #x = torch.sigmoid(x)
        x = F.relu(x)

        # Membrane
        #x = torch.mul(x, self.params['membrane_W'])
        x = x * self.params['membrane_W']
        x = torch.sum(x, 2)

        # Soma
        x = F.relu(self.k * (x - self.qs))

        return x


class DNM_Linear_M(nn.Module):
    def __init__(self, input_size, out_size, M, k=0.5, qs=0.1):
        super(DNM_Linear_M, self).__init__()

        Synapse_W = torch.rand([out_size, M, input_size])#.cuda() # [size_out, M, size_in]
        Synapse_q = torch.rand([out_size, M, input_size])#.cuda()
        Dendritic_W = torch.rand([M])#.cuda() # size_out, M, size_in]
        Dendritic_q = torch.ones([M])#.cuda() # size_out, M, size_in]
        membrane_W = torch.rand([M])#.cuda() # size_out, M, size_in]
        
        torch.nn.init.constant_(Synapse_q, 0.1)
        #k = torch.tensor(k).cuda()
        #qs = torch.tensor(qs).cuda()

        self.params = nn.ParameterDict({'Synapse_W': nn.Parameter(Synapse_W)})
        self.params.update({'Synapse_q': nn.Parameter(Synapse_q)})
        self.params.update({'Dendritic_W': nn.Parameter(Dendritic_W)})
        self.params.update({'Dendritic_q': nn.Parameter(Dendritic_q)})
        self.params.update({'membrane_W': nn.Parameter(membrane_W)})
        self.k = k
        self.qs = qs
        self.input_size = input_size

    def forward(self, x):
        # Synapse
        out_size, M, _ = self.params['Synapse_W'].shape
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 2)
        x = x.repeat(1, out_size, M, 1)
        x = torch.sigmoid(torch.mul(x, self.params['Synapse_W']) + self.params['Synapse_q'])

        # Dendritic
        x = torch.prod(x, 3)
        x = x * self.params['Dendritic_W'] - self.params['Dendritic_q']*self.input_size
        x = torch.sigmoid(x)

        # Membrane
        #x = x * self.params['membrane_W']
        x = torch.sum(x, 2)

        # Soma
        #x = torch.sigmoid(self.k * (x - self.qs))

        return x

class DNM_Linear_M2(nn.Module):
    def __init__(self, input_size, out_size, M, k=0.5, qs=0.1):
        super(DNM_Linear_M2, self).__init__()

        Synapse_W = torch.rand([out_size, M, input_size])#.cuda() # [size_out, M, size_in]
        Synapse_q = torch.rand([out_size, M, input_size])#.cuda()
        Dendritic_W = torch.rand([M])#.cuda() # size_out, M, size_in]
        Dendritic_q = torch.ones([M])#.cuda() # size_out, M, size_in]
        membrane_W = torch.rand([M])#.cuda() # size_out, M, size_in]
        
        torch.nn.init.constant_(Synapse_q, 0.1)
        #k = torch.tensor(k).cuda()
        #qs = torch.tensor(qs).cuda()

        self.params = nn.ParameterDict({'Synapse_W': nn.Parameter(Synapse_W)})
        self.params.update({'Synapse_q': nn.Parameter(Synapse_q)})
        self.params.update({'Dendritic_W': nn.Parameter(Dendritic_W)})
        self.params.update({'Dendritic_q': nn.Parameter(Dendritic_q)})
        self.params.update({'membrane_W': nn.Parameter(membrane_W)})
        self.k = k
        self.qs = qs
        self.input_size = input_size

    def forward(self, x):
        # Synapse
        out_size, M, _ = self.params['Synapse_W'].shape
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 2)
        x = x.repeat(1, out_size, M, 1)
        x = torch.sigmoid(torch.mul(x, self.params['Synapse_W']) + self.params['Synapse_q'])

        # Dendritic
        x = torch.sum(x, 3)
        x = x * self.params['Dendritic_W'] - self.params['Dendritic_q']*self.input_size
        x = torch.sigmoid(x)

        # Membrane
        #x = x * self.params['membrane_W']
        x = torch.sum(x, 2)

        # Soma
        #x = torch.sigmoid(self.k * (x - self.qs))

        return x


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

    def forward(self, x):
        # Synapse
        out_size, M, _ = self.params['Synapse_W'].shape
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 2)
        x = x.repeat(1, out_size, M, 1)
        x = 0.5 * (torch.mul(x, self.params['Synapse_W']) - self.params['Synapse_q'])
        x = torch.sigmoid(x)

        x = torch.mul(x, self.params['Dendritic_W2'])

        # Dendritic
        x = torch.sum(x, 3) #prod 
        x = torch.sigmoid(x)

        # Membrane
        #x = x * self.params['membrane_W']
        x = torch.sum(x, 2)

        # Soma
        # x = torch.sigmoid(self.params['k'] * (x - self.params['qs']))
        x = self.params['k'] * (x - self.params['qs'])

        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class DNM_Linear_E(nn.Module):
    def __init__(self, input_size, out_size, M, k=0.5, qs=0.1):
        super(DNM_Linear_E, self).__init__()

        Synapse_W = torch.rand([out_size, M, input_size])#.cuda() # [size_out, M, size_in]
        Synapse_q = torch.rand([out_size, M, input_size])#.cuda()
        Dendritic_W = torch.rand([M])#.cuda() # size_out, M, size_in]
        Dendritic_q = torch.ones([M])#.cuda() # size_out, M, size_in]
        membrane_W = torch.rand([M])#.cuda() # size_out, M, size_in]
        
        torch.nn.init.constant_(Synapse_q, 0.1)
        #k = torch.tensor(k).cuda()
        #qs = torch.tensor(qs).cuda()

        self.params = nn.ParameterDict({'Synapse_W': nn.Parameter(Synapse_W)})
        self.params.update({'Synapse_q': nn.Parameter(Synapse_q)})
        self.params.update({'Dendritic_W': nn.Parameter(Dendritic_W)})
        self.params.update({'Dendritic_q': nn.Parameter(Dendritic_q)})
        self.params.update({'membrane_W': nn.Parameter(membrane_W)})
        self.k = k
        self.qs = qs
        self.input_size = input_size

    def forward(self, x):
        # Synapse
        out_size, M, _ = self.params['Synapse_W'].shape
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 2)
        x = x.repeat(1, out_size, M, 1)
        x = torch.sigmoid(torch.mul(x, self.params['Synapse_W']) + self.params['Synapse_q'])

        # Dendritic
        x = torch.prod(x, 3)
        x = x * self.params['Dendritic_W'] + self.params['Dendritic_q']*self.input_size
        x = torch.sigmoid(x)

        # Membrane
        #x = x * self.params['membrane_W']
        x = torch.sum(x, 2)

        # Soma
        #x = torch.sigmoid(self.k * (x - self.qs))

        return x

class DNM_multiple_old(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, M, k=0.5, qs=0.1):
        super(DNM_multiple_old, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.DNM_Linear1 = DNM_Linear_M3(input_size, 32, M, k, qs)
        self.DNM_Linear2 = DNM_Linear_M3(32, out_size, M, k, qs)
        self.DNM_Linear3 = DNM_Linear_M3(4, out_size, M, k, qs)
        self.linear_1 = torch.nn.Linear(32, 4)
        self.linear_2 = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.DNM_Linear1(x)
        # x = self.linear_1(x)
        #x = self.DNM_Linear2(x)
        x = self.DNM_Linear2(x)
        #x = self.linear_2(x)
        out = x#F.softmax(x, dim=1)
        return out
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class DNM_multiple(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, M, k=0.5, qs=0.1):
        super(DNM_multiple, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.DNM_Linear1 = DNM_Linear_M3(input_size, 32, M, k, qs)
        self.DNM_Linear2 = DNM_Linear_M3(32, out_size, M, k, qs)
        self.DNM_Linear3 = DNM_Linear_M3(4, out_size, M, k, qs)
        self.linear_1 = torch.nn.Linear(32, 4)
        self.linear_2 = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.DNM_Linear1(x)
        # x = self.linear_1(x)
        #x = self.DNM_Linear2(x)
        x = self.DNM_Linear2(x)
        #x = self.linear_2(x)
        out = x#F.softmax(x, dim=1)
        return out
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class DNM_CNN(nn.Module):
    def __init__(self, input_size, out_size, M, k=0.5, qs=0.1):
        super(DNM_CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,10,kernel_size=3,stride=1,padding=2),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=3),
                                         torch.nn.Conv2d(10,20,kernel_size=3,stride=1,padding=2),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=3))
        
        self.DNM_Linear = DNM_Linear(7*7*20, out_size, M, k, qs)
    
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 7*7*20)
        x = self.DNM_Linear(x)
        out = F.softmax(x, dim=1)

        return out

class CNN(nn.Module):
    def __init__(self, out_size):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,10,kernel_size=3,stride=1,padding=2),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=3),
                                         torch.nn.Conv2d(10,20,kernel_size=3,stride=1,padding=2),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=3))
        
        self.linear = nn.Linear(7*7*20, out_size)
    
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 7*7*20)
        x = self.linear(x)
        out = F.softmax(x, dim=1)

        return out



class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 32)
        self.l2 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = x.float()
        x = self.l1(x)
        x = torch.relu(self.l2(x))
        return x