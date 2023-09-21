import torch 
import torch.nn as nn


class MLPEncoderDecoder(nn.Module):
    def __init__(self, in_dim, state_dim, hidden, activ = nn.ReLU(), **kwargs):
        super().__init__()
        self.act = activ
        self.net = nn.ModuleList([nn.Linear(in_dim, hidden[0])])
        self.net.extend([nn.Linear(hidden[i-1], hidden[i]) for i in range(1,len(hidden))])
        self.tail = nn.Linear(hidden[-1], state_dim)
    
    def encode(self, x):
        for l in self.net:
            x = self.act(l(x))

        return self.tail(x)


class DynGRU(nn.Module):
    # check bias ?
    def __init__(self, input_dim, mem_sz, output_dim, BRC = False, **kwargs):
        super().__init__()
        self.sensor = MLPEncoderDecoder(input_dim, 3*mem_sz, **kwargs)
        self.actuator = MLPEncoderDecoder(mem_sz, output_dim, **kwargs)
        self.memsz = mem_sz
        self.BRC = BRC
        if BRC:
            self.mem2mem = nn.Linear(mem_sz, 2*mem_sz)
        else:
            self.mem2mem = nn.Linear(mem_sz, 3*mem_sz)


    def step(self, ht, xt):
        h_proj = self.mem2mem(ht)
        x_a, x_c, x_h = torch.split(xt, self.memsz, dim = 1)
        if not self.BRC:
            h_a, h_c, h_h = torch.split(h_proj, self.memsz, dim = 1)
        else:
            h_a, h_c, h_h = torch.split(h_proj, self.memsz, dim = 1) + (ht,)

        if self.BRC:
            a = torch.tanh(x_a + h_a) + 1
        else:
            a = torch.sigmoid(x_a + h_a)

        c = torch.sigmoid(x_c + h_c)
        h_next = (1 - c)*ht + c*torch.tanh(x_h + a*h_h)

        return h_next
    

    def forward(self, x_seq):
        x_encoded = self.sensor(x_seq)
        B, L, M = x_encoded.shape
        h_seq = torch.zeros((B,L + 1,M))
        
        for t in range(L):
            h_seq[:,t+1,:] = self.step(h_seq[:,t,:], x_encoded[:,t,:])

        out_decode = self.actuator(h_seq[:,1:,:])

        return out_decode



if __name__ == '__main__':
    enc = DynGRU(1, 64, 3, BRC = False, hidden = (32,32,32))
    print(enc)