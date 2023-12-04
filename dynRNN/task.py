from abc import ABC, abstractmethod
import torch 
import matplotlib.pyplot as plt

class Task(ABC):
    @abstractmethod
    def get_batch(**args):
        pass

    @abstractmethod
    def loss(pred, target):
        pass 


class CopyFirstInput(Task):
    def get_batch(n, max_time):
        return torch.randn((n,max_time,1))
    
    def loss(target, pred):
        return ((target - pred)**2).mean()
         
    def show_pred(pred_seq, target_seq):
        f, ax = plt.subplots()
        ax.plot(target_seq, label='input')
        ax.plot(pred_seq, label='pred')
        ax.plot(target_seq[0]*torch.ones_like(target_seq),'r--', label='target')
        ax.legend()
        
        return ax
    
class CopyFirstInputTh(Task):        
    def get_batch(n, max_time):
        return torch.randn((n,max_time,1))
    
    def loss(seq, pred, th=2.):
        targs = []
        for b in seq: #B x L
            can = b[b.abs() > th]
            targ = 0.
            if can.numel() > 0:
                targ = can[0]
            targs.append(targ)
            
        targs = torch.tensor(targs).to(seq)
        return ((targs.unsqueeze(-1) - pred)**2).mean()
         
    def show_pred(pred_seq, target_seq, th = 2.):
        f, ax = plt.subplots()
        ax.plot(target_seq, label='input')
        ax.plot(pred_seq, label='pred')
        targ = target_seq[target_seq.abs() > th]
        if targ.numel() == 0:
            targ = 0.
        else:
            targ = targ[0]
        ax.plot(targ*torch.ones_like(target_seq),'r--', label='target')
        ax.legend()
        
        return ax
    
    
#signal last 2.7 s -> 27 steps * 2 = 54
#delay last 3 s -> 30 steps
#decision delay last 1s -> 10 steps
#signal is 94 so in/out delay of 56
#a task is 150 steps

class FreqDiscr(Task):
    def get_batch(n):
        data = torch.zeros((n,150))
        label = torch.zeros((n,150))
        
        for i in range(n):
            in_delay = torch.randint(1,46,(1,))
            
            s1 = torch.randint(2,(1,))
            data[i,in_delay:in_delay+27] = FreqDiscr.grouped() if s1 else FreqDiscr.evenly()
            
            s2 = torch.randint(2,(1,))
            data[i,in_delay+47:in_delay+27+47] = FreqDiscr.grouped() if s2 else FreqDiscr.evenly()
            
            label[i,in_delay+27+47+10:] = s1==s2
        
        return data.unsqueeze(-1),label
        
    def loss(target, pred):
        print(target.shape,pred.shape)
        return ((target - pred)**2).mean()
    
    def show_pred(pred_seq, target_seq, input_seq, ax = None):
        if ax is None:
            f, ax = plt.subplots()
        
        ax.plot(input_seq, label='input')
        ax.plot(pred_seq, label='pred')
        ax.plot(target_seq,'r--', label='target')
        ax.legend()
        
        return ax
    
    def grouped():
        gr = torch.zeros(27)       
        
        idx = [0,1,2,
           8,9,10,
           12,13,14,
           16,17,18,
           24,25,26]
        
        gr[idx] = 1
        
        return gr
    
    def evenly():
        ev = torch.zeros(27)       
        
        idx = [0,1,2,
           6,7,8,
           12,13,14,
           18,19,20,
           24,25,26]
        
        ev[idx] = 1

        return ev