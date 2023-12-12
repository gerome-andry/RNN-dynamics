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

    
class IntervalProductionTask(Task):
    ### TODO : In the paper the go cue and signal are sent thorugh two different neurons. Sequence should therefore be 2 dimensional. Current implementation is 1 dimensional.
    # In this task:
    # 1 - the network perceives the interval T between the first two pulses,
    # 2 - mantains the interval during a delay of variable duration t2,
    # 3 - receives a "go" cue,
    # 4 - the network should produces an action after an interval of duration T.
    
    def get_batch(n, min_t1_length=10, max_t1_length=20, min_T_length=10,
                  max_T_length=100, min_t2_length=10, max_t2_length=100, t_padding=10):
        
        max_sequence_length = max_t1_length + max_T_length + max_t2_length + max_T_length + t_padding

        t1_len = torch.randint(min_t1_length, max_t1_length, (n, 1))
        T_len = torch.randint(min_T_length, max_T_length, (n, 1))
        t2_len = torch.randint(min_t2_length, max_t2_length, (n, 1))

        batch_indices = torch.arange(n).unsqueeze(1)  # Shape: (n, 1)

        t1_indices = t1_len + torch.zeros((n, max_sequence_length), dtype=torch.long)
        T_indices = t1_indices + T_len
        t2_indices = T_indices + t2_len
        target_indices = t2_indices + T_len

        input_sequences = torch.zeros((n, max_sequence_length, 1))
        target_outputs = torch.zeros((n, max_sequence_length, 1))

        # Create a mask for setting the continuous range between t1 and T to 1
        mask = (torch.arange(max_sequence_length).unsqueeze(0) >= t1_indices) & (torch.arange(max_sequence_length).unsqueeze(0) < T_indices)
        mask = mask.unsqueeze(-1).to(torch.float)

        # Set the continuous range between t1 and T to 1
        input_sequences += mask

        input_sequences[batch_indices, t2_indices] = -1 # go cue
        
        target_mask = (torch.arange(max_sequence_length).unsqueeze(0) >= t2_indices) & (torch.arange(max_sequence_length).unsqueeze(0) < target_indices)
        target_mask = target_mask.unsqueeze(-1).to(torch.float)

        # Set the continuous range between t2_indices and target_indices to 1 in the target sequence
        target_outputs += target_mask

        return input_sequences, target_outputs
    
    def loss(target, pred):
        return ((target - pred)**2).mean()
    
    def show_pred(pred_seq, target_seq):
        n = pred_seq.size(0)
        plt.figure(figsize=(10, 5))
        for i in range(n):
            plt.subplot(n, 1, i+1)
            plt.title(f"Sequence {i+1}")
            plt.ylabel("Value")
            plt.xlabel("Time")
            plt.plot(pred_seq[i].numpy(),label='prediction')
            plt.plot(target_seq[i].numpy(),label='target')
            plt.legend()
        plt.tight_layout()
        plt.show()     
        return
    
    def plot_sequences(batch_input, batch_target):
        n = batch_input.size(0)
        plt.figure(figsize=(10, 10))
        for i in range(n):
            plt.subplot(n, 1, i+1)
            plt.title(f"Sequence {i+1}")
            plt.ylabel("Value")
            plt.xlabel("Time")
            plt.plot(batch_input[i].numpy(),label='input')
            plt.plot(batch_target[i].numpy(),label='target')
            plt.legend()
        plt.tight_layout()
        plt.show()
        
        
class IntervalComparisonTask(Task):
    ### TODO : In the paper the go cue and signal are sent thorugh two different neurons. Sequence should therefore be 2 dimensional. Current implementation is 1 dimensional.
    
    # In this task:
    # 1 - the network perceives the interval T1 between the first two pulses,
    # 2 - mantains the interval during a delay of variable duration t2,
    # 3 - the network perceives a second interval T2 between two pulses,
    # 4 - the network waits for the go signal after a random time t3
    # 5 - the network decides if T1 >= T2 (output 1) or T1 > T2 (output -1).
    
    def get_batch(n, min_t1_length=10, max_t1_length=20,
                     min_T_length=10, max_T_length=100,
                     min_t2_length=10, max_t2_length=100,
                     min_t3_length= 10, max_t3_length=20,
                     t_padding=10):
        
        max_sequence_length = max_t1_length + max_T_length + max_t2_length + max_T_length + max_t3_length+ t_padding

        t1_len = torch.randint(min_t1_length, max_t1_length, (n, 1))
        T1_len = torch.randint(min_T_length, max_T_length, (n, 1))
        t2_len = torch.randint(min_t2_length, max_t2_length, (n, 1))
        T2_len = torch.randint(min_T_length, max_T_length, (n, 1))
        t3_len = torch.randint(min_t3_length, max_t3_length, (n, 1))

        batch_indices = torch.arange(n).unsqueeze(1)  # Shape: (n, 1)
        
        t1_indices = t1_len + torch.zeros((n, max_sequence_length), dtype=torch.long)
        T1_indices = t1_indices + T1_len
        t2_indices = T1_indices + t2_len
        T2_indices = t2_indices + T2_len
        t3_indices = T2_indices + t3_len
        
        input_sequences = torch.zeros((n, max_sequence_length, 1))
        input_sequences[batch_indices, t1_indices] = 1
        input_sequences[batch_indices, T1_indices] = 1
        input_sequences[batch_indices, t2_indices] = 1
        input_sequences[batch_indices, T2_indices] = 1
        input_sequences[batch_indices, t3_indices] = 1
        
        target_outputs = torch.zeros((n, max_sequence_length, 1))
        comparison_result = torch.where(torch.gt(T1_len, T2_len), 1, -1)
        
        for i in range(n):
            target_outputs[i, t3_indices[i][0]:] = comparison_result[i].float() 

        return input_sequences, target_outputs
    
    def loss(target, pred):
        return ((target - pred)**2).mean()


    def show_pred(pred_seq, target_seq):
        n = pred_seq.size(0)
        plt.figure(figsize=(10, 5))
        for i in range(n):
            plt.subplot(n, 1, i+1)
            plt.title(f"Sequence {i+1}")
            plt.ylabel("Value")
            plt.xlabel("Time")
            plt.plot(pred_seq[i].numpy(),label='prediction')
            plt.plot(target_seq[i].numpy(),label='target')
            plt.legend()
        plt.tight_layout()
        plt.show()     
        return
    
    def plot_sequences(batch_input, batch_target):
        n = batch_input.size(0)
        plt.figure(figsize=(10, 5))
        for i in range(n):
            plt.subplot(n, 1, i+1)
            plt.title(f"Sequence {i+1}")
            plt.ylabel("Value")
            plt.xlabel("Time")
            plt.plot(batch_input[i].numpy(),label='input')
            plt.plot(batch_target[i].numpy(),label='target')
            plt.legend()
        plt.tight_layout()
        plt.show()
        
class TimedSpatialReproductionTask(Task):
    
    def get_batch(n, min_t1_length=10, max_t1_length=20,
                     min_T_length=10, max_T_length=100,
                     min_t2_length=10, max_t2_length=100,
                     min_t3_length= 10, max_t3_length=20,
                     t_padding=10):

        return None

    def loss(target, pred):
        return 0
    
    def plot_sequences(batch_input, batch_target):
        return None

    
if __name__ == '__main__':
    
    ### test of Interval production task
    batch_input, batch_target = IntervalProductionTask.get_batch(5)
    IntervalProductionTask.plot_sequences(batch_input,batch_target)
    
    ### test of Interval comparison task
    #batch_input, batch_target = IntervalComparisonTask.get_batch(5)
    #IntervalComparisonTask.plot_sequences(batch_input,batch_target)
