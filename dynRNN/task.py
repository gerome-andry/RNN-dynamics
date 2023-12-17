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

#
# The system receives a sequence of digital inputs of variable size
class SequenceIngestion(Task):
    def get_batch(n, seq_len=300, t_tok=10, max_val=4):
        L_MAX = (seq_len-2*t_tok)//(2*t_tok)

        seq_in = torch.zeros((n,seq_len))
        seq_tg = torch.zeros((n,seq_len))
        for i in range(n):
            n_tok = int(torch.randint(low=1,high=L_MAX+1,size=(1,)))

            a = torch.randint(low=1, high=max_val+1, size=(n_tok,1)).float()
            a = a.repeat(1,t_tok).flatten()

            seq_in[i,t_tok:(n_tok+1)*t_tok] = a
            seq_tg[i,(n_tok+2)*t_tok:(2*n_tok+2)*t_tok] = a
        
        return seq_in.unsqueeze(-1), seq_tg.unsqueeze(-1)
    
    def loss(target, pred):
        return ((target-pred)**2).mean()
    
    def show_pred(pred, target, input_s):
        f, ax = plt.subplots()
        
        ax.plot(input_s, label='input')
        ax.plot(pred, label='pred')
        ax.plot(target,'r--', label='target')
        ax.legend()
        
        return ax

        
class IntervalProductionTask(Task):
    ### NOTE : In the paper the go pulse and signal pulses are sent thorugh two different neurons. Current implementation is 1 dimensional.
    # In this task:
    # 1 - the network perceives the interval T between the first two pulses of amplitude 1 and duration "pulses_duration",
    # 2 - mantains the interval during a delay of variable duration t2,
    # 3 - receives a "go" pulse of same aplitude and duration,
    # 4 - the network, after a time T, should produce a step of amplitude 1 until the end of task
    
    def get_batch(n, dt = 20, # dt in ms
                  min_t1_length=60, max_t1_length=500, # in ms
                  min_T_length=400, max_T_length=1400,# in ms
                  min_t2_length=600, max_t2_length=1600, # in ms
                  pulses_duration=60, padding_duration = 300, # in ms
                  add_noise = False, noise_stdev = 0.01):
        
        pulses_duration_ts = pulses_duration // dt
        
        max_sequence_length = max_t1_length + pulses_duration + max_T_length + pulses_duration + max_t2_length + max_T_length + padding_duration
        max_sequence_length //= dt
        
        t1_len = torch.randint(min_t1_length, max_t1_length, (n, 1)) // dt
        T_len = torch.randint(min_T_length, max_T_length, (n, 1))    // dt 
        t2_len = torch.randint(min_t2_length, max_t2_length, (n, 1)) // dt
        
        pulse_1_begin_indices = t1_len + torch.zeros((n, max_sequence_length), dtype=torch.long)
        pulse_1_end_indices = pulse_1_begin_indices + pulses_duration_ts
        
        pulse_2_begin_indices = pulse_1_end_indices + T_len
        pulse_2_end_indices =  pulse_2_begin_indices + pulses_duration_ts
        
        pulse_go_begin_indices = pulse_2_end_indices + t2_len
        pulse_go_end_indices = pulse_go_begin_indices + pulses_duration_ts

        input_sequences = torch.zeros((n, max_sequence_length, 1))

        # Create a mask for setting the continuous range for pulse 1
        mask_pulse_1 = (torch.arange(max_sequence_length).unsqueeze(0) >= pulse_1_begin_indices) & (torch.arange(max_sequence_length).unsqueeze(0) < pulse_1_end_indices)
        mask_pulse_1 = mask_pulse_1.unsqueeze(-1).to(torch.float)
        
        # Create a mask for setting the continuous range for pulse 2
        mask_pulse_2 = (torch.arange(max_sequence_length).unsqueeze(0) >= pulse_2_begin_indices) & (torch.arange(max_sequence_length).unsqueeze(0) < pulse_2_end_indices)
        mask_pulse_2 = mask_pulse_2.unsqueeze(-1).to(torch.float)
        
        # Create a mask for setting the continuous range for pulse go
        mask_pulse_go = (torch.arange(max_sequence_length).unsqueeze(0) >= pulse_go_begin_indices) & (torch.arange(max_sequence_length).unsqueeze(0) < pulse_go_end_indices)
        mask_pulse_go = mask_pulse_go.unsqueeze(-1).to(torch.float)

        input_sequences += mask_pulse_1 + mask_pulse_2 + mask_pulse_go
        if add_noise:
            input_sequences = torch.normal(input_sequences, noise_stdev)
        
        target_outputs = torch.zeros((n, max_sequence_length, 1))
        target_indices = pulse_go_end_indices + T_len
        
        target_mask = (torch.arange(max_sequence_length).unsqueeze(0) >= target_indices)
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
    ### NOTE : In the paper the each pulse is sent thorugh a different neuron (2 inputs). Current implementation is 1 dimensional.
    
    # In this task:
    # 1 - the network perceives a pulse of duration T1,
    # 2 - mantains the interval during a delay of variable duration t2,
    # 3 - the network perceives a second pulse of duration T2,
    # 5 - at the offset of T2 the network decides if T1 >= T2 (step to 1) or T1 > T2 (step to -1).
    
    def get_batch(n, dt = 20, # dt in ms
                  min_t1_length=60, max_t1_length=500, # in ms
                  min_T_length=400, max_T_length=1400,# in ms
                  min_t2_length=600, max_t2_length=1600, # in ms
                  padding_duration = 300, # in ms
                  add_noise = False, noise_stdev = 0.01):
        
        max_sequence_length = max_t1_length + max_T_length + max_t2_length + max_T_length + padding_duration
        max_sequence_length //= dt
        
        t1_len = torch.randint(min_t1_length, max_t1_length, (n, 1)) // dt
        T1_len = torch.randint(min_T_length, max_T_length, (n, 1))   // dt 
        t2_len = torch.randint(min_t2_length, max_t2_length, (n, 1)) // dt
        T2_len = torch.randint(min_T_length, max_T_length, (n, 1))   // dt 
        
        pulse_1_begin_indices = t1_len + torch.zeros((n, max_sequence_length), dtype=torch.long)
        pulse_1_end_indices   = pulse_1_begin_indices + T1_len
    
        pulse_2_begin_indices  =  pulse_1_end_indices + t2_len
        pulse_2_end_indices    = pulse_2_begin_indices + T2_len

        input_sequences = torch.zeros((n, max_sequence_length, 1))

        # Create a mask for setting the continuous range for pulse 1
        mask_pulse_1 = (torch.arange(max_sequence_length).unsqueeze(0) >= pulse_1_begin_indices) & (torch.arange(max_sequence_length).unsqueeze(0) < pulse_1_end_indices)
        mask_pulse_1 = mask_pulse_1.unsqueeze(-1).to(torch.float)
        
        # Create a mask for setting the continuous range for pulse 2
        mask_pulse_2 = (torch.arange(max_sequence_length).unsqueeze(0) >= pulse_2_begin_indices) & (torch.arange(max_sequence_length).unsqueeze(0) < pulse_2_end_indices)
        mask_pulse_2 = mask_pulse_2.unsqueeze(-1).to(torch.float)
        
        input_sequences += mask_pulse_1 + mask_pulse_2
        
        if add_noise:
            input_sequences = torch.normal(input_sequences, noise_stdev)
        
        target_outputs = torch.zeros((n, max_sequence_length, 1))
        comparison_result = torch.where(torch.gt(T1_len, T2_len), 1, -1)
        
        # Create a mask for target
        mask_target = (torch.arange(max_sequence_length).unsqueeze(0) >= pulse_2_end_indices)
        mask_target = mask_target.unsqueeze(-1).to(torch.float)
        target_outputs = comparison_result.view(n, 1, 1) * mask_target
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
    
    # In this task:
    # the network has 34 inputs. 32 are neurons that code for location and 2 for scalar pulses.
    # 1 - Network perceives a pulse with location cetered around a neuron of duration "pulses_duration",
    # 2 - After an interval T after the pulse with location a second scalar pulse of amplitude 1 and duration "pulses_duration", is sent
    # 3 - Then there is a delay interval of variable duration t2,
    # 4 - network receives a "go" scalar pulse of same aplitude and duration,
    # 4 - the network, after a time T, should produce a step centered at the same neuron as the first pulse.
    
    def get_batch(n, dt = 20, # dt in ms
                  min_t1_length=60, max_t1_length=500, # in ms
                  min_T_length=400, max_T_length=1400,# in ms
                  min_t2_length=600, max_t2_length=1600, # in ms
                  pulses_duration=60, padding_duration = 300, # in ms
                  add_noise = False, noise_stdev = 0.01):
        
        pulses_duration_ts = pulses_duration // dt
        max_sequence_length = max_t1_length + pulses_duration + max_T_length + pulses_duration + max_t2_length + max_T_length + padding_duration
        max_sequence_length //= dt
        
        pulses_duration_ts = pulses_duration // dt
        
        max_sequence_length = max_t1_length + pulses_duration + max_T_length + pulses_duration + max_t2_length + max_T_length + padding_duration
        max_sequence_length //= dt
        
        t1_len = torch.randint(min_t1_length, max_t1_length, (n, 1)) // dt
        T_len = torch.randint(min_T_length, max_T_length, (n, 1))    // dt 
        t2_len = torch.randint(min_t2_length, max_t2_length, (n, 1)) // dt
        
        pulse_1_begin_indices = t1_len + torch.zeros((n, max_sequence_length), dtype=torch.long)
        pulse_1_end_indices = pulse_1_begin_indices + pulses_duration_ts
        
        pulse_2_begin_indices = pulse_1_end_indices + T_len
        pulse_2_end_indices =  pulse_2_begin_indices + pulses_duration_ts
        
        pulse_go_begin_indices = pulse_2_end_indices + t2_len
        pulse_go_end_indices = pulse_go_begin_indices + pulses_duration_ts

        inputs_sequences = torch.zeros((n,max_sequence_length,34)) #shape (batch_size,sequence_length,34)

        # Create a mask for setting the continuous range for pulse 1
        mask_pulse_1 = (torch.arange(max_sequence_length).unsqueeze(0) >= pulse_1_begin_indices) & (torch.arange(max_sequence_length).unsqueeze(0) < pulse_1_end_indices)
        mask_pulse_1 = mask_pulse_1.unsqueeze(-1).to(torch.float)
        
        location_stimulus = TimedSpatialReproductionTask.generate_multiple_gaussians(n, 32, 6, 26, 2)
        
        # Create a mask for setting the continuous range for pulse 2
        mask_pulse_2 = (torch.arange(max_sequence_length).unsqueeze(0) >= pulse_2_begin_indices) & (torch.arange(max_sequence_length).unsqueeze(0) < pulse_2_end_indices)
        mask_pulse_2 = mask_pulse_2.unsqueeze(-1).to(torch.float)
        
        # Create a mask for setting the continuous range for pulse go
        mask_pulse_go = (torch.arange(max_sequence_length).unsqueeze(0) >= pulse_go_begin_indices) & (torch.arange(max_sequence_length).unsqueeze(0) < pulse_go_end_indices)
        mask_pulse_go = mask_pulse_go.unsqueeze(-1).to(torch.float)

        inputs_sequences[:, :, :32] += mask_pulse_1 * location_stimulus.unsqueeze(1)
        inputs_sequences[:,:,32:] += mask_pulse_2 + mask_pulse_go
        
        if add_noise:
            inputs_sequences = torch.normal(inputs_sequences, noise_stdev)
        
        target_outputs = torch.zeros((n,max_sequence_length,32))
        target_indices = pulse_go_end_indices + T_len
        
        target_mask = (torch.arange(max_sequence_length).unsqueeze(0) >= target_indices)
        target_mask = target_mask.unsqueeze(-1).to(torch.float)

        # Set the continuous range between t2_indices and target_indices to 1 in the target sequence
        target_outputs += target_mask * location_stimulus.unsqueeze(1)

        return inputs_sequences, target_outputs
    
    def generate_multiple_gaussians(L, n, range_min, range_max, sigma):
        indices = torch.arange(0, n, 1, dtype=torch.float32)
        centers = torch.randint(range_min, range_max, (L,))
        
        # Expand dimensions for broadcasting
        indices = indices.unsqueeze(0)  # Shape: (1, n)
        centers = centers.unsqueeze(1)  # Shape: (L, 1)
        
        # Calculate Gaussian curves using broadcasting
        gaussians = torch.exp(-0.5 * ((indices - centers) / sigma) ** 2)
        
        return gaussians
    
    def loss(target, pred):
        return ((target - pred)**2).mean()

    def plot_sequences(batch_input, batch_target):
        batch_size, sequence_length, sequence_dimensions = batch_input.size()

        for i in range(batch_size):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

            im1 = ax1.imshow(batch_input[i].T, cmap='viridis', interpolation='nearest')
            ax1.set_title(f'Batch input {i+1} Heatmap')
            ax1.set_xlabel('Sequence Length')
            ax1.set_ylabel('Sequence Dimensions')

            im2 = ax2.imshow(batch_target[i].T, cmap='viridis', interpolation='nearest')
            ax2.set_title(f'Batch target {i+1} Heatmap')
            ax2.set_xlabel('Sequence Length')
            ax2.set_ylabel('Sequence Dimensions')

            plt.tight_layout()
            plt.show()

    
if __name__ == '__main__':
    
    ### test of Interval production task
    #batch_input, batch_target = IntervalProductionTask.get_batch(5)
    #IntervalProductionTask.plot_sequences(batch_input,batch_target)
    
    ### test of Interval comparison task
    #batch_input, batch_target = IntervalComparisonTask.get_batch(5)
    #IntervalComparisonTask.plot_sequences(batch_input,batch_target)
    
    ### test of TimedSpatialReproductionTask
    #batch_input, batch_target = TimedSpatialReproductionTask.get_batch(1)
    #TimedSpatialReproductionTask.plot_sequences(batch_input,batch_target)
    
    ###
    