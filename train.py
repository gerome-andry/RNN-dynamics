import torch 
import torch.nn as nn
import numpy as np 

from dynRNN.task import CopyFirstInput
import wandb 
from dawgz import job, schedule
import os 
from pathlib import Path 
import matplotlib.pyplot as plt

from tqdm import trange

from dynRNN.task import CopyFirstInput

SCRATCH = os.environ.get("SCRATCH", ".")
PATH = Path(SCRATCH) / "GRU_dyn/Cpy1in"
PATH.mkdir(parents=True, exist_ok=True)

CONFIG = {
    'batch_size' : [256],
    'epochs' : [1024],
    'diff_time_per_epoch' : [1024],
    'lr': [1e-3],
    'wd': [1e-3],
    'mem_size' : [64],
    'max_train_time' : [100],
    'test_time' : [300],
    'better_init_GRU': ['BiLSTM'],
    'device': ['cuda']
}

def build(**config):
    mz = int(config['mem_size'])
    if 'LSTM' in config['better_init_GRU']:
        rnn = nn.LSTM(1, mz, bias = False, batch_first=True).to(config['device'])
    else:        
        rnn = nn.GRU(1, mz, bias = False, batch_first = True).to(config['device'])
    
    decoder = nn.Linear(mz, 1).to(config['device'])

    if config['better_init_GRU'] == 'BRC':
        with torch.no_grad(): 
            rnn.weight_hh_l0[-mz:] = 2*torch.eye(mz, requires_grad=False)

    elif 'Bi' in config['better_init_GRU']:
        with torch.no_grad():
            rnn.weight_hh_l0[2*mz:3*mz][range(mz), range(mz)] += 2.
        
    return rnn, decoder 


@job(array = 3, cpus=2, gpus=1, ram="32GB", time="10:00:00")
def GRU_search(i):
    seed = torch.randint(100, (1,))
    torch.manual_seed(seed)

    config = {key:np.random.choice(values) for key,values in CONFIG.items()}

    run = wandb.init(project="dyn-RNN", config=config, group=f"GRU_analysis")
    runpath = PATH / f"runs/{run.name}_{run.id}"
    runpath.mkdir(parents=True, exist_ok=True)

    #model 
    rnn,decoder = build(**config)

    pars = list(rnn.parameters()) + list(decoder.parameters())
    size = sum(param.numel() for param in pars)
    run.config.n_param = size
    run.config.seed = seed

    batch_sz = config['batch_size']
    n_max_time = config['diff_time_per_epoch']
    t_train = config['max_train_time']
    t_test = config['test_time']
    best_test_loss = torch.inf

    #optim
    optimizer = torch.optim.AdamW(
        pars,
        lr = config['lr'],
        weight_decay=config['wd'],
    )

    dev = config['device']
    for ep in trange(config['epochs']):
        loss_train = []
        loss_test = 0

        rnn.train()
        decoder.train()
        for it in range(n_max_time):
            # tm = torch.randint(10, t_train, (1,))

            data = CopyFirstInput.get_batch(batch_sz, t_train).to(dev)

            if config['better_init_GRU'] == 'BRC':
                mz = config['mem_size']
                with torch.no_grad(): 
                    rnn.weight_hh_l0[-mz:] = 2*torch.eye(mz, requires_grad=False)
            
            pred = decoder(rnn(data)[0][:,-1])
            l = CopyFirstInput.loss(data[:,0,:], pred)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_train.append(l.detach())

        rnn.eval()
        decoder.eval()
        with torch.no_grad():
            data = CopyFirstInput.get_batch(512, t_test).to(dev)

            out_seq, _ = rnn(data)
            last_out = out_seq[:,-1]
            pred = decoder(last_out)
            l = CopyFirstInput.loss(data[:,0,:], pred)

            loss_test = l.item()

            ax = CopyFirstInput.show_pred(decoder(out_seq[0]).cpu(), data[0].cpu())
            plt.show()

            run.log({"Prediction" : wandb.Image(plt)}, step = ep)
            plt.close()

            mem_connect = rnn.weight_hh_l0[-config['mem_size']:]

            plt.imshow(mem_connect.cpu())
            plt.colorbar()
            run.log({'Memory connect' : wandb.Image(plt)}, step = ep)
            plt.close()

        loss_t = torch.stack(loss_train).mean().item()

        run.log(
            {
                "train_loss":loss_t,
                "test_loss":loss_test,
                "epoch":ep
            }, step = ep
        )

        if loss_test < best_test_loss:
            best_test_loss = loss_test
            torch.save(
                {
                    'rnn_check':rnn.state_dict(),
                    'decoder_check':decoder.state_dict()
                },
                runpath / 'checkpoint.pth',
            )

    run.finish()

if __name__ == '__main__':
    schedule(
        GRU_search,
        name='GRU_analysis',
        backend='slurm',
        settings={'export':'ALL'},
        env=[
            "conda activate dynRNN",
            "export WANDB_SILENT=true",
        ],
    )




