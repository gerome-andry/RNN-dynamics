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
PATH = Path(SCRATCH) / "npe_conv/lz96"
PATH.mkdir(parents=True, exist_ok=True)

CONFIG = {
    'batch_size' : [32, 64, 128],
    'epochs' : [1024],
    'diff_time_per_epoch' : [10, 50, 100],
    'lr': np.geomspace(1e-1, 1e-4, 4).tolist(),
    'wd': np.geomspace(1e-4, 1e-2, 4).tolist(),
    'diag_noise': [.1, .01, .001],
    'mem_size' : [32, 128, 512],
    'max_train_time' : [300],
    'test_time' : [1000],
    'better_init_GRU': [False],
    'device': ['cuda']
}

def build(**config):
    rnn = nn.GRU(1, config['mem_size'], bias = False, batch_first = True).to(config['device'])
    decoder = nn.Linear(config['mem_size'], 1)

    if config['better_init_GRU']:
        diag = torch.ones((config['mem_size']))
        diag += config['diag_noise']*torch.randn_like(diag)
        rnn.weight_hh_l[0][-config['mem_size']:][range(config['mem_size']), range(config['mem_size'])] = diag

    return rnn, decoder 


@job(array = 1, cpus=2, gpus=1, ram="16GB", time="6:00:00")
def GRU_search(i):
    seed = torch.randint(100, (1,))
    torch.manual_seed(seed)

    config = {key:np.random.choice(values) for key,values in CONFIG}

    run = wandb.init(project="dynRNN", config=config, group=f"GRU_analysis")
    runpath = PATH / f"runs/{run.name}_{run.id}"
    runpath.mkdir(parents=True, exist_ok=True)

    #model 
    rnn,decoder = build(**config)
    size = sum(param.numel() for param in zip(rnn.parameters(), decoder.parameters()))
    run.config.n_param = size
    run.config.seed = seed

    batch_sz = config['batch_size']
    n_max_time = config['diff_time_per_epoch']
    t_train = config['max_train_time']
    t_test = config['test_time']
    best_test_loss = torch.inf

    #optim
    optimizer = torch.optim.AdamW(
        list(rnn.parameters()) + list(decoder.parameters()),
        lr = config['lr'],
        weight_decay=config['wd'],
    )

    dev = config['device']
    for ep in trange(config['epoch']):
        loss_train = []
        loss_test = 0

        rnn.train()
        decoder.train()
        for it in range(n_max_time):
            mt = np.random.randint(100, t_train)
            data = CopyFirstInput.get_batch(batch_sz, mt).to(dev)

            optimizer.zero_grad()
            pred = decoder(rnn(data)[1])
            l = CopyFirstInput.loss(data[:,-1,:], pred)
            l.backward()

            loss_train.append(l.detach())

        rnn.eval()
        decoder.eval()
        with torch.no_grad():
            data = CopyFirstInput.get_batch(512, t_test).to(dev)

            out_seq, last_out = rnn(data)
            pred = decoder(last_out)
            l = CopyFirstInput.loss(data[:,-1,:], pred)

            loss_test = l.item()

            ax = CopyFirstInput.show_pred(out_seq[0], data[0])
            plt.show()
            run.log({"Prediction" : wandb.Image(plt)})

        loss_t = torch.stack(loss_train).mean().item()

        run.log(
            {
                "train_loss":loss_t,
                "test_loss":loss_test,
                "epoch":ep
            }
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




