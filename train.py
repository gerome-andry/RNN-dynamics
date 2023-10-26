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
    'batch_size' : [64, 128, 256, 512],
    'epochs' : [2048],
    'diff_time_per_epoch' : [10, 50, 100],
    'lr': np.geomspace(1e-1, 1e-4, 4).tolist(),
    'wd': np.geomspace(1e-4, 1e-2, 4).tolist(),
    'diag_noise': [.1, .01, .001],
    'mem_size' : [32, 64, 128],
    'max_train_time' : [20],
    'test_time' : [100],
    'better_init_GRU': [True],
    'device': ['cuda']
}

def build(**config):
    mz = int(config['mem_size'])
    rnn = nn.GRU(1, mz, bias = False, batch_first = True).to(config['device'])
    decoder = nn.Linear(mz, 1).to(config['device'])

    if config['better_init_GRU']:
        with torch.no_grad():
            diag = nn.parameter.Parameter(2*torch.ones((mz)).to(config['device']))
            diag += config['diag_noise']*torch.randn_like(diag)
            rnn.weight_hh_l0[-mz:][range(mz), range(mz)] = diag

    return rnn, decoder 


@job(array = 3, cpus=2, gpus=1, ram="16GB", time="6:00:00")
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
            # mt = np.random.randint(100, t_train)
            data = CopyFirstInput.get_batch(batch_sz, t_train).to(dev)

            optimizer.zero_grad()
            pred = decoder(rnn(data)[1])
            l = CopyFirstInput.loss(data[:,0,:], pred)
            l.backward()

            loss_train.append(l.detach())

        rnn.eval()
        decoder.eval()
        with torch.no_grad():
            data = CopyFirstInput.get_batch(512, t_test).to(dev)

            out_seq, last_out = rnn(data)
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

        # if loss_test < best_test_loss:
        #     best_test_loss = loss_test
        #     torch.save(
        #         {
        #             'rnn_check':rnn.state_dict(),
        #             'decoder_check':decoder.state_dict()
        #         },
        #         runpath / 'checkpoint.pth',
        #     )

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




