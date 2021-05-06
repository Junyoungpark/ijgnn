import argparse
from time import perf_counter

import dgl
import torch
import wandb
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.data.chain_data import generate_graphs_seq
from src.net.IJGNN import IJGNN3
from src.utils import calc_mape


def main(args):
    ns_range = [20, 50]
    op = 'max'

    model = IJGNN3(nf_dim=1,
                   ef_dim=1,
                   hnf_dim=64,
                   hef_dim=64,
                   nf_outdim=1,
                   ef_outdim=1,
                   n_iters=args.internal_hops).to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingWarmRestarts(opt, T_0=32)
    loss_fn = torch.nn.MSELoss()

    config = {'data_order': args.data_order,
              'internal_hops': args.internal_hops}

    wandb.init(project='IJGNN',
               group='IJGNN3',
               config=config)

    for i in range(args.iters):
        if i % args.generate_g_every == 0:
            train_g = generate_graphs_seq(args.batch_size, ns_range, args.data_order, op)
            train_g = dgl.batch(train_g).to(args.device)

        start = perf_counter()
        train_nf, train_ef = train_g.ndata['x'], train_g.edata['x']
        train_y = train_g.ndata['y']
        train_pred, _ = model(train_g, train_nf, train_ef)
        loss = loss_fn(train_pred, train_y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        fit_time = perf_counter() - start

        # logging
        log_dict = {'loss': loss,
                    'mape': calc_mape(train_pred, train_y),
                    'fit_time': fit_time,
                    'lr': opt.param_groups[0]['lr']}
        wandb.log(log_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-iters', type=int, default=5000, help='number of gradient updates')
    parser.add_argument('-generate_g_every', type=int, default=32, help='sample regeneration interval')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-data_order', type=int, default=5, help='data generation parameter')
    parser.add_argument('-internal_hops', type=int, default=6, help='IJGNN internal hops')
    parser.add_argument('-device', type=str, default='cuda:0', help='device for computing tensors')
    args = parser.parse_args()
    main(args)
