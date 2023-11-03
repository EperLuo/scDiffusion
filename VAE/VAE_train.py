import argparse
import os
import time

import numpy as np
import torch
from VAE_model import VAE, MLP
import sys
sys.path.append("..")
from guided_diffusion.cell_datasets_muris import load_data

torch.autograd.set_detect_anomaly(True)
import random

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def prepare_vae(args, state_dict=None):
    """
    Instantiates autoencoder and dataset to run an experiment.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = load_data(
        data_dir=args["data_dir"],
        batch_size=args["batch_size"],
        vae=True,
        ae_dir=args["save_dir"],
        num_gene=args["num_genes"],
    )

    autoencoder = VAE(
        num_genes=args["num_genes"],
        device=device,
        seed=args["seed"],
        hparams="",
        decoder_activation=args["decoder_activation"],
    )
    if state_dict is not None:
        autoencoder.load_state_dict(state_dict)

    return autoencoder, datasets


def train_vae(args, return_model=False):
    """
    Trains a autoencoder
    """

    autoencoder, datasets = prepare_vae(args)
    
    args["hparams"] = autoencoder.hparams

    start_time = time.time()
    for step in range(args["max_steps"]):

        genes, _ = next(datasets)

        minibatch_training_stats = autoencoder.train(genes)

        if step % 1000 == 0:
            for key, val in minibatch_training_stats.items():
                print('step ', step, 'loss ', val)

        ellapsed_minutes = (time.time() - start_time) / 60

        stop = ellapsed_minutes > args["max_minutes"] or (
            step == args["max_steps"] - 1
        )

        if ((step % args["checkpoint_freq"]) == 0 or stop):

            os.makedirs(args["save_dir"],exist_ok=True)
            torch.save(
                autoencoder.state_dict(),
                os.path.join(
                    args["save_dir"],
                    "model_seed={}_step={}.pt".format(args["seed"], step),
                ),
            )

            if stop:
                break

    if return_model:
        return autoencoder, datasets


def parse_arguments():
    """
    Read arguments if this script is called from a terminal.
    """
    parser = argparse.ArgumentParser(description="Autoencoder for gene expression")
    # dataset arguments
    parser.add_argument("--data_dir", type=str, default='/data1/lep/Workspace/guided-diffusion/data/tabula_muris/all.h5ad')
    parser.add_argument("--loss_ae", type=str, default="mse")
    parser.add_argument("--decoder_activation", type=str, default="ReLU")

    # CPA arguments (see set_hparams_() in cpa.model.CPA)
    parser.add_argument("--local_rank", type=int, default=0)    
    parser.add_argument("--split_seed", type=int, default=1234)
    parser.add_argument("--num_genes", type=int, default=18996)# gene numbers after quality control
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hparams", type=str, default="")

    # training arguments
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--max_minutes", type=int, default=3000)
    parser.add_argument("--checkpoint_freq", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--save_dir", type=str, default='../checkpoint/AE/my_AE')
    parser.add_argument("--sweep_seeds", type=int, default=200)
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    seed_everything(1234)
    train_vae(parse_arguments())