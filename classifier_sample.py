"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (   
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_and_diffusion_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
import scanpy as sc
import torch
from VAE.VAE_model import VAE

def load_VAE(ae_dir, num_gene):
    autoencoder = VAE(
        num_genes=num_gene,
        device='cuda',
        seed=0,
        hparams="",
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(ae_dir))
    return autoencoder

def save_data(all_cells, traj, data_dir):
    cell_gen = all_cells
    np.savez(data_dir, cell_gen=cell_gen)
    return

def main(cell_type=[0], multi=False, inter=False):
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("loading classifier...")
    if multi:
        args.num_class = 2 # how many classes in this condition
        classifier1 = create_classifier(**args_to_dict(args, (['num_class']+list(classifier_and_diffusion_defaults().keys()))[:3]))
        classifier1.load_state_dict(
            dist_util.load_state_dict(args.classifier_path1, map_location="cpu")
        )
        classifier1.to(dist_util.dev())
        classifier1.eval()

        args.num_class = 2 # how many classes in this condition
        classifier2 = create_classifier(**args_to_dict(args, (['num_class']+list(classifier_and_diffusion_defaults().keys()))[:3]))
        classifier2.load_state_dict(
            dist_util.load_state_dict(args.classifier_path2, map_location="cpu")
        )
        classifier2.to(dist_util.dev())
        classifier2.eval()

    else:
        args.num_class = 12 # how many classes in this condition
        classifier = create_classifier(**args_to_dict(args, (['num_class']+list(classifier_and_diffusion_defaults().keys()))[:3]))
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        )
        classifier.to(dist_util.dev())
        classifier.eval()

    '''
    control function for Gradient Interpolation Strategy
    '''
    def cond_fn_inter(x, t, y=None):
        assert y is not None
        y1 = y[:,0]
        y2 = y[:,1]
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected1 = log_probs[range(len(logits)), y1.view(-1)]
            selected2 = log_probs[range(len(logits)), y2.view(-1)]
            
            grad1 = th.autograd.grad(selected1.sum(), x_in, retain_graph=True)[0] * args.classifier_scale1
            grad2 = th.autograd.grad(selected2.sum(), x_in, retain_graph=True)[0] * args.classifier_scale2

            return grad1+grad2

    '''
    control function for multi-conditional generation
    Two conditional generation here
    '''
    def cond_fn_multi(x, t, y=None):
        assert y is not None
        y1 = y[:,0]
        y2 = y[:,1]
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits1 = classifier1(x_in, t)
            log_probs1 = F.log_softmax(logits1, dim=-1)
            selected1 = log_probs1[range(len(logits1)), y1.view(-1)]

            logits2 = classifier2(x_in, t)
            log_probs2 = F.log_softmax(logits2, dim=-1)
            selected2 = log_probs2[range(len(logits2)), y2.view(-1)]
            
            grad1 = th.autograd.grad(selected1.sum(), x_in, retain_graph=True)[0] * args.classifier_scale1
            grad2 = th.autograd.grad(selected2.sum(), x_in, retain_graph=True)[0] * args.classifier_scale2
            
            return grad1+grad2

    '''
    control function for one conditional generation
    '''
    def cond_fn_ori(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            grad = th.autograd.grad(selected.sum(), x_in, retain_graph=True)[0] * args.classifier_scale
            return grad
        
    def model_fn(x, t, y=None):
        assert y is not None
        if args.class_cond:
            return model(x, t, y if args.class_cond else None)
        else:
            return model(x, t)
        
    if inter:
        # input real cell expression data as initial noise
        ori_adata = sc.read_h5ad('/data1/lep/Workspace/guided-diffusion/data/tabula_muris/all.h5ad')
        sc.pp.normalize_total(ori_adata, target_sum=1e4)

    logger.log("sampling...")
    all_cell = []
    all_labels = []
    while len(all_cell) * args.batch_size < args.num_samples:
        model_kwargs = {}

        if not multi and not inter:
            classes = (cell_type[0])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)

        if multi:
            classes1 = (cell_type[0])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            classes2 = (cell_type[1])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            # classes3 = ... if more conditions
            classes = th.stack((classes1,classes2), dim=1)

        if inter:
            classes1 = (cell_type[0])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            classes2 = (cell_type[1])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            classes = th.stack((classes1,classes2), dim=1)

        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        if inter:
            adata = ori_adata.copy()

            start_x = adata.X.toarray()
            autoencoder = load_VAE(args.ae_dir, args.num_gene)
            start_x = autoencoder(torch.tensor(start_x,device=dist_util.dev()),return_latent=True).detach().cpu().numpy()

            n, m = start_x.shape  
            if n >= args.batch_size:  
                start_x = start_x[:args.batch_size, :]  
            else:  
                repeat_times = args.batch_size // n  
                remainder = args.batch_size % n  
                start_x = np.concatenate([start_x] * repeat_times + [start_x[:remainder, :]], axis=0)  
            
            noise = diffusion.q_sample(th.tensor(start_x,device=dist_util.dev()),1800*th.ones(start_x.shape[0],device=dist_util.dev(),dtype=torch.long),)

        if multi:
            sample, traj = sample_fn(
                model_fn,
                (args.batch_size, args.input_dim),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_multi,
                device=dist_util.dev(),
                noise = None,
            )
        elif inter:
            sample, traj = sample_fn(
                model_fn,
                (args.batch_size, args.input_dim),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_inter,
                device=dist_util.dev(),
                noise = noise,
            )
        else:
            sample, traj = sample_fn(
                model_fn,
                (args.batch_size, args.input_dim),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_ori,
                device=dist_util.dev(),
                noise = None,
            )

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_cell.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_cell) * args.batch_size} samples")

    arr = np.concatenate(all_cell, axis=0)
    save_data(arr, traj, args.sample_dir)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1000,
        batch_size=1000,
        use_ddim=False,
        class_cond=False, 

        model_path="checkpoint/muris_all/model800000.pt", 

        # commen conditional generation
        classifier_path="checkpoint/classifier_muris_all/model799999.pt",
        # multi-conditional
        classifier_path1="checkpoint/classifier_muris_mam_spl_organ/model599999.pt",
        classifier_path2="checkpoint/classifier_muris_mam_spl_T_B/model599999.pt",

        # commen conditional generation
        classifier_scale=2,
        # in multi-conditional, scale1 and scale2 are the weights of two classifiers
        # in Gradient Interpolation, scale1 and scale2 are the weights of two gradients
        classifier_scale1=2,
        classifier_scale2=2,     

        # if gradient interpolation
        ae_dir='VAE/checkpoint/muris_all/model_seed=0_step=800000.pt',
        num_gene=18996,

        sample_dir="output/test_cell",

    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main(cell_type=[0])

    # for multi-condition, run
    # main(cell_type=[class1,class2],multi=True)

    # for Gradient Interpolation, run
    # main(cell_type=[0,1],inter=True)