import numpy as np
import torch
from torch import nn

class NBLoss(torch.nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, mu, y, theta, eps=1e-8):
        """Negative binomial negative log-likelihood. It assumes targets `y` with n
        rows and d columns, but estimates `yhat` with n rows and 2d columns.
        The columns 0:d of `yhat` contain estimated means, the columns d:2*d of
        `yhat` contain estimated variances. This module assumes that the
        estimated mean and inverse dispersion are positive---for numerical
        stability, it is recommended that the minimum estimated variance is
        greater than a small number (1e-3).
        Parameters
        ----------
        yhat: Tensor
                Torch Tensor of reeconstructed data.
        y: Tensor
                Torch Tensor of ground truth data.
        eps: Float
                numerical stability constant.
        """
        if theta.ndimension() == 1:
            # In this case, we reshape theta for broadcasting
            theta = theta.view(1, theta.size(0))
        log_theta_mu_eps = torch.log(theta + mu + eps)
        res = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + y * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(y + theta)
            - torch.lgamma(theta)
            - torch.lgamma(y + 1)
        )
        res = _nan2inf(res)
        return -torch.mean(res)

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.LayerNorm(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.ReLU(),
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.activation == "ReLU":
            x = self.network(x)
            return self.relu(x)
        return self.network(x)


class VAE(torch.nn.Module):
    """
    Autoencoder
    """
    def __init__(
        self,
        num_genes,
        device="cuda",
        seed=0,
        decoder_activation="linear",
        hparams="",
    ):
        super(VAE, self).__init__()
        # set generic attributes
        self.num_genes = num_genes
        self.device = device
        self.seed = seed
        # early-stopping
        self.best_score = -1e3
        self.patience_trials = 0

        # set hyperparameters
        self.set_hparams_(hparams)

        # set models
        self.encoder = MLP(
            [num_genes]
            + [6000]
            + [self.hparams["dim"]]
        )

        self.decoder = MLP(
                [self.hparams["dim"]]
                + [6000] + [12000]
                + [num_genes],
                last_layer_act=decoder_activation,
                )

        # losses
        self.loss_autoencoder = nn.MSELoss(reduction='mean')

        self.iteration = 0

        self.to(self.device)

        # optimizers
        get_params = lambda model, cond: list(model.parameters()) if cond else []
        _parameters = (
            get_params(self.encoder, True)
            + get_params(self.decoder, True)
        )
        self.optimizer_autoencoder = torch.optim.Adam(
            _parameters,
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"],
        )

        self.normalize_total = Normalize_total()

    def forward(self, genes, return_latent=False, return_decoded=False):
        """
        If return_latent=True, act as encoder only. If return_decoded, genes should 
        be the latent representation and this act as decoder only.
        """
        if return_decoded:
            gene_reconstructions = self.decoder(genes)
            return gene_reconstructions

        latent = self.encoder(genes)
        if return_latent:
            return latent

        gene_reconstructions = self.decoder(latent)
      
        return gene_reconstructions

    def set_hparams_(self, hparams):
        """
        Set hyper-parameters to default values or values fixed by user.
        """

        self.hparams = {
            "dim": 1000,
            "autoencoder_width": 5000,
            "autoencoder_depth": 3,
            "adversary_lr": 3e-4,
            "autoencoder_wd": 4e-7, #4e-7
            "autoencoder_lr": 1e-5, #1e-5
        }

        return self.hparams


    def train(self, genes):
        """
        Train VAE.
        """
        genes = genes.to(self.device)
        gene_reconstructions = self.forward(genes)

        reconstruction_loss = self.loss_autoencoder(gene_reconstructions, genes)

        self.optimizer_autoencoder.zero_grad()
        reconstruction_loss.backward()
        self.optimizer_autoencoder.step()

        self.iteration += 1

        return {
            "loss_reconstruction": reconstruction_loss.item(),
        }



class Normalize_total(nn.Module):
    def __init__(self, target_sum=1e4):
        super(Normalize_total,self).__init__()
        self.target_sum = target_sum

    def forward(self, adata):
        counts_per_cell = adata.sum(axis=1)
        scale_factor = self.target_sum / counts_per_cell
        norm_adata = adata * scale_factor[:, np.newaxis]

        return norm_adata