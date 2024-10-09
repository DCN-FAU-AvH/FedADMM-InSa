import copy, torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from utils.dataset import DatasetSplit
from utils.model import init_model


class Clients(object):
    """Clients' local update processes."""

    def __init__(self, cfg, server_model, dataset):
        self.cfg = cfg  # scenario config
        self.device = cfg.device  # gpu or cpu
        self.datasets = dataset  # clients' datasets
        self._init_params(cfg, server_model)  # initialize parameters

    def local_update(self, model_server, selected_clients, k):
        """Clients' local update processes."""
        self.model_server = model_server
        self.k = k
        for self.i in range(self.cfg.m):
            if self.i in selected_clients:
                # Load client i's local settings
                self.model_i = copy.deepcopy(self.model_server).train()
                self.dataset_i = DatasetSplit(self.datasets, self.i)
                data_loader = init_data_loader(self.cfg, self.dataset_i)
                optimizer = init_optimizer(self.cfg, self.model_i, k)
                count_s, count_e = 0, 0  # finished SGD steps or epochs
                # Start local training
                train_flag = True  # flag for keep local training
                while train_flag:  # update client i's model parameters u_i
                    # check the criterion after each epoch
                    if self._stop_check(count_e):
                        break
                    for data, labels in data_loader:  # iterate over batches
                        self.data, self.labels = data.to(self.device), labels.to(self.device)
                        self._local_step(self.cfg)  # perform single FGD/SGD step
                        optimizer.step()  # update client's model parameters u
                        count_s += 1
                    count_e += 1
                # Personalized actions of different algorithms.
                self._local_step_personalize(self.cfg)
                self.models[self.i] = self.model_i.state_dict()
                self.steps[self.i][self.k] = count_e
                self.res_dict["steps"] = self.steps  # record local FGD/SGD steps
        return self.send_back

    def _stop_check(self, count):
        """Determine whether to stop client's local training."""
        stop_training = False
        # Check the maximum number.
        if count >= self.cfg.E:
            stop_training = True
        # Check the admm inexactness criterion.
        elif self.cfg.alg in ["admm_insa", "admm_in"]:
            if self._inexact_check(count):
                stop_training = True  # stop training
        return stop_training

    def _local_step(self, cfg):
        """Client i performs single FGD/SGD step."""
        if cfg.alg in ["fedavg"]:
            self._grad(self.model_i, self.data, self.labels)
        elif cfg.alg in ["admm", "admm_insa", "admm_in"]:
            if cfg.alg in ["admm_insa", "admm_in"] and self.cfg.bs == 0:
                pass  # grad calculated during the inexactness check, see self._inexact_e()
            else:
                self._admm_u(self.model_i, self.data, self.labels)
        else:
            raise ValueError(f"Invalid algorithm.")

    def _local_step_personalize(self, cfg):
        """Personalized actions of different algorithms."""
        # ADMM related local update process.
        if cfg.alg in ["admm_insa", "admm_in", "admm"]:
            beta_i_k = self.beta[self.i][self.k]
            self._update_lambda(beta_i_k)  # update dual variables lambda
            if self.cfg.alg == "admm_insa":  # udpate penlaty parameter beta
                self._adp_beta(beta_i_k)
            # record params to send back to the server
            self.send_back["beta"] = self.beta[:, self.k]  # beta_i^k use new
            self.res_dict["beta"] = self.beta  # record beta to plot

    def _update_lambda(self, beta):
        """Update the dual variable lambda."""
        u_state = self.model_i.state_dict()
        z_state = self.model_server.state_dict()
        for name, _ in self.model_i.named_parameters():  # without BN statistics
            self.lamda[self.i][name] -= beta * (u_state[name] - z_state[name])

    def _adp_beta(self, beta):
        """Update beta based on the adaptive penalty scheme."""
        u_kplus1, z_k = self.model_i, self.model_server
        u_k = state2model(self.cfg, self.models[self.i])
        primal_residual = beta * l2_norm(u_kplus1, u_k)
        dual_residual = l2_norm(u_kplus1, z_k)

        if self.cfg.mu * primal_residual < dual_residual:
            beta *= self.cfg.tau
        elif self.cfg.mu * dual_residual < primal_residual:
            beta /= self.cfg.tau
        self.beta[self.i][self.k + 1 :] = beta  # update beta

    def _inexact_check(self, count):
        """Check the inexactness criterion."""
        beta_tilde = self.beta[self.i][self.k] / self.cfg.c_i
        sigma = 0.999 * np.sqrt(2) / (np.sqrt(2) + np.sqrt(beta_tilde))
        # first time or using SGD, when e_u remains unchanged, calculate once and reuse
        if count == 0:
            self.e_u = self._inexact_e(self.model_server)  # use z^k instead of u_i^k
        e_u_new = self._inexact_e(self.model_i)
        return True if e_u_new <= sigma * self.e_u else False

    def _inexact_e(self, model: nn.Module):
        """Calculate the l2 norm of the inexactness residual e(u)."""
        data_loader_tmp = DataLoader(self.dataset_i, batch_size=500)
        model.eval()  # if using .train(), normalization layers cause problems
        model.zero_grad()
        for batch_idx, (data, labels) in enumerate(data_loader_tmp):
            data, labels = data.to(self.device), labels.to(self.device)
            pred = model(data)
            loss = model.loss(pred, labels)  # default reduction == "mean"
            loss.backward()  # accumulate grads
        u_state = model.state_dict()  # u_i
        z_state = self.model_server.state_dict()  # z
        for name, param in model.named_parameters():  # without BN statistics
            param.grad /= batch_idx + 1  # fix accumulated grads
            param.grad -= self.lamda[self.i][name]
            param.grad += self.beta[self.i][self.k] * (u_state[name] - z_state[name])
        res = 0
        for param in model.parameters():  # without BN statistics
            res += torch.linalg.norm(param.grad).square()
        return res.sqrt().item()

    def _grad(self, model, data, labels):
        """Calculate the gradients of the local update."""
        model.train()
        model.zero_grad()
        pred = model(data)
        loss = model.loss(pred, labels)  # default reduction == "mean"
        loss.backward()  # compute the gradients of f_i(u_i)

    def _admm_u(self, model, data, labels):
        """Calculate the gradients of the ADMM u-subproblem."""
        self._grad(model, data, labels)  # get gradients
        u_state = model.state_dict()  # u_i
        z_state = self.model_server.state_dict()  # z
        for name, param in model.named_parameters():  # without BN statistics
            param.grad -= self.lamda[self.i][name]
            param.grad += self.beta[self.i][self.k] * (u_state[name] - z_state[name])

    def _init_params(self, cfg, model_server):
        """Initialize parameters and hyperparameters."""
        model_state = model_server.state_dict()
        self.models = [copy.deepcopy(model_state) for _ in range(cfg.m)]
        self.send_back = {}  # Results to send back to the server
        self.send_back["models"] = self.models  # local model parameters
        self.steps = np.full((cfg.m, cfg.K + 1), 0)  # to save local FGD/SGD steps
        self.accus = np.full((cfg.m, cfg.K + 1), 0, dtype=float)  # local model accuracy
        if cfg.alg in ["admm_insa", "admm_in", "admm"]:
            # dual variables lambda
            lamda = copy.deepcopy(model_state)
            for key in model_state.keys():
                lamda[key].zero_()
            self.lamda = [copy.deepcopy(lamda) for _ in range(cfg.m)]
            self.send_back["lambda"] = self.lamda
            # penalty parameter beta
            self.beta = np.full((cfg.m, cfg.K + 2), cfg.beta, dtype=float)  # round k from 1 to K
            self.send_back["beta"] = [cfg.beta] * cfg.m


# auxiliary functions
def init_optimizer(cfg, model, k):
    """Initialize the optimizer."""
    lr = cfg.lr * np.power(cfg.lr_decay, k - 1) if cfg.lr_decay else cfg.lr
    if cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=cfg.decay, momentum=cfg.momentum
        )
    elif cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer


def init_data_loader(cfg, dataset_i, shuffle=True):
    """Initialize the dataloader."""
    if cfg.bs:  # SGD
        data_loader = DataLoader(dataset_i, batch_size=cfg.bs, shuffle=shuffle)
    else:  # FGD
        data_loader = DataLoader(dataset_i, batch_size=len(dataset_i), shuffle=shuffle)
    return data_loader


def l2_norm(x: nn.Module, y: nn.Module, square=False):
    """Calculate the l2 norm of x and y."""
    res = 0
    with torch.no_grad():
        for x_item, y_item in zip(x.parameters(), y.parameters()):  # without BN statistics
            res += torch.linalg.norm(x_item - y_item).square()
        res = res if square else torch.sqrt(res)
    return res.item()


def state2model(cfg: dict, state: dict):
    """Create a model with given parameters."""
    model = init_model(cfg, True)
    model.load_state_dict(state)
    return model
