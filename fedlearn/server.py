import copy
import torch
import numpy as np
from utils.model import init_model


class Server(object):
    """Server side operations."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = init_model(cfg)  # model z_k
        self.state = self.model.state_dict()  # state z_k

    def select_clients(self, frac):
        """Randomly select a subset of clients."""
        num = max(np.ceil(frac * self.cfg.m).astype(int), 1)  # number of clients to sample
        self.active_clients = np.random.choice(range(self.cfg.m), num, replace=False)
        return self.active_clients

    def aggregate(self, res_clients: dict):
        """Server aggregation process."""
        alpha = self.cfg.alpha
        model_u = res_clients["models"]  # list of clients' models
        model_z = copy.deepcopy(model_u[0])  # init model_server_new
        m = len(model_u)  # number of clients to aggregate
        # Iterate over model layers for aggregation.
        for key in model_z.keys():
            model_z[key].zero_()  # reset model parameters
            if "num_batches_tracked" in key:  # for BN batches count
                continue
            elif "running" in key:  # for BN running mean/var
                for i in range(m):
                    model_z[key] += alpha[i] * model_u[i][key]
            else:  # for other layers
                # FedADMM server aggregation.
                if self.cfg.alg in ["admm", "admm_in", "admm_insa"]:
                    beta, lamda = res_clients["beta"], res_clients["lambda"]
                    for i in range(m):  # iterate over clients
                        model_z[key] += alpha[i] * (beta[i] * model_u[i][key] - lamda[i][key])
                    tmp = [alpha[i] * beta[i] for i in range(m)]
                    model_z[key] = torch.div(model_z[key], sum(tmp))
                # FedAvg server aggregation.
                elif self.cfg.alg in ["fedavg"]:
                    for i in range(m):  # iterate over clients
                        model_z[key] += alpha[i] * model_u[i][key]
                else:
                    raise ValueError(f"Invalid algorithm.")
        # Server aggregation with memory
        if self.cfg.alg in ["admm_in", "admm_insa"]:
            model_z = self._admm_memory(model_z)
        self.model.load_state_dict(model_z)  # update server's model

    def _admm_memory(self, model_new):
        """Server aggregation with memory."""
        delta = self.cfg.delta
        cof1 = 1 / (1 + delta)
        cof2 = delta / (1 + delta)
        model = copy.deepcopy(model_new)
        for key in model.keys():  # iterate over model layers
            model[key].zero_()  # reset model parameters
            model[key] = cof1 * model_new[key] + cof2 * self.state[key]
        return model
