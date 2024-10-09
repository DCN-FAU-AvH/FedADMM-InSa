import os
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from utils.dataset import DatasetSplit
from utils.utils import *


def FL_train(cfg, server, clients, dataset):
    """Run FL training."""
    # Initialization.
    res_dict, res_path, log = prepare(cfg, clients)  # logging system
    evaluate(cfg, server.model, dataset, res_dict, log)  # record initial states
    task2 = cfg.progress.add_task("[green] Sub loop:", total=cfg.K)  # progress bar
    # Start training.
    log.info(f"\n-----Start training-----\n")
    train_start_time = time.time()
    for k in range(1, cfg.K + 1):  # Iterate over FL training rounds.
        log.info(f"\nRound: {k}")
        round_start_time = time.time()
        # server selects clients
        selected_clients = server.select_clients(frac=cfg.frac)
        # clients' local updates
        res_clients = clients.local_update(server.model, selected_clients, k)
        # server aggregation
        server.aggregate(res_clients)
        # evaluation and save results
        log.info(f" Round time: {(time.time() - round_start_time)/60:.1f} min")
        if k % cfg.save_freq == 0 or k in [1, cfg.K]:
            loss = evaluate(cfg, server.model, dataset, res_dict, log, k)
            np.save(res_path, np.asarray(res_dict, dtype=object))
            # torch.save(server.model, os.path.join(dir_res, f"/0_model.pth"))  # save model
            if cfg.plot:
                plot_res(cfg, res_path)
            if loss > 200 or np.isnan(loss):  # early stop
                log.info(f"\n# Early stop dut to the loss explosion #\n")
                break
        cfg.progress.update(task2, advance=1)  # complete one round
        log.info(f" Total time: {(time.time() - train_start_time)/60:.1f} min\n")
    cfg.progress.remove_task(task2)  # complete all rounds
    log.info(f"-----End training-----\n")


def prepare(cfg, clients):
    """Preparation for training."""
    res_dict = defaultdict(list)  # dict to save results
    res_path = os.path.join(cfg.dir_res, f"0_results.npy")  # path to saved results
    log = get_logger(cfg.dir_res)  # logging system
    clients.res_dict = res_dict  # to save results during clients' local updates
    clients.log = log  # to save log
    return res_dict, res_path, log


def evaluate(cfg, model, dataset, res_dict, log, k=0):
    """Evaluate model performance."""
    res_dict["round"].append(k)
    loss = cal_loss(cfg, model, dataset)
    res_dict["loss"].append(loss)
    log.info(f" Loss: {loss:.3f}")
    if cfg.dataset != "linreg":
        acc_test = cal_accu(cfg, model, dataset["test"])
        res_dict["accu_test"].append(acc_test)
        log.info(f" Accu: {acc_test:.3f}")
    else:
        res_dict["accu_test"].append(-1)  # dummy save for the plot
    return loss


def cal_loss(cfg, model, dataset, bs=500):
    """Return the loss of the model over the dataset."""
    loss = 0
    model.eval()
    with torch.no_grad():
        # loss of the training set
        for i in range(cfg.m):
            loss_i = 0  # client i's loss
            dataset_i = DatasetSplit(dataset, i)
            data_loader = DataLoader(dataset_i, batch_size=bs)
            for batch_idx, (data, label) in enumerate(data_loader):
                data, label = data.to(cfg.device), label.to(cfg.device)
                pred = model(data)
                loss_i += model.loss(pred, label)  # default reduction == "mean"
            loss_i /= batch_idx + 1
            loss += cfg.alpha[i] * loss_i  # total weighted loss
    return loss.item()


def cal_accu(cfg, model, dataset, bs=500):
    """Return the accuracy of the model over the dataset."""
    correct = 0
    model.eval()
    data_loader = DataLoader(dataset, batch_size=bs)
    with torch.no_grad():
        for data, label in data_loader:
            data, label = data.to(cfg.device), label.to(cfg.device)
            pred = model(data)
            _, pred = torch.max(pred, 1)  # predicted labels
            correct += (pred.view_as(label) == label).sum()  # correct predictions
        accuracy = correct / len(dataset)
    return accuracy.item()
