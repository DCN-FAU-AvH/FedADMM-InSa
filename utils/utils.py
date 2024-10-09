import torch, os, sys, time, random, re, pytz, logging, yaml
import numpy as np
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties


def load_config(arg, save_cfg=True):
    """Load params from yaml and overwrite them with command line arguments."""
    os.system("cls")  # clear terminal
    os.system("clear")  # clear terminal
    arg_dict = vars(arg)  # parameters from the command line
    with open(arg.cfg, "r") as file:
        cfg = yaml.safe_load(file)  # parameters from the yaml file
    for key, value in arg_dict.items():  # do not set default values in args.py
        if value is not None:
            cfg[key] = value  # override yaml with args
    cfg = edict(cfg)  # allow to access dict values as attributes
    if save_cfg:
        cfg.dir_res = create_folder(cfg)  # folder to save the results
        print(f"Working in {cfg.dir_res}.\n")
        save_config(cfg, cfg.dir_res)  # save the test configuration
    return cfg


def save_config(cfg, dir_res):
    """Save config to a yaml file, cfg here is an EasyDict."""
    # cfg.pop("device", "")  # cannot be saved in yaml
    cfg.pop("progress", "")  # cannot be saved in yaml
    with open(os.path.join(dir_res, "config.yaml"), "w") as file:
        yaml.dump(dict(cfg), file, default_flow_style=False)


def get_logger(dir_res):
    """Return the logger."""
    dir_log = os.path.join(dir_res, "0_log.txt")
    logging.basicConfig(
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
        format="%(message)s",
    )
    log = logging.getLogger(dir_log)
    log.addHandler(logging.FileHandler(dir_log))
    return log


def set_seed(seed):
    """Setup the random seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_time_stamp(time_zone="Europe/Berlin"):
    """Return the time stamp of a specifid time zone."""
    time_stamp = int(time.time())
    time_zone = pytz.timezone(time_zone)
    test_time = pytz.datetime.datetime.fromtimestamp(time_stamp, time_zone)
    test_time = test_time.strftime("%Y%m%d-%H%M%S")
    return test_time


def create_folder(cfg, dir_parent="results", prefix="test_", postfix=""):
    """Create a test folder with a timestamp in Berlin time zone."""
    dir_parent = "results/debug" if cfg.debug else f"results/serious"
    if cfg.tag0:  # parent folder
        dir_parent = os.path.join(dir_parent, cfg.tag0)
    if cfg.dataset == "linreg":
        prefix = f"test_{cfg.dataset}_{cfg.tag}_"
    else:
        prefix = f"test_{cfg.dataset}_{cfg.model}_{cfg.tag}_"
    test_time = get_time_stamp(time_zone="Europe/Berlin")
    dir_results = os.path.join(sys.path[0], dir_parent, f"{prefix}{test_time}{postfix}")
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    return dir_results


def plot_res(cfg, data_path):
    # figure formats
    label_font = FontProperties(family="sans-serif", weight="normal", size=12)
    data_dir = os.path.split(data_path)[0]
    data = np.load(data_path, allow_pickle=True).tolist()
    # remove nan and inf values
    data["loss"] = [1e2 if x == float("inf") else x for x in data["loss"]]
    data["loss"] = [1e2 if isinstance(x, float) and (x != x) else x for x in data["loss"]]
    # set plot range
    start = 1
    stop = len(data["loss"])
    xaxis = np.arange(stop)
    # plot training loss
    fig = plt.figure(dpi=100)
    plt.plot(xaxis[start:stop], data["loss"][start:stop])
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Communication rounds", fontproperties=label_font)
    plt.ylabel("Loss", fontproperties=label_font)
    plt.xticks(fontproperties=label_font)
    plt.yticks(fontproperties=label_font)
    plt.title("Training loss")
    fig.set_facecolor("white")
    plt.savefig(f"{data_dir}/res_loss.jpg", bbox_inches="tight")
    plt.close()

    if cfg.dataset != "linreg":
        # plot test accuracy
        fig = plt.figure(dpi=100)
        plt.plot(xaxis[start:stop], data["accu_test"][start:stop])
        plt.xlabel(f"Communication rounds", fontproperties=label_font)
        plt.ylabel("Accuracy", fontproperties=label_font)
        plt.xticks(fontproperties=label_font)
        plt.yticks(fontproperties=label_font)
        plt.title("Test accuracy.")
        plt.yticks(np.arange(0, 1.01, step=0.1), fontproperties=label_font)
        plt.grid(linestyle="--", linewidth=0.5)
        fig.set_facecolor("white")
        plt.savefig(f"{data_dir}/res_accu.jpg", bbox_inches="tight")
        plt.close()

        if "admm" in cfg.alg:
            # plot average beta
            fig = plt.figure(dpi=100)
            data_beta = np.average(data["beta"], axis=0)
            plt.plot(xaxis[start:stop], data_beta[start:stop])
            plt.xlabel(f"Communication rounds", fontproperties=label_font)
            plt.ylabel(r"$\bar{\beta^k}$", fontproperties=label_font)
            plt.xticks(fontproperties=label_font)
            plt.yticks(fontproperties=label_font)
            plt.title(rf"The average of all clients' $\beta_i$")
            fig.set_facecolor("white")
            plt.savefig(f"{data_dir}/res_beta_avg.jpg", bbox_inches="tight")
            plt.close()

            # plot the average number of steps of active clients
            fig = plt.figure(dpi=100)
            data_steps = np.sum(data["steps"], axis=0)
            data_steps = data_steps / (
                cfg.m * cfg.frac
            )  # average number of steps of active clients
            plt.plot(xaxis[start:stop], data_steps[start:stop])
            plt.xlabel(f"Communication rounds", fontproperties=label_font)
            plt.ylabel("Number of GD steps", fontproperties=label_font)
            plt.xticks(fontproperties=label_font)
            plt.yticks(fontproperties=label_font)
            plt.title("The average number of steps of active clients.")
            fig.set_facecolor("white")
            plt.savefig(f"{data_dir}/res_steps_avg.jpg", bbox_inches="tight")
            plt.close()
