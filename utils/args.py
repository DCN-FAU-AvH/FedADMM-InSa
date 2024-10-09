import argparse, os, sys


def args_parser(cfg_file="config.yaml"):
    """
    Load parameters from the command line
    In general, no default values to avoid overwriting, see func load_config
    """
    args = argparse.ArgumentParser()
    path_cfg = os.path.join(sys.path[0], "utils", cfg_file)  # default path of the config file
    args.add_argument("--cfg", type=str, default=path_cfg, help="path to the config")
    args.add_argument("--alg", type=str, help="FL algorithm")
    args.add_argument("--beta_0", type=float, help="initial penalty parameter")
    args.add_argument("--bs", type=int, help="size of local mini-batches")
    args.add_argument("--dataset", type=str, help="name of the dataset")
    args.add_argument("--debug", type=str2bool, help="mode for debugging")
    args.add_argument("--E", type=int, help="number of epochs")
    args.add_argument("--frac", type=float, help="fraction of active clients")
    args.add_argument("--iid", type=str2bool, help="whether iid or not")
    args.add_argument("--inexact_z", type=str2bool, help="mode of the inexactness criterion")
    args.add_argument("--K", type=int, help="number of training rounds")
    args.add_argument("--lr", type=float, help="learning rate")
    args.add_argument("--decay", type=float, help="weight decay for SGD")
    args.add_argument("--momentum", type=float, help="momentum for SGD")
    args.add_argument("--c_i", type=float, help="strong convexity constant")
    args.add_argument("--m", type=int, help="number of users")
    args.add_argument("--model", type=str, help="name of the model")
    args.add_argument("--mu", type=float, help="adaptive penalty scheme parameter")
    args.add_argument("--tau", type=float, help="adaptive penalty scheme parameter")
    args.add_argument("--num_classes", type=int, help="number of label classes")
    args.add_argument("--optimizer", type=str, help="type of optimizer")
    args.add_argument("--seed", type=int, help="seed")
    args.add_argument("--subset", type=str2bool, help="use a subset of the whole dataset")
    args.add_argument("--tag", type=str, help="tag for saving")
    args.add_argument("--tag0", type=str, help="parent tag for saving")
    args.add_argument("--plot", type=str2bool, help="flag for plotting")
    args = args.parse_args()
    return args


def str2bool(str):
    """Parse true/false from the command line."""
    if isinstance(str, bool):
        return str
    if str.lower() in ["yes", "true", "t", "y", "1"]:
        return True
    elif str.lower() in ["no", "false", "f", "n", "0"]:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
