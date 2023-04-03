import argparse
import os

from trainer import Trainer
import random
import numpy as np
import torch

from logger import logger


def arg_parse():

    parser = argparse.ArgumentParser(description="MLPInit")

    parser.add_argument("--dataset", type=str, default="ogbn-products", choices=["ogbn-products", "ogbn-arxiv", "Reddit", "Flickr", "Yelp", "AmazonProducts", "Reddit2"])
    parser.add_argument(
        "--gnn_model",
        type=str,
        default="GraphSAGE",
        choices=["GraphSAGE", "ClusterGCN", "GraphSAINT", "GCN"],
    )
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument(
        "--cuda",
        type=bool,
        default=True,
        required=False,
        help="run in cuda mode",
    )

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="number of training the one shot model",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=5,
        help="interval steps to evaluate model performance",
    )

    parser.add_argument(
        "--multi_label",
        type=bool,
        default=False,
        help="multi_label or single_label task",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--dim_hidden", type=int, default=128)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="batch size depending on methods, " "need to provide fair batch for different approaches",
    )

    # parameters for MLPInit and randomInit
    parser.add_argument("--train_model_type", type=str, default="gnn", choices=["gnn", "mlp"], help="train model type, mlp for PeerMLP, gnn for MLPInit/RandomInit")
    parser.add_argument("--gnn_type", type=str, default="gnn")
    parser.add_argument("--use_checkpoint", type=bool, default=False)
    parser.add_argument("--pretrained_checkpoint", type=str, default="no_pretrained_checkpoint")

    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--random_seed", type=int, default=3)
    parser.add_argument("--dataset_dir", type=str, default="")


    # parameters for GraphSAINT
    parser.add_argument("--walk_length", type=int, default=2, help="walk length of RW sampler")
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--sample_coverage", type=int, default=0)
    parser.add_argument("--use_norm", type=bool, default=False)

    # parameters for ClusterGCN
    parser.add_argument("--num_parts", type=int, default=1500)



    args = parser.parse_args()
    args = reset_dataset_dependent_parameters(args)

    return args


# setting the common hyperparameters used for comparing different methods of a trick
def reset_dataset_dependent_parameters(args):
    if args.dataset == "Flickr":
        args.num_classes = 7
        args.num_feats = 500

    elif args.dataset == "Reddit":
        args.num_classes = 41
        args.num_feats = 602

    elif args.dataset == "Reddit2":
        args.num_classes = 41
        args.num_feats = 602

    elif args.dataset == "ogbn-products":
        args.multi_label = False
        args.num_classes = 47
        args.num_feats = 100

    elif args.dataset == "AmazonProducts":
        args.multi_label = True
        args.num_classes = 107
        args.num_feats = 200

    elif args.dataset == "Yelp":
        args.multi_label = True
        args.num_classes = 100
        args.num_feats = 300

    elif args.dataset == "ogbn-arxiv":
        args.num_feats = 128
        args.num_classes = 40
        args.N_nodes = 169343
    return args


def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


if __name__ == "__main__":
    args = arg_parse()

    # ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    # ch.setFormatter(formatter)
    # logger.addHandler(ch)

    logger.info(f"args: {args}")

    set_seed(args)

    # model training
    if (
        os.path.exists(f"./{args.dataset}_{args.gnn_model}_{args.train_model_type}.pt") and args.use_checkpoint == True
    ):  # using MLPInit
        trnr = Trainer(args)
        trnr.test_gnn_mlp()

    else:  # using random init
        args.gnn_type = args.train_model_type
        trnr = Trainer(args)
        train_loss, valid_acc, test_acc = trnr.train_and_test()

    logger.info(f"Done!!!!")
