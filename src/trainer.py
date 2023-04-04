from load_dataset import load_data

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


from models.ClusterGCNStyleSampling import ClusterGCN
from models.GraphSAINTStyleSampling import GraphSAINT
from models.GraphSAGEStyleSampling import GraphSAGE


from logger import logger


def get_model_path_from_args(args):
    return f"{args.save_dir}/{args.dataset}_{args.gnn_model}_{args.gnn_type}_{args.dim_hidden}_{args.num_layers}_{args.random_seed}.pt"


class Trainer(object):
    def __init__(self, args):

        self.dataset = args.dataset
        self.device = torch.device(f"cuda" if args.cuda else "cpu")
        self.args = args
        self.args.device = self.device

        self.gnn_model = args.gnn_model
        self.epochs = args.epochs
        self.eval_steps = args.eval_steps

        # used to indicate multi-label classification.
        self.multi_label = args.multi_label

        self.loss_op = torch.nn.BCEWithLogitsLoss() if self.multi_label else torch.nn.NLLLoss()

        (
            self.data,
            self.x,
            self.y,
            self.split_masks,
            self.evaluator,
            self.processed_dir,
        ) = load_data(args.dataset, args.dataset_dir)

        if self.gnn_model in ["GraphSAGE", "GCN"]:
            self.model = GraphSAGE(args, self.data, self.split_masks["train"], self.processed_dir)

        elif self.gnn_model == "GraphSAINT":
            self.model = GraphSAINT(args, self.data, self.split_masks["train"], self.processed_dir)

        elif self.gnn_model == "ClusterGCN":
            self.model = ClusterGCN(args, self.data, self.split_masks["train"], self.processed_dir)

        else:
            raise NotImplementedError

        logger.info(f"model: {self.model}")

        self.model.to(self.device)

        if args.pretrained_checkpoint != "no_pretrained_checkpoint":
            print("loading mlpinit weight......")
            self.model.load_state_dict(torch.load(args.pretrained_checkpoint), strict=False)

        if len(list(self.model.parameters())) != 0:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            self.optimizer = None

    def train_and_test(self):

        results = []
        best_valid_acc = 0

        out, losses, result = self.test_net()
        results.append(result)
        train_acc, valid_acc, test_acc = result
        train_loss, valid_loss, test_loss = losses

        init_dict = dict()
        init_dict["train_loss"] = train_loss
        init_dict["valid_loss"] = valid_loss
        init_dict["test_loss"] = test_loss
        init_dict["train_acc"] = train_acc
        init_dict["valid_acc"] = valid_acc
        init_dict["test_acc"] = test_acc

        logger.info(f"init_dict: {init_dict}")

        for epoch in range(1, self.epochs + 1):

            training_loss, train_acc = self.train_net(epoch)

            if epoch % self.eval_steps == 0:
                out, losses, result = self.test_net()
                results.append(result)
                train_acc, valid_acc, test_acc = result
                train_loss, valid_loss, test_loss = losses

                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_test_acc = test_acc
                    if self.args.save_dir != "":
                        print("saving model......")
                        torch.save(self.model.state_dict(), get_model_path_from_args(self.args))

            epoch_dict = {}
            epoch_dict["epoch"] = epoch
            epoch_dict["loss"] = training_loss
            epoch_dict["train_loss"] = train_loss
            epoch_dict["valid_loss"] = valid_loss
            epoch_dict["test_loss"] = test_loss
            epoch_dict["train_acc"] = train_acc
            epoch_dict["valid_acc"] = valid_acc
            epoch_dict["test_acc"] = test_acc
            logger.info(f"epoch_dict: {epoch_dict}")

        return train_acc, valid_acc, test_acc

    @torch.no_grad()
    def test_gnn_mlp(self):

        self.args.gnn_type = "gnn"

        if self.gnn_model in ["GraphSAGE", "GCN"]:
            self.model = GraphSAGE(
                self.args,
                self.data,
                self.split_masks["train"],
                self.processed_dir,
            )

        elif self.gnn_model == "GraphSAINT":
            self.model = GraphSAINT(
                self.args,
                self.data,
                self.split_masks["train"],
                self.processed_dir,
            )
            # print( self.model )

        elif self.gnn_model == "ClusterGCN":
            self.model = ClusterGCN(
                self.args,
                self.data,
                self.split_masks["train"],
                self.processed_dir,
            )
        self.model.to(self.device)

        self.model.load_state_dict(
            torch.load(
                f"{self.args.save_dir}/{self.args.dataset}_{self.args.gnn_model}_{self.args.gnn_type}_{self.args.dim_hidden}_{self.args.num_layers}_{self.args.random_seed}.pt"
            )
        )
        (
            out,
            (train_loss, val_loss, test_loss),
            (train_acc, valid_acc, test_acc),
        ) = self.test_net()
        print(
            f"{self.args.train_model_type}_{self.args.gnn_type}: "
            f"Best train: {100*train_acc:.2f}%, "
            f"Best valid: {100*valid_acc:.2f}% "
            f"Best test: {100*test_acc:.2f}%"
        )

        self.args.gnn_type = "mlp"
        if self.gnn_model in ["GraphSAGE", "GCN"]:
            self.model = GraphSAGE(
                self.args,
                self.data,
                self.split_masks["train"],
                self.processed_dir,
            )

        elif self.gnn_model == "GraphSAINT":
            self.model = GraphSAINT(
                self.args,
                self.data,
                self.split_masks["train"],
                self.processed_dir,
            )

        elif self.gnn_model == "ClusterGCN":
            self.model = ClusterGCN(
                self.args,
                self.data,
                self.split_masks["train"],
                self.processed_dir,
            )
        self.model.to(self.device)

        self.model.load_state_dict(torch.load(get_model_path_from_args(self.args)))
        (
            out,
            (train_loss, val_loss, test_loss),
            (train_acc, valid_acc, test_acc),
        ) = self.test_net()
        print(
            f"{self.args.train_model_type}_{self.args.gnn_type}: "
            f"Best train: {100*train_acc:.2f}%, "
            f"Best valid: {100*valid_acc:.2f}% "
            f"Best test: {100*test_acc:.2f}%"
        )

        return train_loss, val_loss, test_loss, train_acc, valid_acc, test_acc

    def train_net(self, epoch):
        self.model.train()
        input_dict = self.get_input_dict(epoch)
        train_loss, train_acc = self.model.train_net(input_dict)
        return train_loss, train_acc

    def get_input_dict(self, epoch):
        if self.gnn_model in ["GraphSAGE", "GraphSAINT", "ClusterGCN", "GCN"]:
            input_dict = {
                "x": self.x,
                "y": self.y,
                "optimizer": self.optimizer,
                "loss_op": self.loss_op,
                "device": self.device,
            }
        else:
            Exception(f"the model of {self.gnn_model} has not been implemented")
        return input_dict

    @torch.no_grad()
    def test_net(self):
        self.model.eval()
        input_dict = {
            "x": self.x,
            "y": self.y,
            "device": self.device,
            "model": self.model,
        }
        out = self.model.inference(input_dict)

        # compute loss
        if isinstance(self.loss_op, torch.nn.NLLLoss):
            out = F.log_softmax(out, dim=-1)

        y_true = self.y

        train_loss = self.loss_op(out[self.split_masks["train"]], y_true[self.split_masks["train"]]).item()
        val_loss = self.loss_op(out[self.split_masks["valid"]], y_true[self.split_masks["valid"]]).item()
        test_loss = self.loss_op(out[self.split_masks["test"]], y_true[self.split_masks["test"]]).item()

        if self.evaluator is not None:
            y_true = self.y.unsqueeze(-1)
            y_pred = out.argmax(dim=-1, keepdim=True)

            train_acc = self.evaluator.eval(
                {
                    "y_true": y_true[self.split_masks["train"]],
                    "y_pred": y_pred[self.split_masks["train"]],
                }
            )["acc"]
            valid_acc = self.evaluator.eval(
                {
                    "y_true": y_true[self.split_masks["valid"]],
                    "y_pred": y_pred[self.split_masks["valid"]],
                }
            )["acc"]
            test_acc = self.evaluator.eval(
                {
                    "y_true": y_true[self.split_masks["test"]],
                    "y_pred": y_pred[self.split_masks["test"]],
                }
            )["acc"]
        else:

            if not self.multi_label:
                pred = out.argmax(dim=-1).to("cpu")
                y_true = self.y
                correct = pred.eq(y_true)
                train_acc = correct[self.split_masks["train"]].sum().item() / self.split_masks["train"].sum().item()
                valid_acc = correct[self.split_masks["valid"]].sum().item() / self.split_masks["valid"].sum().item()
                test_acc = correct[self.split_masks["test"]].sum().item() / self.split_masks["test"].sum().item()

            else:
                pred = (out > 0).float().numpy()
                y_true = self.y.numpy()
                # calculating F1 scores
                train_acc = (
                    f1_score(
                        y_true[self.split_masks["train"]],
                        pred[self.split_masks["train"]],
                        average="micro",
                    )
                    if pred[self.split_masks["train"]].sum() > 0
                    else 0
                )

                valid_acc = (
                    f1_score(
                        y_true[self.split_masks["valid"]],
                        pred[self.split_masks["valid"]],
                        average="micro",
                    )
                    if pred[self.split_masks["valid"]].sum() > 0
                    else 0
                )

                test_acc = (
                    f1_score(
                        y_true[self.split_masks["test"]],
                        pred[self.split_masks["test"]],
                        average="micro",
                    )
                    if pred[self.split_masks["test"]].sum() > 0
                    else 0
                )

        return (
            out,
            (train_loss, val_loss, test_loss),
            (train_acc, valid_acc, test_acc),
        )
