import fastai.text.all as fta
import re
import torch
import pandas as pd
import numpy as np
import os
import sys

# +
from fastai.callback.core import Callback
from fastai.learner import CancelFitException


class NoOverfittingCB(Callback):
    def __init__(self, patience=3, min_delta=0.01, reset_on_fit=True):
        self.patience = patience
        self.min_delta = min_delta
        self.reset_on_fit = reset_on_fit

    def before_fit(self):
        if self.reset_on_fit is None:
            self.best = float("inf")
        self.wait = 0

    def after_epoch(self):
        train_loss = self.learn.recorder.losses[-1]  # last training loss
        valid_loss = self.learn.recorder.values[-1][0]  # last validation loss

        diff = train_loss - valid_loss + self.min_delta

        if diff < 0:
            self.wait += 1
        else:
            self.wait = 0

        if self.wait >= self.patience:
            print("Training cancelled due to overfitting...")
            raise CancelFitException()

    def after_fit(self):
        self.run = True


# +
# import torch.nn as nn
# import torch.nn.functional as F

# class CustomLoss(nn.Module):
#     y_int=True # y interpolation
#     def __init__(self,
#         gamma:float=2.0, # Focusing parameter. Higher values down-weight easy examples' contribution to loss
#         weight:Tensor=None, # Manual rescaling weight given to each class
#         reduction:str='mean' # PyTorch reduction to apply to the output
#     ):
#         "Applies Focal Loss: https://arxiv.org/pdf/1708.02002.pdf"
#         store_attr()

#     def forward(self, inp:Tensor, targ:Tensor) -> Tensor:
#         "Applies focal loss based on https://arxiv.org/pdf/1708.02002.pdf"
#         ce_loss = F.cross_entropy(inp, targ, weight=self.weight, reduction="none")
#         p_t = torch.exp(-ce_loss)
#         loss = (1 - p_t)**self.gamma * ce_loss
#         if self.reduction == "mean":
#             loss = loss.mean()
#         elif self.reduction == "sum":
#             loss = loss.sum()
#         return loss

#     def decodes(self, x:Tensor) -> Tensor:
#         "Converts model output to target format"
#         return x.argmax(dim=self.axis)

#     def activation(self, x:Tensor) -> Tensor:
#         "`F.cross_entropy`'s fused activation function applied to model output"
#         return F.softmax(x, dim=self.axis)


# -


def find_appropriate_lr(
    model: fta.Learner,
    lr_diff: int = 15,
    loss_threshold: float = 0.05,
    adjust_value: float = 1,
) -> float:
    # Run the Learning Rate Finder
    model.lr_find()

    # Get loss values and their corresponding gradients, and get lr values
    losses = np.array(model.recorder.losses)
    assert lr_diff < len(losses)
    loss_grad = np.gradient(losses)
    lrs = model.recorder.lrs

    # Search for index in gradients where loss is lowest before the loss spike
    # Initialize right and left idx using the lr_diff as a spacing unit
    # Set the local min lr as -1 to signify if threshold is too low
    r_idx = -1
    l_idx = r_idx - lr_diff
    while (l_idx >= -len(losses)) and (
        abs(loss_grad[r_idx] - loss_grad[l_idx]) > loss_threshold
    ):
        local_min_lr = lrs[l_idx]
        r_idx -= 1
        l_idx -= 1

    lr_to_use = local_min_lr * adjust_value
    return lr_to_use


class SAM(fta.Callback):
    "Sharpness-Aware Minimization"
    # https://arxiv.org/abs/2010.01412
    def __init__(self, zero_grad=True, rho=0.05, eps=1e-12, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.state = fta.defaultdict(dict)
        fta.store_attr()

    def params(self):
        return self.learn.opt.all_params(with_grad=True)

    def _grad_norm(self):
        return torch.norm(
            torch.stack([p.grad.norm(p=2) for p, *_ in self.params()]), p=2
        )

    @torch.no_grad()
    def first_step(self):
        scale = self.rho / (self._grad_norm() + self.eps)
        for p, *_ in self.params():
            self.state[p]["e_w"] = e_w = p.grad * scale
            p.add_(e_w)  # climb to the local maximum "w + e(w)"
        if self.zero_grad:
            self.learn.opt.zero_grad()

    @torch.no_grad()
    def second_step(self):
        for p, *_ in self.params():
            p.sub_(self.state[p]["e_w"])

    def before_step(self, **kwargs):
        self.first_step()
        self.learn.pred = self.model(*self.xb)
        self.learn("after_pred")
        self.loss_func(self.learn.pred, *self.yb).backward()
        self.second_step()


def get_weights(dls):

    # 0th index would provide the vocab from text
    # 1st index would provide the vocab from classes
    classes = dls.vocab[1]

    # Get label ids from the dataset using map
    # train_lb_ids = L(map(lambda x: x[1], dls.train_ds))
    # Get the actual labels from the label_ids & the vocab
    # train_lbls = L(map(lambda x: classes[x], train_lb_ids))

    # Combine the above into a single
    train_lbls = fta.L(map(lambda x: classes[x[1]], dls.train_ds))
    label_counter = fta.Counter(train_lbls)
    n_most_common_class = max(label_counter.values())
    print(f"Occurrences of the most common class {n_most_common_class}")

    # Source: https://discuss.pytorch.org/t/what-is-the-weight-values-mean-in-torch-nn-crossentropyloss/11455/9
    weights = [n_most_common_class / v for k, v in label_counter.items() if v > 0]
    return weights


def get_items(df):
    return df.to_dict(orient="records")


def get_y(record):
    return record["finch_cat_id"]


def get_x(record):
    return record["text"]


def textify_data(df: pd.DataFrame):
    """
    This method loads the local data and turns the features we
    want to predict from into a single text field, clean and neat!
    """
    df.loc[:, "text"] = (
        df.loc[:, "name"].apply(str)
        + df.loc[:, "brand_name"].apply(str)
        + df.loc[:, "description"].apply(str)
        + df.loc[:, "features"].apply(str)
    )

    punc = lambda text: re.sub(
        r"[^a-zA-Z0-9:$-,%.?!]+", " ", text
    )  # remove uneccesary punctuation
    numb = lambda text: re.sub(r"[^a-zA-Z:$-,%.?!]+", " ", text)  # remove numbers
    link = lambda text: re.sub(r"http\S+", "", text)  # remove links
    lists = lambda text: re.sub(r"[\[\]\(\)\{}]", "", text)
    df.loc[:, "text"] = df.loc[:, "text"].apply(punc)
    df.loc[:, "text"] = df.loc[:, "text"].apply(numb)
    df.loc[:, "text"] = df.loc[:, "text"].apply(link)
    df.loc[:, "text"] = df.loc[:, "text"].apply(lists)

    # Convert text to lowercase
    df.loc[:, "text"] = df.loc[:, "text"].str.lower()

    return df.loc[:, ["finch_cat_id", "text"]]


def download_train_save_model(model_name: str):
    print(f"WORKING ON {model_name}...")
    assert os.path.exists(
        f"{os.getenv('BASE_DIR')}/data/{model_name}/data.csv"
    ), f"data for {model_name}  doesn't exist"

    if not os.path.exists(f"{os.getenv('BASE_DIR')}/models"):
        os.mkdir(f"{os.getenv('BASE_DIR')}/models")

    print("Reading local data...", end="")
    df = pd.read_csv(f"{os.getenv('BASE_DIR')}/data/{model_name}/data.csv")
    print("done!")

    assert "finch_cat_id" in df.columns
    assert "text" in df.columns
    df = df[["text", "finch_cat_id"]]

    # temp truncating data:
    if model_name == "parent":
        df = pd.DataFrame(df.groupby("finch_cat_id").head(10000))

    tok = fta.Tokenizer.from_df(text_cols="text")
    text_block = fta.TextBlock.from_df("text", tok_tfm=tok)

    print("Creating datablock and dataloader...", end="")
    cls = fta.DataBlock(
        blocks=(text_block, fta.CategoryBlock),
        get_items=get_items,
        get_x=get_x,
        get_y=get_y,
        splitter=fta.RandomSplitter(),
    )
    bs = 64
    dls = cls.dataloaders(df, bs=bs)
    print("done!")

    avg = "macro"
    metrics = [
        fta.accuracy,
        fta.Precision(average=avg),
        fta.Recall(average=avg),
        fta.F1Score(average=avg),
    ]
    learn = fta.text_classifier_learner(
        dls,
        fta.AWD_LSTM,
        drop_mult=0.5,
        metrics=fta.accuracy,
        cbs=[
            fta.EarlyStoppingCallback(monitor="valid_loss", min_delta=0.05, patience=5),
            SAM(),
        ],
        loss_func=fta.FocalLossFlat(),  # CustomLoss(num_classes = len(dls.vocab[-1]), bs = bs),
    )

    print("Computing optimal learning rate...", end="")
    lr = find_appropriate_lr(learn)
    print("done")

    learn.fine_tune(5, lr)

    print(learn.predict("dove bar soap blah blah soapy soapy mmmm"))
    # learn.save(f"{os.getenv('BASE_DIR')}/models/{model_name}_model.pkl", with_opt=False)
    learn.export(f"{os.getenv('BASE_DIR')}/models/{model_name}_model.pkl")


if __name__ == "__main__":
    model_names = [
        "parent",
        "household_essentials",
        "baby",
        "pets",
        "home",
        "personal_care",
    ]
    assert (
        type(sys.argv[1]) == str
    ), f"bad input type! {sys.argv[1]}, try using: {model_names}"

    model_name = sys.argv[1]

    assert model_name in model_names, f"Bad Model Name! {model_name}"

    download_train_save_model(model_name)
