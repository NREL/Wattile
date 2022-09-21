import pathlib

import pandas as pd
import torch

from wattile.error import ConfigsError
from wattile.models import lstm, rnn


def save_model(model, epoch_num, n_iter, filepath):
    torch.save(
        {
            "epoch_num": epoch_num,
            "model_state_dict": model.state_dict(),
            "n_iter": n_iter,
        },
        filepath,
    )


def _get_output_dim(configs):

    window_width_target = configs["data_processing"]["input_output_window"][
        "window_width_target"
    ]
    resample_interval = configs["data_processing"]["resample_interval"]
    initial_num = int(
        pd.Timedelta(window_width_target) / pd.Timedelta(resample_interval)
    )
    arch_version = configs["learning_algorithm"]["arch_version"]

    if arch_version == "alfa":
        return len(configs["learning_algorithm"]["quantiles"])

    elif arch_version == "bravo":
        return (
            initial_num
            + configs["data_processing"]["input_output_window"]["secondary_num"]
        ) * len(configs["learning_algorithm"]["quantiles"])

    else:
        ConfigsError(f"{arch_version} not a valid arch_version")


def init_model(configs):
    if configs["learning_algorithm"]["arch_type_variant"] == "vanilla":
        model = rnn.RNNModel
    elif configs["learning_algorithm"]["arch_type_variant"] == "lstm":
        model = lstm.LSTM_Model
    else:
        raise ConfigsError(
            "{} is not a supported architecture variant".format(
                configs["learning_algorithm"]["arch_type_variant"]
            )
        )

    hidden_dim = int(configs["learning_algorithm"]["hidden_size"])
    output_dim = _get_output_dim(configs)
    input_dim = configs["input_dim"]
    num_layers = configs["learning_algorithm"]["num_layers"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return model(input_dim, hidden_dim, num_layers, output_dim, device=device)


def load_model(configs):
    model = init_model(configs)

    filepath = pathlib.Path(configs["data_output"]["exp_dir"]) / "torch_model"
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint["model_state_dict"])
    resume_num_epoch = checkpoint["epoch_num"]
    resume_n_iter = checkpoint["n_iter"]

    return model, resume_num_epoch, resume_n_iter
