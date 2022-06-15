import pathlib

import torch

from intelcamp.error import ConfigsError
from intelcamp.models import lstm, rnn


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
    arch_version = int(configs["arch_version"])

    if arch_version == 4:
        return len(configs["qs"])

    elif arch_version == 5:
        return (
            configs["S2S_stagger"]["initial_num"]
            + configs["S2S_stagger"]["secondary_num"]
        ) * len(configs["qs"])

    else:
        ConfigsError(f"{arch_version} not a valid arch_version")


def init_model(configs):
    if configs["arch_type_variant"] == "vanilla":
        model = rnn.RNNModel
    elif configs["arch_type_variant"] == "lstm":
        model = lstm.LSTM_Model
    else:
        raise ConfigsError(
            f"{configs['arch_type_variant']} is not a supported architecture variant"
        )

    hidden_dim = int(configs["hidden_nodes"])
    output_dim = _get_output_dim(configs)
    input_dim = configs["input_dim"]
    layer_dim = configs["layer_dim"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return model(input_dim, hidden_dim, layer_dim, output_dim, device=device)


def load_model(configs):
    model = init_model(configs)

    filepath = pathlib.Path(configs["exp_dir"]) / "torch_model"
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint["model_state_dict"])
    resume_num_epoch = checkpoint["epoch_num"]
    resume_n_iter = checkpoint["n_iter"]

    return model, resume_num_epoch, resume_n_iter
