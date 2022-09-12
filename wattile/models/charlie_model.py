import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init

PROJECT_DIRECTORY = Path().resolve().parent.parent


#########################################################################################
# building the S2S model
class S2S_Model(nn.Module):
    def __init__(self, cell_type, input_size, hidden_size, use_cuda):
        super(S2S_Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type

        if self.cell_type not in ["rnn", "gru", "lstm"]:
            raise ValueError(
                self.cell_type,
                " is not an appropriate cell type. Please select one of rnn, gru, or lstm.",
            )
        if self.cell_type == "rnn":
            self.Ecell = nn.RNNCell(self.input_size, self.hidden_size)
            self.Dcell = nn.RNNCell(1, self.hidden_size)
        if self.cell_type == "gru":
            self.Ecell = nn.GRUCell(self.input_size, self.hidden_size)
            self.Dcell = nn.GRUCell(1, self.hidden_size)
        if self.cell_type == "lstm":
            self.Ecell = nn.LSTMCell(self.input_size, self.hidden_size)
            self.Dcell = nn.LSTMCell(1, self.hidden_size)

        self.lin_usage = nn.Linear(self.hidden_size, 1)
        self.use_cuda = use_cuda
        self.init()

    # function to intialize weight parameters.
    # Refer to Saxe at al. paper that explains why to use orthogonal init weights
    def init(self):
        if self.cell_type == "rnn" or self.cell_type == "gru":
            for p in self.parameters():
                if p.dim() > 1:
                    init.orthogonal_(p.data, gain=1.0)
                if p.dim() == 1:
                    init.constant_(p.data, 0.0)
        elif self.cell_type == "lstm":
            for p in self.parameters():
                if p.dim() > 1:
                    init.orthogonal_(p.data, gain=1.0)
                if p.dim() == 1:
                    init.constant_(p.data, 0.0)
                    init.constant_(p.data[self.hidden_size : 2 * self.hidden_size], 1.0)

    def consume(self, x):
        # encoder forward function
        # for rnn and gru
        if self.cell_type == "rnn" or self.cell_type == "gru":
            h = torch.zeros(x.shape[0], self.hidden_size)
            if self.use_cuda:
                h = h.cuda()
            for T in range(x.shape[1]):
                h = self.Ecell(x[:, T, :], h)
            pred_usage = self.lin_usage(h)
        # for lstm
        elif self.cell_type == "lstm":
            h0 = torch.zeros(x.shape[0], self.hidden_size)
            c0 = torch.zeros(x.shape[0], self.hidden_size)
            if self.use_cuda:
                h0 = h0.cuda()
                c0 = c0.cuda()
            h = (h0, c0)
            for T in range(x.shape[1]):
                h = self.Ecell(x[:, T, :], h)
            pred_usage = self.lin_usage(h[0])
        return pred_usage, h

    def predict(self, pred_usage, h, target_length):
        # decoder forward function
        preds = []
        # for rnn and gru
        if self.cell_type == "rnn" or self.cell_type == "gru":
            for step in range(target_length):
                h = self.Dcell(pred_usage, h)
                pred_usage = self.lin_usage(h)
                preds.append(pred_usage.unsqueeze(1))
            preds = torch.cat(preds, 1)
        # for lstm
        elif self.cell_type == "lstm":
            for step in range(target_length):
                h = self.Dcell(pred_usage, h)
                pred_usage = self.lin_usage(h[0])
                preds.append(pred_usage.unsqueeze(1))
            preds = torch.cat(preds, 1)
        return preds


#########################################################################################
# Bahdanau Attention model
# refer to : AuCson github code
# building the model
class S2S_BA_Model(nn.Module):
    """
    class `S2S_BA_Model`: S2S model with Bahdanau Attention model
    (https://machinelearningmastery.com/the-bahdanau-attention-mechanism/)
    """

    def __init__(self, cell_type, input_size, hidden_size, use_cuda):
        super(S2S_BA_Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.cell_type = cell_type

        if self.cell_type not in ["rnn", "gru", "lstm"]:
            raise ValueError(
                self.cell_type,
                " is not an appropriate cell type. Please select one of rnn, gru, or lstm.",
            )
        if self.cell_type == "rnn":
            self.Ecell = nn.RNNCell(self.input_size, self.hidden_size)
            self.Dcell = nn.RNNCell(1 + self.hidden_size, self.hidden_size)
        if self.cell_type == "gru":
            self.Ecell = nn.GRUCell(self.input_size, self.hidden_size)
            self.Dcell = nn.GRUCell(1 + self.hidden_size, self.hidden_size)
        if self.cell_type == "lstm":
            self.Ecell = nn.LSTMCell(self.input_size, self.hidden_size)
            self.Dcell = nn.LSTMCell(1 + self.hidden_size, self.hidden_size)

        self.Wattn_energies = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.Wusage = nn.Linear(self.hidden_size, 1)
        self.Wout = nn.Linear(1 + self.hidden_size * 2, self.hidden_size)
        self.v = nn.Parameter(torch.rand(self.hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.init()

    # function to intialize weight parameters
    def init(self):
        if self.cell_type == "rnn" or self.cell_type == "gru":
            for p in self.parameters():
                if p.dim() > 1:
                    init.orthogonal_(p.data, gain=1.0)
                if p.dim() == 1:
                    init.constant_(p.data, 0.0)
        elif self.cell_type == "lstm":
            for p in self.parameters():
                if p.dim() > 1:
                    init.orthogonal_(p.data, gain=1.0)
                if p.dim() == 1:
                    init.constant_(p.data, 0.0)
                    init.constant_(p.data[self.hidden_size : 2 * self.hidden_size], 1.0)

    def consume(self, x):
        # for rnn and gru
        if self.cell_type == "rnn" or self.cell_type == "gru":
            # encoder forward function
            h = torch.zeros(x.shape[0], self.hidden_size)
            encoder_outputs = torch.zeros(x.shape[1], x.shape[0], self.hidden_size)
            if self.use_cuda:
                h = h.cuda()
                encoder_outputs = encoder_outputs.cuda()
            # encoder part
            for T in range(x.shape[1]):
                h = self.Ecell(x[:, T, :], h)
                encoder_outputs[T] = h
            pred_usage = self.Wusage(h)
        # for lstm
        elif self.cell_type == "lstm":
            h0 = torch.zeros(x.shape[0], self.hidden_size)
            c0 = torch.zeros(x.shape[0], self.hidden_size)
            encoder_outputs = torch.zeros(x.shape[1], x.shape[0], self.hidden_size)
            if self.use_cuda:
                h0 = h0.cuda()
                c0 = c0.cuda()
                encoder_outputs = encoder_outputs.cuda()
            h = (h0, c0)
            for T in range(x.shape[1]):
                h = self.Ecell(x[:, T, :], h)
                encoder_outputs[T] = h[0]
            pred_usage = self.Wusage(h[0])
        return pred_usage, h, encoder_outputs

    def predict(self, pred_usage, h, encoder_outputs, target_length):
        # decoder with attention function
        preds = []
        # for rnn and gru
        if self.cell_type == "rnn" or self.cell_type == "gru":
            for step in range(target_length):
                h_copies = h.expand(encoder_outputs.shape[0], -1, -1)
                energies = torch.tanh(
                    self.Wattn_energies(torch.cat((h_copies, encoder_outputs), 2))
                )
                score = torch.sum(self.v * energies, dim=2)
                attn_weights = score.t()
                attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(1)
                context = attn_weights.bmm(encoder_outputs.transpose(0, 1)).squeeze(1)
                gru_input = torch.cat((pred_usage, context), 1)
                h = self.Dcell(gru_input, h)
                output = self.Wout(torch.cat((pred_usage, h, context), 1))
                pred_usage = self.Wusage(output)
                preds.append(pred_usage.unsqueeze(1))
            preds = torch.cat(preds, 1)
        # for lstm
        elif self.cell_type == "lstm":
            for step in range(target_length):
                h_copies = h[0].expand(encoder_outputs.shape[0], -1, -1)
                energies = torch.tanh(
                    self.Wattn_energies(torch.cat((h_copies, encoder_outputs), 2))
                )
                score = torch.sum(self.v * energies, dim=2)
                attn_weights = score.t()
                attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(1)
                context = attn_weights.bmm(encoder_outputs.transpose(0, 1)).squeeze(1)
                gru_input = torch.cat((pred_usage, context), 1)
                h = self.Dcell(gru_input, h)
                output = self.Wout(torch.cat((pred_usage, h[0], context), 1))
                pred_usage = self.Wusage(output)
                preds.append(pred_usage.unsqueeze(1))
            preds = torch.cat(preds, 1)
        return preds


#########################################################################################
# Luong Attention module
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method

        if self.method not in ["dot", "general", "concat"]:
            raise ValueError(
                self.method,
                " is not an appropriate attention method, "
                "please select one of dot, general, or concat.",
            )
        self.hidden_size = hidden_size
        if self.method == "general":
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        if self.method == "concat":
            self.attn = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.v = nn.Parameter(torch.rand(self.hidden_size))
            stdv = 1.0 / math.sqrt(self.v.size(0))
            self.v.data.normal_(mean=0, std=stdv)

    def dot_score(self, hidden, encoder_output):
        attn_energies = torch.sum(hidden * encoder_output, dim=2)
        return attn_energies

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        attn_energies = torch.sum(hidden * energy, dim=2)
        return attn_energies

    def concat_score(self, hidden, encoder_output):
        energy = torch.tanh(
            self.attn(
                torch.cat(
                    (hidden.expand(encoder_output.shape[0], -1, -1), encoder_output), 2
                )
            )
        )
        return torch.sum(self.v * energy, dim=2)

    # calculate the attention weights (energies) based on the given method
    def forward(self, hidden, encoder_outputs):
        if self.method == "dot":
            attn_energies = self.dot_score(hidden, encoder_outputs)
        elif self.method == "general":
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            attn_energies = self.concat_score(hidden, encoder_outputs)
        attn_energies = attn_energies.t()
        attn_weights = torch.softmax(attn_energies, dim=1).unsqueeze(1)
        return attn_weights


#########################################################################################
#  building the S2S LA model
class S2S_LA_Model(nn.Module):
    """
    S2S model with Luong Attention module
    (https://machinelearningmastery.com/the-luong-attention-mechanism/).
    `attn` module is part of `S2S_LA_Model`
    """

    def __init__(self, cell_type, attn_method, input_size, hidden_size, use_cuda):
        super(S2S_LA_Model, self).__init__()
        self.cell_type = cell_type
        self.attn_method = attn_method
        self.input_size = input_size
        self.hidden_size = hidden_size

        if self.cell_type == "rnn":
            self.Ecell = nn.RNNCell(self.input_size, self.hidden_size)
            self.Dcell = nn.RNNCell(1, self.hidden_size)
        if self.cell_type == "gru":
            self.Ecell = nn.GRUCell(self.input_size, self.hidden_size)
            self.Dcell = nn.GRUCell(1, self.hidden_size)
        if self.cell_type == "lstm":
            self.Ecell = nn.LSTMCell(self.input_size, self.hidden_size)
            self.Dcell = nn.LSTMCell(1, self.hidden_size)

        self.lin_usage = nn.Linear(hidden_size, 1)
        self.lin_concat = nn.Linear(hidden_size * 2, hidden_size)
        self.attn = Attn(self.attn_method, self.hidden_size)
        self.use_cuda = use_cuda
        self.init()

    # function to intialize weight parameters
    def init(self):
        if self.cell_type == "rnn" or self.cell_type == "gru":
            for p in self.parameters():
                if p.dim() > 1:
                    init.orthogonal_(p.data, gain=1.0)
                if p.dim() == 1:
                    init.constant_(p.data, 0.0)
        elif self.cell_type == "lstm":
            for p in self.parameters():
                if p.dim() > 1:
                    init.orthogonal_(p.data, gain=1.0)
                if p.dim() == 1:
                    init.constant_(p.data, 0.0)
                    init.constant_(p.data[self.hidden_size : 2 * self.hidden_size], 1.0)

    def consume(self, x):
        if self.cell_type == "rnn" or self.cell_type == "gru":
            # encoder forward function
            h = torch.zeros(x.shape[0], self.hidden_size)
            encoder_outputs = torch.zeros(x.shape[1], x.shape[0], self.hidden_size)
            if self.use_cuda:
                h = h.cuda()
                encoder_outputs = encoder_outputs.cuda()
            # encoder part
            for T in range(x.shape[1]):
                h = self.Ecell(x[:, T, :], h)
                encoder_outputs[T] = h
            pred_usage = self.lin_usage(h)
        elif self.cell_type == "lstm":
            h0 = torch.zeros(x.shape[0], self.hidden_size)
            c0 = torch.zeros(x.shape[0], self.hidden_size)
            encoder_outputs = torch.zeros(x.shape[1], x.shape[0], self.hidden_size)
            if self.use_cuda:
                h0 = h0.cuda()
                c0 = c0.cuda()
                encoder_outputs = encoder_outputs.cuda()
            h = (h0, c0)
            for T in range(x.shape[1]):
                h = self.Ecell(x[:, T, :], h)
                encoder_outputs[T] = h[0]
            pred_usage = self.lin_usage(h[0])
        return pred_usage, h, encoder_outputs

    def predict(self, pred_usage, h, encoder_outputs, target_length):
        # decoder with attention function
        preds = []
        if self.cell_type == "rnn" or self.cell_type == "gru":
            for step in range(target_length):
                h = self.Dcell(pred_usage, h)
                attn_weights = self.attn(h, encoder_outputs)
                context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
                context = context.squeeze(1)
                concat_input = torch.cat((h, context), 1)
                concat_output = torch.tanh(self.lin_concat(concat_input))
                pred_usage = self.lin_usage(concat_output)
                preds.append(pred_usage.unsqueeze(1))
            preds = torch.cat(preds, 1)
        elif self.cell_type == "lstm":
            for step in range(target_length):
                h = self.Dcell(pred_usage, h)
                attn_weights = self.attn(h[0], encoder_outputs)
                context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
                context = context.squeeze(1)
                concat_input = torch.cat((h[0], context), 1)
                concat_output = torch.tanh(self.lin_concat(concat_input))
                pred_usage = self.lin_usage(concat_output)
                preds.append(pred_usage.unsqueeze(1))
            preds = torch.cat(preds, 1)
        return preds


#########################################################################################
class CharlieModel:
    def __init__(self, configs):
        self.configs = configs
        self.file_prefix = Path(configs["exp_dir"])
        self.file_prefix.mkdir(parents=True, exist_ok=True)

    def main(self, train_df, val_df):  # noqa: C901 TODO: remove noqa
        """
        process the data into three-dimensional for S2S model, train the model, and test the restuls
        """
        window_target_size = self.configs["data_processing"]["S2S_window"][
            "window_width_target"
        ]
        hidden_size = self.configs["learning_algorithm"]["hidden_size"]
        cell_type = "lstm"
        la_method = "none"
        attention_model = "BA"
        cuda = False
        epochs = self.configs["learning_algorithm"]["num_epochs"]
        batch_size = self.configs["learning_algorithm"]["train_batch_size"]
        loss_function_qs = self.configs["learning_algorithm"]["quantiles"]
        save_model = True
        seed = self.configs["data_processing"]["random_seed"]
        resample_interval = self.configs["data_processing"]["resample_interval"]
        window_target_size_count = int(
            pd.Timedelta(window_target_size) / pd.Timedelta(resample_interval)
        )

        t0 = time.time()
        np.random.seed(seed)
        torch.manual_seed(seed)

        # loss function, qs here is an integer, not a list of integer
        def quantile_loss(output, target, qs, window_target_size):
            """
            loss function with quntile number as parameter.
            For now, it cannot support list of quantiles as parameters.

            :param output: (Tensor)
            :param target: (Tensor)
            :param qs: (int)
            :param window_target_size: (int)
            :return: (Tensor) Loss for this study (single number)
            """

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            resid = target - output
            tau = torch.tensor([qs], device=device).repeat_interleave(
                window_target_size
            )

            alpha = 0.001
            log_term = torch.zeros_like(resid, device=device)
            log_term[resid < 0] = torch.log(1 + torch.exp(resid[resid < 0] / alpha)) - (
                resid[resid < 0] / alpha
            )
            log_term[resid >= 0] = torch.log(1 + torch.exp(-resid[resid >= 0] / alpha))
            loss = resid * tau + alpha * log_term
            loss = torch.mean(torch.mean(loss, 0))

            return loss

        train_df_predictor = train_df["predictor"].astype(np.float32).copy()
        train_df_target = train_df["target"].astype(np.float32).copy()
        val_df_predictor = val_df["predictor"].astype(np.float32).copy()
        val_df_target = val_df["target"].astype(np.float32).copy()

        input_dim = self.configs["input_dim"] = train_df_predictor.shape[-1]

        # call the respective model
        if attention_model == "none":
            model = S2S_Model(cell_type, input_dim, hidden_size, use_cuda=cuda)

        elif attention_model == "BA":
            model = S2S_BA_Model(cell_type, input_dim, hidden_size, use_cuda=cuda)

        elif attention_model == "LA":
            model = S2S_LA_Model(
                cell_type, la_method, input_dim, hidden_size, use_cuda=cuda
            )

        if cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model.cuda()

        print("MODEL ARCHITECTURE IS: ")
        print(model)

        print(
            "\nModel parameters are on cuda: {}".format(
                next(model.parameters()).is_cuda
            )
        )

        opt = optim.Adam(model.parameters(), lr=1e-3)
        #     loss_fn = nn.MSELoss(reduction='sum')
        loss_fn = quantile_loss

        EPOCHES = epochs
        BATCH_SIZE = batch_size

        print("\nStarting training...")

        train_loss = []
        test_loss = []

        for epoch in range(EPOCHES):
            t_one_epoch = time.time()
            print("Epoch {}".format(epoch + 1))

            # training
            total_usage_loss = 0
            for b_idx in range(0, train_df_predictor.shape[0], BATCH_SIZE):

                x = torch.from_numpy(
                    train_df_predictor[b_idx : b_idx + BATCH_SIZE]
                ).float()
                y = torch.from_numpy(
                    train_df_target[b_idx : b_idx + BATCH_SIZE]
                ).float()

                if cuda:
                    x = x.cuda()
                    y = y.cuda()

                # encoder forward, for respective models (with and without attention)
                if attention_model == "none":
                    y_pred, h = model.consume(x)

                elif attention_model == "BA":
                    y_pred, h, encoder_outputs = model.consume(x)

                elif attention_model == "LA":
                    y_pred, h, encoder_outputs = model.consume(x)

                # decoder forward, for respective models
                if attention_model == "none":
                    pred = model.predict(y_pred, h, window_target_size_count)

                elif attention_model == "BA":
                    pred = model.predict(
                        y_pred, h, encoder_outputs, window_target_size_count
                    )

                elif attention_model == "LA":
                    pred = model.predict(
                        y_pred, h, encoder_outputs, window_target_size_count
                    )

                # compute lose
                loss_usage = loss_fn(
                    pred,
                    y,
                    qs=loss_function_qs,
                    window_target_size=window_target_size_count,
                )

                # backprop and update
                opt.zero_grad()

                loss_usage.sum().backward()

                opt.step()

                total_usage_loss += loss_usage.item()

            train_loss.append(total_usage_loss)
            print("\tTRAINING: {} total train USAGE loss.\n".format(total_usage_loss))

            # testing
            y = None
            y_pred = None
            pred = None
            total_usage_loss = 0
            all_preds = []
            for b_idx in range(0, val_df_predictor.shape[0], BATCH_SIZE):
                with torch.no_grad():
                    x = torch.from_numpy(
                        val_df_predictor[b_idx : b_idx + BATCH_SIZE]
                    ).float()
                    y = torch.from_numpy(val_df_target[b_idx : b_idx + BATCH_SIZE])

                    if cuda:
                        x = x.cuda()
                        y = y.cuda()

                    # encoder forward, for respective models
                    if attention_model == "none":
                        y_pred, h = model.consume(x)

                    elif attention_model == "BA":
                        y_pred, h, encoder_outputs = model.consume(x)

                    elif attention_model == "LA":
                        y_pred, h, encoder_outputs = model.consume(x)

                    # decoder forward, for respective models
                    if attention_model == "none":
                        pred = model.predict(y_pred, h, window_target_size_count)

                    elif attention_model == "BA":
                        pred = model.predict(
                            y_pred, h, encoder_outputs, window_target_size_count
                        )

                    elif attention_model == "LA":
                        pred = model.predict(
                            y_pred, h, encoder_outputs, window_target_size_count
                        )

                    # compute loss
                    loss_usage = loss_fn(
                        pred,
                        y,
                        qs=loss_function_qs,
                        window_target_size=window_target_size_count,
                    )

                    if epoch == epochs - 1:
                        all_preds.append(pred)

                    total_usage_loss += loss_usage.item()

            test_loss.append(total_usage_loss)

            print("\tTESTING: {} total test USAGE loss".format(total_usage_loss))
            print("\tTESTING:\n")
            print("\tSample of prediction:")
            print("\t\t TARGET: {}".format(y[-1].cpu().detach().numpy().flatten()))
            print(
                "\t\t   PRED: {}\n\n".format(pred[-1].cpu().detach().numpy().flatten())
            )

            t2_one_epoch = time.time()
            time_one_epoch = t2_one_epoch - t_one_epoch
            print(
                "TIME OF ONE EPOCH: {} seconds and {} minutes".format(
                    time_one_epoch, time_one_epoch / 60.0
                )
            )

        # saving model
        if save_model:
            torch.save(
                model.state_dict(),
                "{}/torch_model".format(self.configs["exp_dir"]),
            )

        # saving results
        predictions = pd.DataFrame(all_preds[0].numpy().squeeze())
        predictions = predictions.add_prefix("{}_".format(str(loss_function_qs)))
        targets = pd.DataFrame(val_df_target.squeeze())
        predictions.to_hdf(
            os.path.join(self.file_prefix, "predictions.h5"), key="df", mode="w"
        )
        targets.to_hdf(
            os.path.join(self.file_prefix, "measured.h5"), key="df", mode="w"
        )

        # total time of run
        t1 = time.time()
        total = t1 - t0
        print("\nTIME ELAPSED: {} seconds OR {} minutes".format(total, total / 60.0))
        print("\nEnd of run")
