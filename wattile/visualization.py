import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import n_colors
from plotly.subplots import make_subplots

#################################################
# Common settings for graphics
#################################################
font_family = "Raleway, PT Sans Narrow, Overpass, Arial"
width_graphic = 1500
color_base = "rgb(236,200,178)"
color_accent = "rgb(80,70,69)"


def timeseries_comparison(configs, time_ahead):  # noqa: C901 TODO: remove noqa

    ######################################################################
    # reading measured and prediction files
    ######################################################################
    predictions = pd.read_hdf(configs["exp_dir"] + "/predictions.h5")
    measured = pd.read_hdf(configs["exp_dir"] + "/measured.h5")

    ######################################################################
    # filter data based on portion defined in configs.json
    ######################################################################
    plot_comparison_portion_start = configs["plot_comparison_portion_start"]
    plot_comparison_portion_end = configs["plot_comparison_portion_end"]
    predictions = predictions.iloc[
        int(predictions.shape[0] * plot_comparison_portion_start) : int(
            predictions.shape[0] * plot_comparison_portion_end
        ),
        :,
    ]
    measured = measured.iloc[
        int(measured.shape[0] * plot_comparison_portion_start) : int(
            measured.shape[0] * plot_comparison_portion_end
        ),
        :,
    ]

    ######################################################################
    # plot setting
    ######################################################################
    len_tot = len(predictions.columns)
    target = measured.to_numpy()
    c_target = "rgb(228,26,28)"
    len_c_list = int((len(configs["learning_algorithm"]["quantiles"]) - 1) / 2)
    if len_c_list == 1:
        c_list = [color_base]
    else:
        c_list = n_colors(color_base, color_accent, len_c_list, colortype="rgb")
    list_c_low_penalty = c_list.copy()
    list_c_high_penalty = c_list.copy()
    list_c_is = c_list.copy()
    list_c_is.reverse()
    y_min = 10000000
    y_max = 0
    window_target_size = configs["data_processing"]["S2S_window"]["window_width_target"]
    resample_interval = configs["data_processing"]["resample_interval"]

    ######################################################################
    # initialize plotting area
    ######################################################################
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.5, 0.15, 0.15, 0.15],
    )

    ######################################################################
    # if quantiles were only defined with median
    ######################################################################
    if len(configs["learning_algorithm"]["quantiles"]) == 1:

        # -----------------------------------------------------------------
        # extracting median prediction
        # -----------------------------------------------------------------
        prediction_median = predictions.iloc[:, 0].values

        if min(prediction_median.min(), measured.iloc[:, 0].min()) < y_min:
            y_min = min(prediction_median.min(), measured.iloc[:, 0].min())

        if max(prediction_median.max(), measured.iloc[:, 0].max()) > y_max:
            y_max = max(prediction_median.max(), measured.iloc[:, 0].max())

    ######################################################################
    # if quantiles were defined properly
    ######################################################################
    else:

        # -----------------------------------------------------------------
        # extracting median prediction
        # -----------------------------------------------------------------
        if configs["learning_algorithm"]["arch_version"] == "alfa":

            idx_col_med = len_c_list
            prediction_median = predictions.iloc[:, idx_col_med].values
            time_ahead = 0

        elif configs["learning_algorithm"]["arch_version"] == "bravo":

            idx_col_med_start = (
                len_c_list * configs["data_processing"]["S2S_stagger"]["initial_num"]
            )
            idx_col_med_end = (
                idx_col_med_start
                + configs["data_processing"]["S2S_stagger"]["initial_num"]
            )
            prediction_median = (
                predictions.iloc[:, idx_col_med_start:idx_col_med_end]
                .iloc[:, time_ahead]
                .values
            )

        elif configs["learning_algorithm"]["arch_version"] == "charlie":

            window_target_size_count = window_target_size_count = int(
                pd.Timedelta(window_target_size) / pd.Timedelta(resample_interval)
            )
            idx_col_med_start = len_c_list * window_target_size_count
            idx_col_med_end = idx_col_med_start + window_target_size_count
            prediction_median = (
                predictions.iloc[:, idx_col_med_start:idx_col_med_end]
                .iloc[:, time_ahead]
                .values
            )

        # -----------------------------------------------------------------
        # extracting quantile predictions
        # -----------------------------------------------------------------
        count_qntl = 0
        for qntl in range(
            0, int((len(configs["learning_algorithm"]["quantiles"]) - 1) / 2)
        ):

            # -------------------------------------------------------------
            # filtering data for correct quantile
            # -------------------------------------------------------------
            if configs["learning_algorithm"]["arch_version"] == "alfa":

                idx_col_low_start = qntl
                idx_col_high_start = len_tot - idx_col_low_start - 1

                prediction_low = predictions.iloc[:, idx_col_low_start].to_numpy()
                prediction_high = predictions.iloc[:, idx_col_high_start].to_numpy()

            elif configs["learning_algorithm"]["arch_version"] == "bravo":

                idx_col_low_start = (
                    qntl * configs["data_processing"]["S2S_stagger"]["initial_num"]
                )
                idx_col_low_end = (
                    qntl * configs["data_processing"]["S2S_stagger"]["initial_num"]
                    + configs["data_processing"]["S2S_stagger"]["initial_num"]
                )
                idx_col_high_start = (len_tot - idx_col_low_start) - configs[
                    "data_processing"
                ]["S2S_stagger"]["initial_num"]
                idx_col_high_end = len_tot - idx_col_low_start

                prediction_low = predictions.iloc[
                    :, idx_col_low_start:idx_col_low_end
                ].to_numpy()
                prediction_high = predictions.iloc[
                    :, idx_col_high_start:idx_col_high_end
                ].to_numpy()

            elif configs["learning_algorithm"]["arch_version"] == "charlie":

                window_target_size_count = int(
                    pd.Timedelta(window_target_size) / pd.Timedelta(resample_interval)
                )

                idx_col_low_start = qntl * window_target_size_count
                idx_col_low_end = (
                    qntl * window_target_size_count + window_target_size_count
                )
                idx_col_high_start = (
                    len_tot - idx_col_low_start
                ) - window_target_size_count
                idx_col_high_end = len_tot - idx_col_low_start

                prediction_low = predictions.iloc[
                    :, idx_col_low_start:idx_col_low_end
                ].to_numpy()
                prediction_high = predictions.iloc[
                    :, idx_col_high_start:idx_col_high_end
                ].to_numpy()

            # -------------------------------------------------------------
            # calculating Interval Score details
            # -------------------------------------------------------------
            alph = 1 - (
                configs["learning_algorithm"]["quantiles"][-(qntl + 1)]
                - configs["learning_algorithm"]["quantiles"][qntl]
            )
            IS = (
                (prediction_high - prediction_low)
                + (2 / alph) * (prediction_low - target) * (target < prediction_low)
                + (2 / alph) * (target - prediction_high) * (target > prediction_high)
            )  # (batches * time) for a single quantile

            df_prediction_low = pd.DataFrame(prediction_low)
            df_prediction_high = pd.DataFrame(prediction_high)
            df_is = pd.DataFrame(IS)

            df_viol_l = (df_prediction_low[time_ahead] - measured[time_ahead]) * (
                df_prediction_low[time_ahead] > measured[time_ahead]
            )
            df_viol_u = (measured[time_ahead] - df_prediction_high[time_ahead]) * (
                measured[time_ahead] > df_prediction_high[time_ahead]
            )

            if df_prediction_low[time_ahead].min() < y_min:
                y_min = df_prediction_low[time_ahead].min()

            if df_prediction_high[time_ahead].max() > y_max:
                y_max = df_prediction_high[time_ahead].max()

            # -------------------------------------------------------------
            # plotting prediction lower bound
            # -------------------------------------------------------------
            fig.add_trace(
                go.Scatter(
                    y=df_prediction_low[time_ahead],
                    mode="lines",
                    name="Quantile{}".format(qntl),
                    legendgroup="Quantile{}".format(qntl),
                    showlegend=True,
                    xaxis="x",
                    opacity=0.1,
                    line=dict(color=c_list[qntl], width=0),
                ),
                row=1,
                col=1,
            )

            # -------------------------------------------------------------
            # plotting prediction higher bound
            # -------------------------------------------------------------
            fig.add_trace(
                go.Scatter(
                    y=df_prediction_high[time_ahead],
                    mode="lines",
                    name="Quantile{}".format(qntl),
                    legendgroup="Quantile{}".format(qntl),
                    showlegend=False,
                    xaxis="x",
                    fill="tonexty",
                    opacity=0.1,
                    line=dict(color=c_list[qntl], width=0),
                ),
                row=1,
                col=1,
            )

            # -------------------------------------------------------------
            # plotting prediction lower violation
            # -------------------------------------------------------------
            fig.add_trace(
                go.Scatter(
                    y=df_viol_l,
                    mode="lines",
                    name="Quantile{}".format(qntl),
                    legendgroup="Quantile{}".format(qntl),
                    line_color=list_c_low_penalty[count_qntl],
                    showlegend=False,
                    xaxis="x",
                    yaxis="y1",
                ),
                row=2,
                col=1,
            )

            # -------------------------------------------------------------
            # plotting prediction higher violation
            # -------------------------------------------------------------
            fig.add_trace(
                go.Scatter(
                    y=df_viol_u,
                    mode="lines",
                    name="Quantile{}".format(qntl),
                    legendgroup="Quantile{}".format(qntl),
                    line_color=list_c_high_penalty[count_qntl],
                    showlegend=False,
                    xaxis="x",
                    yaxis="y1",
                ),
                row=3,
                col=1,
            )

            # -------------------------------------------------------------
            # plotting Interval Score
            # -------------------------------------------------------------
            fig.add_trace(
                go.Scatter(
                    y=df_is[time_ahead],
                    mode="lines",
                    name="Quantile{}".format(qntl),
                    legendgroup="Quantile{}".format(qntl),
                    line_color=list_c_is[count_qntl],
                    showlegend=False,
                    xaxis="x",
                ),
                row=4,
                col=1,
            )

            count_qntl += 1

    ######################################################################
    # plotting median prediction
    ######################################################################
    fig.add_trace(
        go.Scatter(
            y=prediction_median,
            mode="lines",
            name="Median",
            legendgroup="Median",
            line=dict(color="black", width=1, dash="dot"),
            xaxis="x",
        ),
        row=1,
        col=1,
    )

    ######################################################################
    # plotting measured target
    ######################################################################
    fig.add_trace(
        go.Scatter(
            y=measured[time_ahead],
            mode="lines",
            name="Target",
            legendgroup="Target",
            line=dict(color=c_target, width=1),
            xaxis="x",
        ),
        row=1,
        col=1,
    )

    ######################################################################
    # plot setting
    ######################################################################
    fig.update_layout(
        width=width_graphic,
        height=500,
        margin=dict(
            l=30,
            r=0,
            t=30,
            b=0,
        ),
        plot_bgcolor="rgb(245,245,245)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="center",
            x=0.5,
            font=dict(
                size=12,
                family=font_family,
            ),
        ),
    )

    fig.update_xaxes(
        title=dict(
            text="<b>Appended Multiple Time Windows</b><br>(single window size = {})".format(
                configs["data_processing"]["sequential_splicer"]["window_width"]
            ),
            font=dict(
                size=14,
                family=font_family,
            ),
        ),
        showticklabels=False,
        row=4,
        col=1,
    )

    fig.update_yaxes(
        title=dict(
            text="<b>Power<br>Consumption<b>",
            font=dict(
                size=14,
                family=font_family,
            ),
        ),
        tickfont_family=font_family,
        range=[y_min * 0.95, y_max * 1.05],
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title=dict(
            text="<b>Penalty<br>Low</b>",
            font=dict(
                size=14,
                family=font_family,
            ),
        ),
        tickfont_family=font_family,
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title=dict(
            text="<b>Penalty<br>High</b>",
            font=dict(
                size=14,
                family=font_family,
            ),
        ),
        tickfont_family=font_family,
        row=3,
        col=1,
    )

    fig.update_yaxes(
        title=dict(
            text="<b>Interval<br>Score</b>",
            font=dict(
                size=14,
                family=font_family,
            ),
        ),
        tickfont_family=font_family,
        row=4,
        col=1,
    )

    fig.update_xaxes(
        showgrid=False,
    )

    fig.update_yaxes(
        showgrid=False,
    )

    ######################################################################
    # saving plot to file
    ######################################################################
    timeseries_comparison = configs["exp_dir"] + "/Vis_TimeseriesComparisons.svg"
    print("saving timeseries comparison in {}".format(timeseries_comparison))
    pio.write_image(fig, timeseries_comparison)
    fig.write_html(configs["exp_dir"] + "/Vis_TimeseriesComparisons.html")
