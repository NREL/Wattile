import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import n_colors
from plotly.subplots import make_subplots

"""
Common settings for graphics
"""
font_family = "Raleway, PT Sans Narrow, Overpass, Arial"
width_graphic = 1500
color_outer = "rgb(240,240,240)"
color_inner = "rgb(216,179,101)"
color_actual = "rgb(0,60,48)"
color_background = "rgb(255,255,255)"


def timeseries_comparison(configs, time_ahead):  # noqa: C901 TODO: remove noqa

    """
    read measured and prediction files
    """
    predictions = pd.read_hdf(configs["data_output"]["exp_dir"] + "/predictions.h5")
    measured = pd.read_hdf(configs["data_output"]["exp_dir"] + "/measured.h5")

    """
    read configs parameters
    """
    window_width_target = configs["data_processing"]["input_output_window"][
        "window_width_target"
    ]
    bin_interval = configs["data_processing"]["resample"]["bin_interval"]
    initial_num = (pd.Timedelta(window_width_target) // pd.Timedelta(bin_interval)) + 1
    use_case = configs["learning_algorithm"]["use_case"]
    quantiles = configs["learning_algorithm"]["quantiles"]

    """
    set colors
    """
    len_predictions = len(predictions.columns)
    len_colors = int((len(quantiles) - 1) / 2)
    if len_colors == 1:
        list_color = [color_outer]
    else:
        list_color = n_colors(color_outer, color_inner, len_colors, colortype="rgb")

    """
    initialize y-axis boundaries for finding min and max
    """
    y_max = 0
    y_min = 99999999999999

    """
    check timestamp order
    """
    timestamporder = (
        "timestamp in sequence" if use_case == "validation" else "timestamp shuffled"
    )

    """
    filter data based on portion defined in configs
    """
    plot_comparison_portion_start = configs["data_output"][
        "plot_comparison_portion_start"
    ]
    plot_comparison_portion_end = configs["data_output"]["plot_comparison_portion_end"]
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

    """
    initialize plot
    """
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.5, 0.15, 0.15, 0.15],
    )

    """
    extract median prediction and, if necessary, plot prediction quantiles/errors
    """
    if len(quantiles) == 1:

        # extract median prediction
        prediction_median = predictions.iloc[:, 0].values

        # find min/max y-values
        if min(prediction_median.min(), measured.iloc[:, 0].min()) < y_min:
            y_min = min(prediction_median.min(), measured.iloc[:, 0].min())
        if max(prediction_median.max(), measured.iloc[:, 0].max()) > y_max:
            y_max = max(prediction_median.max(), measured.iloc[:, 0].max())

    else:

        # extract median prediction
        if configs["learning_algorithm"]["arch_version"] == "alfa":

            idx_col_med = len_colors
            prediction_median = predictions.iloc[:, idx_col_med].values
            time_ahead = 0

        elif configs["learning_algorithm"]["arch_version"] == "bravo":

            idx_col_med_start = len_colors * initial_num
            idx_col_med_end = idx_col_med_start + initial_num
            prediction_median = (
                predictions.iloc[:, idx_col_med_start:idx_col_med_end]
                .iloc[:, time_ahead]
                .values
            )

        elif configs["learning_algorithm"]["arch_version"] == "charlie":

            window_target_size_count = window_target_size_count = int(
                pd.Timedelta(window_width_target) / pd.Timedelta(bin_interval)
            )
            idx_col_med_start = len_colors * window_target_size_count
            idx_col_med_end = idx_col_med_start + window_target_size_count
            prediction_median = (
                predictions.iloc[:, idx_col_med_start:idx_col_med_end]
                .iloc[:, time_ahead]
                .values
            )

        # extract quantile predictions
        for qntl in range(0, int(len_colors)):

            # filter data for correct quantile
            if configs["learning_algorithm"]["arch_version"] == "alfa":

                idx_col_low_start = qntl
                idx_col_high_start = len_predictions - idx_col_low_start - 1
                prediction_qntl_low = predictions.iloc[:, idx_col_low_start].to_numpy()
                prediction_qntl_high = predictions.iloc[
                    :, idx_col_high_start
                ].to_numpy()

            elif configs["learning_algorithm"]["arch_version"] == "bravo":

                idx_col_low_start = qntl * initial_num
                idx_col_low_end = qntl * initial_num + initial_num
                idx_col_high_start = (len_predictions - idx_col_low_start) - initial_num
                idx_col_high_end = len_predictions - idx_col_low_start
                prediction_qntl_low = predictions.iloc[
                    :, idx_col_low_start:idx_col_low_end
                ].to_numpy()
                prediction_qntl_high = predictions.iloc[
                    :, idx_col_high_start:idx_col_high_end
                ].to_numpy()

            elif configs["learning_algorithm"]["arch_version"] == "charlie":

                window_target_size_count = int(
                    pd.Timedelta(window_width_target) / pd.Timedelta(bin_interval)
                )
                idx_col_low_start = qntl * window_target_size_count
                idx_col_low_end = (
                    qntl * window_target_size_count + window_target_size_count
                )
                idx_col_high_start = (
                    len_predictions - idx_col_low_start
                ) - window_target_size_count
                idx_col_high_end = len_predictions - idx_col_low_start
                prediction_qntl_low = predictions.iloc[
                    :, idx_col_low_start:idx_col_low_end
                ].to_numpy()
                prediction_qntl_high = predictions.iloc[
                    :, idx_col_high_start:idx_col_high_end
                ].to_numpy()

            # calculate Interval Score details
            alph = 1 - (quantiles[-(qntl + 1)] - quantiles[qntl])
            IS = (
                (prediction_qntl_high - prediction_qntl_low)
                + (2 / alph)
                * (prediction_qntl_low - measured.to_numpy())
                * (measured.to_numpy() < prediction_qntl_low)
                + (2 / alph)
                * (measured.to_numpy() - prediction_qntl_high)
                * (measured.to_numpy() > prediction_qntl_high)
            )  # (batches * time) for a single quantile
            prediction_qntl_low = pd.DataFrame(prediction_qntl_low)
            prediction_qntl_high = pd.DataFrame(prediction_qntl_high)
            df_is = pd.DataFrame(IS)

            # calculate upper/lower quantile violations
            df_viol_l = (prediction_qntl_low[time_ahead] - measured[time_ahead]) * (
                prediction_qntl_low[time_ahead] > measured[time_ahead]
            )
            df_viol_u = (measured[time_ahead] - prediction_qntl_high[time_ahead]) * (
                measured[time_ahead] > prediction_qntl_high[time_ahead]
            )

            # find min/max y values
            if prediction_qntl_low[time_ahead].min() < y_min:
                y_min = prediction_qntl_low[time_ahead].min()
            if prediction_qntl_high[time_ahead].max() > y_max:
                y_max = prediction_qntl_high[time_ahead].max()

            # plot prediction quantile lower bound
            fig.add_trace(
                go.Scatter(
                    y=prediction_qntl_low[time_ahead],
                    mode="lines",
                    name="Quantile {}".format(
                        str(quantiles[qntl]) + "-" + str(quantiles[-qntl - 1])
                    ),
                    legendgroup="Quantile {}".format(
                        str(quantiles[qntl]) + "-" + str(quantiles[-qntl - 1])
                    ),
                    showlegend=False,
                    xaxis="x",
                    opacity=0.1,
                    line=dict(color=list_color[qntl], width=1),
                ),
                row=1,
                col=1,
            )

            # plot prediction quantile higher bound
            fig.add_trace(
                go.Scatter(
                    y=prediction_qntl_high[time_ahead],
                    mode="lines",
                    name="Quantile {}".format(
                        str(quantiles[qntl]) + "-" + str(quantiles[-qntl - 1])
                    ),
                    legendgroup="Quantile {}".format(
                        str(quantiles[qntl]) + "-" + str(quantiles[-qntl - 1])
                    ),
                    showlegend=True,
                    xaxis="x",
                    fill="tonexty",
                    opacity=0.1,
                    line=dict(color=list_color[qntl], width=1),
                ),
                row=1,
                col=1,
            )

            # plot prediction quantile higher bound violation
            fig.add_trace(
                go.Scatter(
                    y=df_viol_l,
                    mode="lines",
                    name="Quantile {}".format(
                        str(quantiles[qntl]) + "-" + str(quantiles[-qntl - 1])
                    ),
                    legendgroup="Quantile {}".format(
                        str(quantiles[qntl]) + "-" + str(quantiles[-qntl - 1])
                    ),
                    line_color=list_color[qntl],
                    showlegend=False,
                    xaxis="x",
                    yaxis="y1",
                ),
                row=2,
                col=1,
            )

            # plot prediction quantile higher bound violation
            fig.add_trace(
                go.Scatter(
                    y=df_viol_u,
                    mode="lines",
                    name="Quantile {}".format(
                        str(quantiles[qntl]) + "-" + str(quantiles[-qntl - 1])
                    ),
                    legendgroup="Quantile {}".format(
                        str(quantiles[qntl]) + "-" + str(quantiles[-qntl - 1])
                    ),
                    line_color=list_color[qntl],
                    showlegend=False,
                    xaxis="x",
                    yaxis="y1",
                ),
                row=3,
                col=1,
            )

            # plot Interval Score
            fig.add_trace(
                go.Scatter(
                    y=df_is[time_ahead],
                    mode="lines",
                    name="Quantile {}".format(
                        str(quantiles[qntl]) + "-" + str(quantiles[-qntl - 1])
                    ),
                    legendgroup="Quantile {}".format(
                        str(quantiles[qntl]) + "-" + str(quantiles[-qntl - 1])
                    ),
                    line_color=list_color[qntl],
                    showlegend=False,
                    xaxis="x",
                ),
                row=4,
                col=1,
            )

    """
    plot median prediction
    """
    fig.add_trace(
        go.Scatter(
            y=prediction_median,
            mode="lines",
            name="Median Prediction",
            legendgroup="Median Prediction",
            line=dict(color=color_inner, width=2, dash="dot"),
            xaxis="x",
        ),
        row=1,
        col=1,
    )

    """
    plot measured target
    """
    fig.add_trace(
        go.Scatter(
            y=measured[time_ahead],
            mode="lines",
            name="Target",
            legendgroup="Target",
            line=dict(color=color_actual, width=0.5),
            xaxis="x",
        ),
        row=1,
        col=1,
    )

    """
    configure plot
    """
    fig.update_layout(
        width=width_graphic,
        height=500,
        margin=dict(
            l=30,
            r=0,
            t=30,
            b=0,
        ),
        plot_bgcolor=color_background,
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
            text="<b>Time</b> ({})".format(timestamporder),
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

    """
    save plot image
    """
    timeseries_comparison = (
        configs["data_output"]["exp_dir"] + "/Vis_TimeseriesComparisons.svg"
    )
    print("saving timeseries comparison in {}".format(timeseries_comparison))
    pio.write_image(fig, timeseries_comparison)
    fig.write_html(
        configs["data_output"]["exp_dir"] + "/Vis_TimeseriesComparisons.html"
    )
