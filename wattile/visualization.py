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


def timeseries_comparison(configs):
    predictions = pd.read_hdf(configs["exp_dir"] + "/predictions.h5")
    measured = pd.read_hdf(configs["exp_dir"] + "/measured.h5")

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

    len_tot = len(predictions.columns)
    target = measured.to_numpy()

    # color
    c_target = "rgb(228,26,28)"
    len_c_list = int((len(configs["qs"]) - 1) / 2)
    c_list = n_colors(color_base, color_accent, len_c_list, colortype="rgb")
    list_c_low_penalty = c_list.copy()
    list_c_high_penalty = c_list.copy()
    list_c_is = c_list.copy()
    list_c_is.reverse()

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.5, 0.15, 0.15, 0.15],
    )

    y_min = 10000000
    y_max = 0
    for ahd in range(len(measured.columns)):

        if ahd == 0:

            count_qntl = 0

            for qntl in range(0, int((len(configs["qs"]) - 1) / 2)):

                if (configs["arch_version"] == "bravo") | (
                    configs["arch_version"] == "charlie"
                ):

                    low_start = (
                        qntl * configs["data_processing"]["S2S_stagger"]["initial_num"]
                    )
                    low_end = (
                        qntl * configs["data_processing"]["S2S_stagger"]["initial_num"]
                        + configs["data_processing"]["S2S_stagger"]["initial_num"]
                    )
                    high_start = (len_tot - low_start) - configs["data_processing"][
                        "S2S_stagger"
                    ]["initial_num"]
                    high_end = len_tot - low_start

                    low = predictions.iloc[:, low_start:low_end].to_numpy()
                    high = predictions.iloc[:, high_start:high_end].to_numpy()

                elif configs["arch_version"] == "alfa":

                    low_start = qntl
                    high_start = len_tot - low_start - 1

                    low = predictions.iloc[:, low_start].to_numpy()
                    high = predictions.iloc[:, high_start].to_numpy()

                alph = 1 - (configs["qs"][-(qntl + 1)] - configs["qs"][qntl])
                IS = (
                    (high - low)
                    + (2 / alph) * (low - target) * (target < low)
                    + (2 / alph) * (target - high) * (target > high)
                )  # (batches * time) for a single quantile

                df_low = pd.DataFrame(low)
                df_high = pd.DataFrame(high)
                df_x = pd.DataFrame(target)
                df_is = pd.DataFrame(IS)

                df_viol_l = (df_low[ahd] - df_x[ahd]) * (df_low[ahd] > df_x[ahd])
                df_viol_u = (df_x[ahd] - df_high[ahd]) * (df_x[ahd] > df_high[ahd])

                if df_low[ahd].min() < y_min:
                    y_min = df_low[ahd].min()

                if df_high[ahd].max() > y_max:
                    y_max = df_high[ahd].max()

                fig.add_trace(
                    go.Scatter(
                        y=df_low[ahd],
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

                fig.add_trace(
                    go.Scatter(
                        y=df_high[ahd],
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

                fig.add_trace(
                    go.Scatter(
                        y=df_is[ahd],
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

            fig.add_trace(
                go.Scatter(
                    y=df_x[ahd],
                    mode="lines",
                    name="Target",
                    legendgroup="Target",
                    line=dict(color=c_target, width=1),
                    xaxis="x",
                ),
                row=1,
                col=1,
            )

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

    timeseries_comparison = configs["exp_dir"] + "/Vis_TimeseriesComparisons.svg"
    print("saving timeseries comparison in {}".format(timeseries_comparison))
    pio.write_image(fig, timeseries_comparison)
    fig.write_html(configs["exp_dir"] + "/Vis_TimeseriesComparisons.html")
