from numbers import Number
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from mleasy.input_validation import check_argument_types
from mleasy.input_validation import is_matplotlib_color


def line_plot(x,
              y,
              x_ticks_loc=None,
              x_ticks_labels=None,
              x_ticks_rotation: float = 0,
              formats: list[str] | str = None,
              colors: list = None,
              series_labels: list[str] | str = None,
              title: str = "",
              y_axis_label: str = "",
              x_axis_label: str = "",
              plot_legend: bool = True,
              fig_size: Tuple = (8, 8),
              ax: Axes = None) -> None:
    """Creates a line plot from data.

    Parameters
    ----------
    x : array-like or list of array-like
        The data representing the independent variable to be used to create the
        line plot. It can be an array-like when only one line should be drawn,
        and a list of array-like in case multiple lines should be drawn. This
        arrays must contain numbers.

    y : array-like or list of array-like
        The data representing the dependent variable to be used to create the
        line plot. It can be an array-like when only one line should be drawn,
        and a list of array-like in case multiple lines should be drawn. All the
        array-like contained in this variable must have the same shape of the
        array-like contained in the `x` argument.

    x_ticks_loc : array-like, default=None
        The location at which printing the ticks labels. These will be also the
        labels in case the argument `x_ticks_labels` is None.

    x_ticks_labels : array-like, default=None
        The labels of the ticks on the x to be printed on the plot, they start
        at the first sample and end at the last sample if `x_ticks_loc` is None.
        Otherwise, they will be printed exactly at the position specified by the
        other argument.
        
    x_ticks_rotation : float, default=0.0
        The rotation of the ticks on the x-axis.

    formats : list[str] or str, default=None
        The formats for the lines to be drawn on the plot.

    colors : list[color] or color, default=None
        The colors of the lines to be drawn.

    series_labels: list[str] or str, default=None
        The labels of the lines to plot.

    title : str, default=""
        The title of the plot.

    y_axis_label : str, default=""
        The label to print on the y-axis.

    x_axis_label : str, default=""
        The label to print on the x-axis.

    plot_legend : bool, default=True
        States if the legend must be plot on the line plot.

    fig_size : tuple, default=(8,8)
        The dimension of the matplotlib figure to be drawn.
        
    ax : Axes, default=None
        The axis on which to add the plot. If this is not None, the plot will be
        added to the axes, no new figure will be created and printed.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        At least one of the arguments has been passed with the wrong type.

    ValueError
        At least one variable has unacceptable value or inconsistent value.
    """
    if isinstance(x, list):
        check_argument_types([y], [list], ["y"])

        if len(x) != len(y):
            raise ValueError("x and y must have the same dimension")

        for i, couple in enumerate(zip(x, y)):
            var_dep, var_ind = couple
            var_dep = np.array(var_dep)
            var_ind = np.array(var_ind)
            x[i] = var_ind
            y[i] = var_dep

            if var_dep.shape != var_ind.shape:
                raise ValueError("the dependent and independent arrays must "
                                 "have the same shape")
            elif var_dep.ndim != 1:
                raise ValueError("when you pass a list of arrays, arrays must "
                                 "have 1 dimension")
    else:
        x = np.array(x)
        y = np.array(y)

        if x.shape != y.shape:
            raise ValueError("the dependent and independent arrays must have "
                             "the same shape")
        elif x.ndim != 1:
            raise ValueError("arrays have more than one dimension, if you want "
                             "to plot multiple lines, pass a list of 1d arrays")

    if x_ticks_loc is not None:
        x_ticks_loc = np.array(x_ticks_loc)

    if x_ticks_labels is not None:
        x_ticks_labels = np.array(x_ticks_labels)

    check_argument_types([x_ticks_rotation, formats, title, y_axis_label, x_axis_label, fig_size, ax],
                         [Number, [str, list, None], str, str, str, tuple, [Axes, None]],
                         ["x_ticks_rotation", "formats", "title", "y_axis_label", "x_axis_label", "fig_size", "ax"])

    if colors is not None and not isinstance(colors, list):
        colors = [colors] * (len(x) if isinstance(x, list) else x.shape[0])

    # check type
    if colors is not None and not is_matplotlib_color(colors):
        raise TypeError("bars_colors must be a valid matplotlib color")
    elif not isinstance(plot_legend, bool):
        raise TypeError("plot_legend must be bool")

    # check values
    if isinstance(x, list):
        if formats is not None and len(x) < len(formats):
            raise ValueError("the number of formats must be at most equal to "
                             "the number of lines")
        elif colors is not None and len(x) < len(colors):
            raise ValueError("the number of colors must be at most equal to "
                             "the number of lines")
        elif series_labels is not None and len(x) < len(series_labels):
            raise ValueError("the number of labels must be equal to the number "
                             "of lines")
    elif series_labels is not None and not isinstance(series_labels, str):
        raise TypeError("series_labels must be a string if only one line has to"
                        " be plotted")
    elif formats is not None and not isinstance(formats, str):
        raise TypeError("if only one line is passed, format must be None or "
                        "a single format, not a list")
    elif isinstance(colors, list):
        raise ValueError("if only one line is passed, colors must be None or "
                         "a single color, not a list")
    elif x_ticks_loc is not None and x_ticks_labels is not None:
        if x_ticks_loc.shape != x_ticks_labels.shape:
            raise ValueError("if both x_ticks_loc and x_ticks_labels are passed"
                             ", they must have the same shape")
        elif x_ticks_loc.ndim != 1:
            raise ValueError("x_ticks_loc and x_ticks_labels must have at most "
                             "1 dimension")
    elif x_ticks_loc is not None and x_ticks_loc.ndim != 1:
        raise ValueError("x_ticks_loc must have at most 1 dimension")
    elif x_ticks_labels is not None and x_ticks_labels.ndim != 1:
        raise ValueError("x_ticks_labels must have at most 1 dimension")

    # implementation
    if ax is None:
        fig = plt.figure(figsize=fig_size)

    # TODO: evaluate if this can be made top-level
    def add_line(ind, dep, line_color, line_format, axes, label):
        if label is not None:
            other_params = {"label": label}
        else:
            other_params = {}

        if axes is None:
            if line_format is not None:
                plt.plot(ind, dep, line_format, color=line_color, **other_params)
            else:
                plt.plot(ind, dep, color=line_color, **other_params)
        else:
            if line_format is not None:
                axes.plot(ind, dep, line_format, color=line_color, **other_params)
            else:
                axes.plot(ind, dep, color=line_color, **other_params)

    # TODO: evaluate if this can be made top-level
    def add_ticks(loc, label, rotation, axes):
        if axes is None:
            plt.xticks(loc, label, rotation=rotation)
        else:
            axes.set_xticks(loc, label, rotation=rotation)

    # plots all the lines
    if isinstance(x, list):
        for i, couple in enumerate(zip(x, y)):
            var_dep, var_ind = couple
            line_fmt = None
            line_col = None
            if formats is not None and i < len(formats):
                line_fmt = formats[i]
            if colors is not None and i < len(colors):
                line_col = colors[i]

            label = None if series_labels is None else series_labels[i]

            add_line(var_ind, var_dep, line_col, line_fmt, ax, label)
    else:
        add_line(x, y, colors, formats, ax, series_labels)

    # put ticks on the x if they are passed to the function
    if x_ticks_loc is not None or x_ticks_labels is not None:
        if x_ticks_loc is not None and x_ticks_labels is not None:
            # both are specified
            add_ticks(x_ticks_loc, x_ticks_labels, x_ticks_rotation, ax)
        elif x_ticks_loc is not None:
            # loc will also serve as label
            add_ticks(x_ticks_loc, x_ticks_loc, x_ticks_rotation, ax)
        else:
            # labels must go from the start to the end
            if isinstance(x, list):
                start = np.inf
                end = - np.inf
                for seq in x:
                    if np.min(seq) < start:
                        start = np.min(seq)
                    if np.max(seq) > end:
                        end = np.max(seq)
            else:
                start = np.min(x)
                end = np.max(x)
            x_ticks_loc = np.linspace(start, end, x_ticks_labels.shape[0])
            add_ticks(x_ticks_loc, x_ticks_labels, x_ticks_rotation, ax)

    if ax is None:
        plt.title(title)
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.tight_layout()
        if series_labels is not None and plot_legend:
            plt.legend()
    
        plt.show()
    else:
        ax.set_title(title)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        if series_labels is not None and plot_legend:
            ax.legend()
