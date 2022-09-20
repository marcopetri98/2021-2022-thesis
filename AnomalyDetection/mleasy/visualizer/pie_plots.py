from numbers import Number
from typing import Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np

from mleasy.input_validation import check_argument_types
from mleasy.input_validation import is_matplotlib_color


def pie_plot(wedges,
             labels: list[str] = None,
             colors: list = None,
             radius: float = 1.0,
             percentage_fmt: str = None,
             percentage_dst: float = 0.6,
             title: str = "",
             fig_size: Tuple = (8, 8)) -> None:
    """Draws a pie plot with the specified parameters.
    
    Parameters
    ----------
    wedges : array-like of shape (n_classes)
        The wedges to be used for the pie plot.
    
    labels : list[str], default=None
        The labels of the classes.
    
    colors : list, default=None
        The color of the classes. The colors are the colors accepted by
        matplotlib.
        
    radius : float, default=1.0
        The radius of the circle of the pie plot.
        
    percentage_fmt : str, default=None
        The format in which to show the percentage of the wedges. If None, the
        percentages are not shown.
        
    percentage_dst : float, default=0.6
        The distance from the centre of the percentages, if present.
        
    title : str, default=""
        The title of the bar plot.
    
    fig_size : tuple, default=(8,8)
        The dimension of the figure to prompt.

    Returns
    -------
    None
    """
    wedges = np.array(wedges)
    
    check_argument_types([labels, colors, radius, percentage_fmt, percentage_dst, title, fig_size],
                         [[list, None], [list, None], Number, [str, Callable, None], float, str, tuple],
                         ["labels", "colors", "radius", "percentage_fmt", "percentage_dst", "title", "fig_size"])

    # check variable types
    if colors is not None and not is_matplotlib_color(colors):
        raise TypeError("colors must be a valid matplotlib color")
        
    # check values
    if wedges.ndim != 1:
        raise ValueError("wedges must have one dimension")
    elif labels is not None and len(labels) != wedges.shape[0]:
        raise ValueError("labels and wedges must have the same dimension")
    elif isinstance(colors, list) and len(colors) != wedges.shape[0]:
        raise ValueError("colors must have the same length of wedges")
    
    # implementation
    fig = plt.Figure(figsize=fig_size, tight_layout=True)
    
    plt.pie(wedges,
            labels=labels,
            colors=colors,
            radius=radius,
            autopct=percentage_fmt,
            pctdistance=percentage_dst)
    plt.title(title)
    
    plt.show()

# FIXME: too similar to pie_plot, evaluate its removal
def pie_class_distribution(classes,
                           labels: list[str] = None,
                           colors: list = None,
                           radius: float = 1.0,
                           percentage_fmt: str = None,
                           percentage_dst: float = 0.6,
                           title: str = "",
                           fig_size: Tuple = (8, 8)) -> None:
    """Draws a pie plot from the frequencies of classes.
    
    Parameters
    ----------
    classes : array-like of shape (n_classes)
        An array having the dimension equal to the number of classes in which
        the elements are the number of examples for each class.
        
    labels : list[str], default=None
        The labels of the classes.
    
    colors : list, default=None
        The color of the classes. The colors are the colors accepted by
        matplotlib.
        
    radius : float, default=1.0
        The radius of the circle of the pie plot.
        
    percentage_fmt : str, default=None
        The format in which to show the percentage of the wedges. If None, the
        percentages are not shown.
        
    percentage_dst : float, default=0.6
        The distance from the centre of the percentages, if present.
        
    title : str, default=""
        The title of the bar plot.
    
    fig_size : tuple, default=(8,8)
        The dimension of the figure to prompt.

    Returns
    -------
    None
    """
    classes = np.array(classes)
    
    check_argument_types([labels, colors, radius, percentage_fmt, percentage_dst, title, fig_size],
                         [[list, None], [list, None], Number, [str, Callable, None], float, str, tuple],
                         ["labels", "colors", "radius", "percentage_fmt", "percentage_dst", "title", "fig_size"])

    # check variable types
    if colors is not None and not is_matplotlib_color(colors):
        raise TypeError("colors must be a valid matplotlib color")
    
    # check values
    if classes.ndim != 1:
        raise ValueError("classes must have only one dimension")
    elif isinstance(labels, list) and len(labels) != classes.shape[0]:
        raise ValueError("labels must have the same length of classes")
    elif isinstance(colors, list) and len(colors) != classes.shape[0]:
        raise ValueError("colors must have the same length of classes")
    
    # implementation
    total = np.sum(classes)
    wedges = classes / total
    
    pie_plot(wedges=wedges,
             labels=labels,
             colors=colors,
             radius=radius,
             percentage_fmt=percentage_fmt,
             percentage_dst=percentage_dst,
             title=title,
             fig_size=fig_size)
