import string
import textwrap
import warnings
from typing import Union, Tuple, Literal, List, Callable, Dict
from pathlib import Path

from colorsys import hls_to_rgb, rgb_to_hls
from typing import Dict, Tuple, Union
from warnings import warn

import matplotlib as mpl
import matplotlib.axes as mpl_axes
import matplotlib.colors as colors
import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import cnames, to_hex, to_rgb
from matplotlib.legend_handler import (
    HandlerLine2D,
    HandlerPathCollection,
    HandlerPolyCollection,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


from sdm_eurec4a import replace_path_suffix

# Use colorblind-safe colors
_default_colors = [
    "#CC6677",
    "#6E9CB3",
    "#CA8727",
    "#44AA99",
    "#AA4499",
    "#D6BE49",
    "#A494F5",
]

_known_unit_keys_ = ["units", "unit", "Units", "Unit", "UNITS", "UNIT"]


def set_custom_rcParams() -> list:
    """
    Set the default configuration parameters for matplotlib. The colorblind-
    save colors were chosen with the help of
    https://davidmathlogic.com/colorblind.

    Returns:
    --------
    colors (np.ndarray) Array containing the default colors in HEX format

    Note:
    -----
    This function modifies the global matplotlib configuration.

    Examples:
    ---------
        >>> set_custom_rcParams()
    """

    # Set default figure size
    plt.rcParams["figure.figsize"] = (16 / 2, 9 / 2)

    # Set font sizes
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 15
    HUGHER_SIZE = 18
    plt.rc("font", size=MEDIUM_SIZE)  # Default text sizes
    plt.rc("figure", titlesize=MEDIUM_SIZE)  # Figure title size
    plt.rc("figure", labelsize=MEDIUM_SIZE)  # X and Y labels size

    plt.rc("axes", titlesize=MEDIUM_SIZE)  # Axes title size
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # X and Y labels size
    plt.rc("xtick", labelsize=SMALL_SIZE)  # X tick labels size
    plt.rc("ytick", labelsize=SMALL_SIZE)  # Y tick labels size
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # Legend fontsize

    # Set axis spines visibility
    plt.rc(
        "axes.spines",
        **{
            "left": False,
            "right": False,
            "bottom": False,
            "top": False,
        },
    )

    # Set legend location
    plt.rc(
        "legend",
        **dict(
            loc="upper right",
            frameon=True,
            framealpha=0.5,
            fancybox=False,
            edgecolor="none",
        ),
    )

    plt.rcParams["axes.prop_cycle"] = cycler(color=_default_colors)
    return _default_colors


def get_current_colors() -> list:
    """
    Get the current color cycle of the matplotlib rcParams.

    Returns:
    --------
    colors (np.ndarray) Array containing the current colors in HEX format

    Examples:
    ---------
        >>> colors = get_current_colors()
    """
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_colors(colors) -> Tuple[mpl_figure.Figure, mpl_axes.Axes]:
    """
    Plot a scatter plot of colors.

    Parameters:
    -----------
    colors : list
        List of color values to plot.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    axs : matplotlib.axes.Axes
        The matplotlib Axes object.

    Examples:
    ---------
        >>> colors = ['#FF0000', '#00FF00', '#0000FF']
        >>> fig, axs = plot_colors(colors)
    """
    fig, axs = plt.subplots(figsize=(5, 1))
    for idx, color in enumerate(colors):
        axs.scatter(idx, 1, color=color, s=300)

    axs.set_yticks([])
    return fig, axs


def plot_state_with_probability(
    ax,
    x_value,
    state,
    prob,
    stds=1.96,
    line_kwargs={},
    fill_kwargs=dict(alpha=0.3, label=None),
    output: bool = False,
) -> Union[None, Tuple]:
    """
    Plot the state variable with its corresponding probability distribution.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The matplotlib axes object to plot on.
    x_value : np.ndarray
        The x-axis values.
    state : np.ndarray
        The state variable values.
    prob : np.ndarray
        The probability distribution of the state variable.
    stds : float, optional
        The number of standard deviations to use for the probability interval (default: 1.96).
    line_kwargs : dict, optional
        Additional keyword arguments for the line plot (default: {}).
    fill_kwargs : dict, optional
        Additional keyword arguments for the fill_between plot (default: {'alpha': 0.3, 'label': None}).

    Returns:
    --------
    None

    Examples:
    ---------
        >>> import matplotlib.pyplot as plt
        >>> x = np.linspace(0, 10, 100)
        >>> state = np.sin(x)
        >>> prob = np.abs(np.cos(x))
        >>> fig, ax = plt.subplots()
        >>> plot_state_with_probability(ax, x, state, prob)
    """
    p = ax.plot(x_value, state, **line_kwargs)
    f = ax.fill_between(
        x_value,
        state - stds * np.sqrt(prob),
        state + stds * np.sqrt(prob),
        color=p[0].get_color(),
        **fill_kwargs,
    )
    if output:
        return p, f


def adjust_lightness(color: str, amount: float = 0.75) -> str:
    """
    Adjusts the lightness of a color by the specified amount.

    This function takes a color name or a hexadecimal color code as input and adjusts its lightness
    by the specified amount. The color name can be one of the predefined color names from the Matplotlib
    `cnames` dictionary or a custom color name. If the input color is a hexadecimal color code, it will
    be converted to the corresponding RGB values before adjusting the lightness.

    The lightness adjustment is performed by converting the color to the HLS (Hue, Lightness, Saturation)
    color space, modifying the lightness component by the specified amount, and then converting it back
    to the RGB color space.

    Parameters:
        color (str): The color name or hexadecimal color code to adjust the lightness of.
        amount (float, optional): The amount by which to adjust the lightness.
            Positive values increase the lightness, while negative values decrease it.
            Default is 0.75.

    Returns:
        str: The adjusted color as a hexadecimal color code.

    Example:
        >>> color = "red"
        >>> adjusted_color = adjust_lightness(color, 0.5)
        >>> print(f"Adjusted color: {adjusted_color}")

    References:
        - Function created by Ian Hincks, available at:
          https://stackoverflow.com/a/49601444/16372843
    """
    try:
        try:
            c = cnames[color]
        except:
            c = color
        c = rgb_to_hls(*to_rgb(c))
        c = hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
        return to_hex(c)
    except ValueError:
        warnings.warn(f"Color {color} not found in cnames. Returning original color.")
        return color  # Return the original color if conversion fails


def adjust_lightness_array(colors: Union[list, np.ndarray], amount: float = 0.75) -> np.ndarray:
    """
    Adjusts the lightness of an array of colors by the specified amount.

    This function takes an array of color names or hexadecimal color codes as input and adjusts their lightness
    by the specified amount. The color names can be one of the predefined color names from the Matplotlib
    `cnames` dictionary or a custom color name. If the input color is a hexadecimal color code, it will
    be converted to the corresponding RGB values before adjusting the lightness.

    The lightness adjustment is performed by converting the color to the HLS (Hue, Lightness, Saturation)
    color space, modifying the lightness component by the specified amount, and then converting it back
    to the RGB color space.

    Parameters:
        colors (np.ndarray): The array of color names or hexadecimal color codes to adjust the lightness of.
        amount (float, optional): The amount by which to adjust the lightness.
            Positive values increase the lightness, while negative values decrease it.
            Default is 0.75.

    Returns:
        np.ndarray: The adjusted colors as a hexadecimal color codes.

    Example:
        >>> colors = np.array(["red", "blue", "green"])
        >>> adjusted_colors = adjust_lightness_array(colors, 0.5)
        >>> print(f"Adjusted colors: {adjusted_colors}")

    References:
        - Function created by Ian Hincks, available at:
          https://stackoverflow.com/a/49601444/16372843
    """
    return np.array([adjust_lightness(color, amount) for color in colors])


_dark_colors = adjust_lightness_array(_default_colors, 0.5)


def __set_handler_alpha_to_1__(handle, orig):
    """
    Set the alpha (transparency) of the handler to 1.

    This internal function is used to set the alpha value of a legend handler to 1.
    It is used as an update function in `handler_map_alpha` to modify the legend handler's alpha value.

    Parameters:
        handle: The legend handler object to update.
        orig: The original legend handler object.

    Returns:
        None

    Reference:
        https://stackoverflow.com/a/59629242/16372843
    """
    handle.update_from(orig)
    handle.set_alpha(1)


def handler_map_alpha():
    """
    Create a mapping of legend handler types to update functions.

    This function returns a dictionary that maps specific legend handler types to their corresponding
    update functions. The update functions are used to modify the legend handler's properties,
    such as the alpha (transparency) value.

    Returns:
        dict: A dictionary mapping legend handler types to their update functions.

    Example:
        >>> ax.legend(handler_map = handhandler_map_alpha())
        >>> print(handler_map)
    """
    return {
        PathCollection: HandlerPathCollection(update_func=__set_handler_alpha_to_1__),
        Line2D: HandlerLine2D(update_func=__set_handler_alpha_to_1__),
        Polygon: HandlerPolyCollection(update_func=__set_handler_alpha_to_1__),
    }


def ncols_nrows_from_N(N: int) -> Dict[str, int]:
    """
    Calculate the number of columns and rows for a grid based on the total number of
    elements.

    Given the total number of elements `N`, this function calculates the optimal number of
    columns and rows for a grid layout that can accommodate all the elements.

    Parameters:
        N (int): The total number of elements.

    Returns:
        dict: A dictionary containing the number of columns (`ncols`) and rows (`nrows`) for the grid.

    Examples:
        >>> ncols_nrows_from_N(12)
        {'ncols': 4, 'nrows': 3}

        >>> ncols_nrows_from_N(8)
        {'ncols': 3, 'nrows': 3}

        >>> ncols_nrows_from_N(1)
        {'ncols': 1, 'nrows': 1}
    """
    if not isinstance(N, (int, float)):
        try:
            N = int(N)
            warn(f"N should be and type int but is {type(N)}\nConverted to int. N : {N}")
        except Exception:
            raise ValueError(
                f"N should be and type int but is {type(N)}\nConvertion to int not possible"
            )
    if N <= 0:
        raise ValueError(f"N need to be greater than 1 but is {N}")

    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    return dict(ncols=cols, nrows=rows)


def symmetrize_axis(axes: Axes, axis: Union[int, str]) -> None:
    """
    Symmetrize the given axis of the matplotlib Axes object.

    This function adjusts the limits of the specified axis of the matplotlib Axes object
    to make it symmetrical by setting the minimum and maximum values to their absolute maximum.

    Parameters:
        axes (Axes): The matplotlib Axes object.
        axis (Union[int, str]): The axis to symmetrize. It can be specified either by index (0 for x-axis, 1 for y-axis)
            or by string identifier ("x" for x-axis, "y" for y-axis).

    Returns:
        None

    Examples:
        >>> # Example 1: Symmetrize the x-axis
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot(x, y)
        >>> symmetrize_axis(ax, axis=0)
        >>> plt.show()

        >>> # Example 2: Symmetrize the y-axis
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot(x, y)
        >>> symmetrize_axis(ax, axis="y")
        >>> plt.show()
    """
    if axis in [0, "x"]:
        maxi = np.abs(axes.get_xlim()).max()
        axes.set_xlim(xmin=-maxi, xmax=maxi)
    elif axis in [1, "y"]:
        maxi = np.abs(axes.get_ylim()).max()
        axes.set_ylim(ymin=-maxi, ymax=maxi)


def colorlist_from_cmap(cmap, n, reverse=False) -> list:
    """
    From https://github.com/binodbhttr/mycolorpy.

    Generates n distinct color from a given colormap.

    Args:
        cmap(str): The name of the colormap you want to use.
            Refer https://matplotlib.org/stable/tutorials/colors/colormaps.html to choose
            Suggestions:
            For Metallicity in Astrophysics: Use coolwarm, bwr, seismic in reverse
            For distinct objects: Use gnuplot, brg, jet,turbo.

        n(int): Number of colors you want from the cmap you entered.

        reverse(bool): False by default. Set it to True if you want the cmap result to be reversed.

    Returns:
        colorlist(list): A list with hex values of colors.
    """
    c_map = plt.cm.get_cmap(str(cmap))  # select the desired cmap
    arr = np.linspace(0, 1, n)  # create a list with numbers from 0 to 1 with n items
    colorlist = list()
    for c in arr:
        rgba = c_map(c)  # select the rgba value of the cmap at point c which is a number between 0 to 1
        clr = colors.rgb2hex(rgba)  # convert to hex
        colorlist.append(str(clr))  # create a list of these colors

    if reverse == True:
        colorlist.reverse()
    return colorlist


def colorlist_from_cmap_normalized(cmap, data_arr, reverse=False, vmin=0, vmax=0) -> list:
    """
    From https://github.com/binodbhttr/mycolorpy.

    Generates n distinct color from a given colormap for an array of desired data.
    Args:
        cmap(str): The name of the colormap you want to use.
            Refer https://matplotlib.org/stable/tutorials/colors/colormaps.html to choose

            Some suggestions:
            For Metallicity in Astrophysics: use coolwarm, bwr, seismic in reverse
            For distinct objects: Use gnuplot, brg, jet,turbo.

        data_arr(numpy.ndarray): The numpy array of data for which you want these distinct colors.

        reverse(bool): False by default. Set it to True if you want the cmap result to be reversed.

        vmin(float): 0 by default which sets vmin=minimum value in the data.
            When vmin is assigned a non zero value it normalizes the color based on this minimum value


        vmax(float): 0 by default which set vmax=maximum value in the data.
            When vmax is assigned a non zero value it normalizes the color based on this maximum value

    Returns:
        colorlist_normalized(list): A normalized list of colors with hex values for the given array.
    """

    if (vmin == 0) and (vmax == 0):
        data_min = np.min(data_arr)
        data_max = np.max(data_arr)

    else:
        if vmin > np.min(data_arr):
            warn_string = "vmin you entered is greater than the minimum value in the data array " + str(
                np.min(data_arr)
            )
            warnings.warn(warn_string)

        if vmax < np.max(data_arr):
            warn_string = "vmax you entered is smaller than the maximum value in the data array " + str(
                np.max(data_arr)
            )
            warnings.warn(warn_string)

        data_arr = np.append(data_arr, [vmin, vmax])
        data_min = np.min(data_arr)
        data_max = np.max(data_arr)

    c_map = plt.cm.get_cmap(str(cmap))  # select the desired cmap

    colorlist_normalized = list()
    for c in data_arr:
        norm = (c - data_min) / (data_max - data_min) * 0.99
        rgba = c_map(
            norm
        )  # select the rgba value of the cmap at point c which is a number between 0 to 1
        clr = colors.rgb2hex(rgba)  # convert to hex
        colorlist_normalized.append(str(clr))  # create a list of these colors

    if reverse == True:
        del colorlist_normalized
        colorlist_normalized = list()
        for c in data_arr:
            norm = (c - data_min) / (data_max - data_min) * 0.99
            rgba = c_map(
                1 - norm
            )  # select the rgba value of the cmap at point c which is a number between 0 to 1
            clr = colors.rgb2hex(rgba)  # convert to hex
            colorlist_normalized.append(str(clr))  # create a list of these colors
    if (vmin == 0) and (vmax == 0):
        return colorlist_normalized
    else:
        colorlist_normalized = colorlist_normalized[:-2]
        return colorlist_normalized


def symlog_from_array(
    a: np.ndarray,
    axes: mpl_axes.Axes,
    base: int = 10,
    linthresh: Union[int, None] = None,
    subs: Union[int, None] = None,
    linscale: float = 0.2,
    offset: float = -1,
) -> mpl.scale.SymmetricalLogScale:
    """
    Create a symlog scale for the given data array. The scale is based on the minimum
    value of the data array. Round to next power of ten as the lowest value in the
    logaritmic part of the scale. The base of the scale is 10.

    Parameters
    ----------
    a : np.ndarray
        The data array for which the scale is created.
    axes : mpl.axes.Axes
        The axes for which the scale is created. Default is mpl.Axes.
    base : int, optional
        The base of the scale. Default is 10.
    linthresh : int, optional
        The threshold at which the scale switches from linear to logarithmic. Default is None.
        If None, the threshold is set to the next power of ten of the minimum value of the data array.
    subs : int, optional
        The number of subdivisions of the scale. Default is None.
    linscale : float, optional
        The scale of the linear part of the scale. Default is 0.2.
    offset : float, optional
        The offset of the scale. Default is -1.

    Returns
    -------
    mpl.scale.SymmetricalLogScale
        The symlog scale for the given data array.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib as mpl

    >>> fig, ax = plt.subplots()
    >>> a = np.linspace(-100, 100, 1000)
    >>> ax.plot(a)
    >>> ax.set_yscale(symlog_from_array(a, ax))
    >>> plt.show()
    """

    if isinstance(a, xr.DataArray):
        a = a.values

    if linthresh is None:
        # remove zeros
        a = a[a != 0]
        # round to next power of ten as the lowest value in the logaritmic part of the scale
        linthresh = 10 ** (np.floor(np.log10(np.abs(np.min(a)))) + offset)

    return mpl.scale.SymmetricalLogScale(
        axes, base=base, linthresh=linthresh, subs=subs, linscale=linscale
    )


def plot_thermodynamics(
    fig: mpl_figure.Figure,
    axs: np.ndarray,
    drop_sondes: Union[xr.Dataset, None] = None,
    fit_dict: Union[dict, None] = None,
    fig_title: str = "",
    default_colors: list = get_current_colors(),
    dark_colors: Union[list, None] = None,
    plot_kwargs: dict = dict(alpha=0.75, linewidth=0.7),
    plot_fit_kwargs: dict = dict(alpha=0.75, linewidth=1.5),
) -> Tuple[mpl_figure.Figure, np.ndarray]:
    """
    Plot the thermodynamic profiles of the dropsondes. This will plot the
    following variables:

    - Specific humidity
    - Relative humidity
    - Air temperature
    - Potential temperature

    Parameters:
    -----------
    fig : plt.Figure
        The matplotlib Figure object to plot on.
    axs : np.ndarray
        The matplotlib Axes objects to plot on.
        This needs to be a 2x2 array of Axes objects.
    drop_sondes : xr.Dataset or None
        The dropsondes dataset containing the thermodynamic profiles.
        It should have the following variables:
        'specific_humidity', 'relative_humidity', 'air_temperature', 'potential_temperature'
        and the coordinate 'alt'.
        Default is None (no data is plotted).
    fit_dict : dict or None
        A dictionary containing the fit functions for the thermodynamic profiles.
        The keys should be the variable names and the values should be the fit functions.
        Default is None (no data is plotted).
    fig_title : str, optional
        The title of the figure. Default is an empty string.
    default_colors : list, optional
        The default colors to use for the plot. Default is the current color cycle.
    dark_colors : list, optional
        The darkened colors to use for the fit lines. Default is None.
    plot_kwargs : dict, optional
        Additional keyword arguments for the plot function. Default is {'alpha': 0.75, 'linewidth': 0.7}.
    plot_fit_kwargs : dict, optional
        Additional keyword arguments for the plot function of the fit lines. Default is {'alpha': 0.75, 'linewidth': 1.5}.

    Returns:
    --------
    fig : plt.Figure
        The matplotlib Figure object.
    axs : np.ndarray
        The matplotlib Axes objects.

    Examples:
    ---------
        >>> fig, axs = plot_thermodynamics(fig, drop_sondes, fit_dict, fig_title)
    """

    if dark_colors is None:
        dark_colors = adjust_lightness_array(default_colors, 0.5)

    assert axs.shape == (2, 2), "The number of subplots should be 2x2"

    ax_q, ax_ta = axs[0]
    ax_rh, ax_theta = axs[1]
    ax_q.set_xlim(0, 20)
    ax_rh.set_xlim(30, 110)
    ax_ta.set_xlim(285, 305)
    ax_theta.set_xlim(295, 308)

    plot_dict = {
        "specific_humidity": dict(
            title="Specific humidity [g/kg]",
            xlabel="Specific humidity [g/kg]",
            ylabel="alt [m]",
            ax=ax_q,
            multiplier=1e3,
        ),
        "relative_humidity": dict(
            title="Relative humidity [%]",
            xlabel="Relative humidity [%]",
            ylabel="alt [m]",
            ax=ax_rh,
        ),
        "air_temperature": dict(
            title="Air temperature [K]",
            xlabel="Air temperature [K]",
            ylabel="alt [m]",
            ax=ax_ta,
        ),
        "potential_temperature": dict(
            title="Potential temperature [K]",
            xlabel="Potential temperature [K]",
            ylabel="alt [m]",
            ax=ax_theta,
        ),
    }

    for i, var in enumerate(plot_dict):
        ax = plot_dict[var]["ax"]

        if drop_sondes is None:
            pass
        elif isinstance(drop_sondes, xr.Dataset):
            data = drop_sondes[var]

            if var == "specific_humidity":
                data = 1e3 * data

            ax.plot(
                data.T,
                drop_sondes["alt"].T,
                color=default_colors[i],
                **plot_kwargs,
            )
        else:
            raise ValueError(f"drop_sondes should be a xr.Dataset but is {type(drop_sondes)}")

    # plot fits if available
    for i, var in enumerate(plot_dict):
        ax = plot_dict[var]["ax"]
        if fit_dict is None:
            pass
        elif isinstance(fit_dict, dict):
            fit = fit_dict.get(var)

            if fit != None:
                fitted_data = fit.eval_func(drop_sondes["alt"])[0]
                if var == "specific_humidity":
                    fitted_data = 1e3 * fitted_data
                ax.plot(
                    fitted_data,
                    drop_sondes["alt"],
                    color=dark_colors[i],
                    label="fit",
                    **plot_fit_kwargs,
                )
                ax.axhline(
                    fit.get_x_split(), color="black", linestyle="--", label="cloud base estimated"
                )
        else:
            raise ValueError(f"fit_dict should be a dict but is {type(fit_dict)}")

        ax.xaxis.label.set_color(default_colors[i])  # Set the color of x-axis label
        ax.tick_params(axis="x", colors=default_colors[i])  # Set the color of x-axis ticks
        ax.set_title(plot_dict[var]["title"])
        ax.set_xlabel(plot_dict[var]["xlabel"])
        ax.set_ylabel(plot_dict[var]["ylabel"])

    for ax in axs.flatten():
        ax.legend(handler_map=handler_map_alpha())
        ax.tick_params(axis="x", labelrotation=-33)

    fig.suptitle(fig_title)

    return fig, axs


def set_xticks_time(ax):
    xticks = [0, 500, 1000]
    ax.set_xticks(xticks)


def set_yticks_height(ax):
    yticks = [0, 500, 1000, 1500, 2000]
    ax.set_yticks(yticks)


def set_yticks_height_km(ax):
    yticks = [0, 0.5, 1, 1.5, 2]
    ax.set_yticks(yticks)


def set_logxticks_meter(ax):
    xticks = [1e-6, 1e-3]
    xticklabels = [r"$10^{-6}$", r"$10^{-3}$"]
    ax.set_xticks(xticks, xticklabels)


def set_logxticks_micrometer(ax):
    xticks = [1e-3, 1e0, 1e3]
    xticklabels = [r"$10^{-3}$", r"$10^{0}$", r"$10^{3}$"]
    ax.set_xticks(xticks, xticklabels)


def set_logtyticks_psd(ax):
    yticks = [1e0, 1e6]
    yticklabels = [r"$10^0$", r"$10^6$"]
    ax.set_yticks(yticks, yticklabels)


def label_from_attrs(
    da: xr.DataArray,
    return_name: bool = True,
    return_units: bool = True,
    linebreak: bool = False,
    name_width: Union[int, None] = None,
    units_appendix: Union[str, None] = None,
) -> str:
    """
    This function creates a label from the attributes of a DataArray. It assumes the
    attributes 'long_name' and 'units' are present. If 'long_name' are not present, it
    uses the name of the DataArray. If 'units' are not present, it uses '[???]'.

    Parameters:
    -----------
    da : xr.DataArray
        The DataArray for which to create the label.
    return_name : bool, optional
        Whether to return the name. Default is True.
    return_units : bool, optional
        Whether to return the units. Default is True.
    linebreak : bool, optional
        Whether to insert a linebreak between the name and units. Default is False.
    name_width : int, optional
        The maximum width of the name. Default is None.
        This can be set to a specific value to wrap the name to a specific width.
        The units will still be on the same line as the last part of the name.
        To give it a linebreak, set `linebreak` to True.
    units_appendix : str, optional
        An additional string to append to the units. Default is None.

    Returns:
    --------
    str : The label created from the attributes of the DataArray of the form 'name $[units]$'.
    """

    try:
        name = f"{da.attrs['long_name']}"
    except KeyError:
        name = f"{da.name}"

    # extract the unit key from the attributes, if any is present which fits the known unit keys
    unit_key = find_unit_key(da.attrs)
    if unit_key != None:
        units = f"{da.attrs[unit_key]}"
        if "$" not in units:
            units = f"${units}$"

        units = units.replace("$", " ")
    elif "unit" in da.attrs:
        units = f"{da.attrs['unit']}"
        if "$" not in units:
            units = f"${units}$"

        units = units.replace("$", " ")
    else:
        units = "???"

    # append units appendix if wanted (e.g. [bla] - [bla / log(µm)])
    if units_appendix != None:
        units = f"{units} {units_appendix}"

    # create latex string
    units = rf"$\left[ {units} \right]$"

    # decide the way to return the label string
    if return_name == True:
        if name_width == None:
            name = name
        else:
            name = textwrap.fill(name, name_width)

    if return_name == True and return_units == True:
        if linebreak == True:
            return f"{name}\n{units}"
        else:
            return f"{name} {units}"

    elif return_name == True and return_units == False:
        return f"{name}"
    elif return_name == False and return_units == True:
        return f"{units}"
    else:
        return ""


def add_subplotlabel(
    axs: np.ndarray,
    location: str = "upper left",
    labels=string.ascii_lowercase,
    prefix: str = "(",
    suffix: str = ")",
    count_offset: int = 0,
    **kwargs,
) -> None:
    """
    Add subplot labels to a grid of subplots.

    Parameters
    ----------
    axs : np.ndarray
        The matplotlib Axes objects to add labels to.
    location : str, optional
        The location of the labels. Default is "upper left".
        The available locations for height are:
        - "upper"
        - "lower"
        - "middle"
        The available locations for width are:
        - "left"
        - "right"
        - "center"
        Also the location can be set to "title" to the left of the title. Problems with long titles.
    labels : list, optional
        The labels to add to the subplots.
        Default is the lowercase alphabet.
    prefix : str, optional
        The prefix to add to the labels. Default is '('.
    suffix : str, optional
        The suffix to add to the labels. Default is ')'.
    count_offset : int, optional
        The offset to add to the count. Default is 0.
        With this, the count can be started at a different number.
        For instance 'count = 2' will start the count at '(c)'.

    **kwargs : dict, optional
        Additional keyword arguments for the text function.

    Returns
    --------
    None

    Examples:
    ---------
        >>> fig, axs = plt.subplots(2, 2)
        >>> add_subplotlabel(axs, location="upper left")
    """

    xlocation = None
    xoffset = None
    ylocation = None
    yoffset = None
    at_title = False

    if "left" in location:
        xlocation = 0.05
        xoffset = 0.5
    elif "right" in location:
        xlocation = 0.95
        xoffset = -0.5
    elif "center" in location:
        xlocation = 0.5
        xoffset = 0.0
    if "upper" in location:
        ylocation = 0.95
        yoffset = -0.5
    elif "lower" in location:
        ylocation = 0.05
        yoffset = 0.5
    elif "middle" in location:
        ylocation = 0.5
        yoffset = 0.0

    if xlocation == None or ylocation == None:
        if "title" in location:
            at_title = True
        else:
            raise ValueError(f"Invalid location: {location}")

    for i, ax in enumerate(axs.flatten()):
        i += count_offset
        label = f"{prefix}{labels[i]}{suffix}"

        if at_title == True:
            label = f"  {label}"
            ax.set_title(label, loc="left", **kwargs)
        else:
            ax.annotate(
                label,
                xy=(xlocation, ylocation),
                xycoords="axes fraction",
                xytext=(xoffset, yoffset),
                textcoords="offset fontsize",
                **kwargs,
            )


def find_unit_key(attrs: dict) -> Union[str, None]:
    """
    Check if there is any key similar to 'unit', 'units', 'UNIT', etc., in a dictionary.

    Parameters:
    -----------
    attrs : dict
        The dictionary to check for unit keys.

    Returns:
    --------
    str or None
        The key if found, otherwise None.

    Examples:
    ---------
        >>> attrs = {'units': 'm/s', 'long_name': 'Wind Speed'}
        >>> find_unit_key(attrs)
        'units'
    """

    for key in _known_unit_keys_:
        if key in attrs:
            return key
    return None


def plot_one_one(ax: mpl_axes.Axes, N: int = 100, **kwargs: dict):
    """
    This function plots a one-to-one line on a given axis.

    Parameters:
    -----------
    ax : plt.Axes
        The matplotlib Axes object to plot on.
    N : int, optional
        The number of points to plot. Default is 100.
    **kwargs : dict
        Additional keyword arguments for the plot function.
        It uses the ax.plot function to plot the one-to-one line.
    """
    # get current axis limits
    lims = np.linspace(
        np.nanmin([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.nanmax([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        N,
    )
    ax.plot(lims, lims, **kwargs)


def save_figure(
    fig: mpl_figure.Figure,
    fig_dir: Path = Path("."),
    name: Union[Path, str] = "test",
    formats: dict = {
        ".png": dict(dpi=300),
        ".pdf": dict(),
    },
) -> None:
    """
    Save a figure to the fig_dir with the given name.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    fig_dir : Path
        The directory to save the figure
    name : Path
        The path to store the figure at, relative to ``fig_dir``.
    formats : dict
        The formats to save the figure in.
        The keys are the suffixes of the file, the values are the keyword arguments for the savefig method.
        Defaults is:

        {
            '.png' : dict(dpi = 300),
            '.pdf' : dict(),
        }

    Returns
    -------
    None
    """

    if isinstance(name, str):
        name = Path(name)

    for ext, kwargs in formats.items():

        if "." != ext[0]:
            warnings.warn("The extension should start with a dot.\nCorrection performed.")
            ext = "." + ext

        save_name = replace_path_suffix(name, ext)

        fig.savefig(fig_dir / save_name, **kwargs)


def add_second_x_axis(
    ax: plt.Axes,
    new_xticks_func: Callable,
    xlabel: str,
    position: Literal["top", "bottom"] = "bottom",
    offset_position: Union[
        Tuple[Literal["outward", "axes", "data"], float], Literal["center", "zero"]
    ] = ("zero"),
) -> plt.Axes:
    xticks = ax.get_xticks()
    xlim = ax.get_xlim()
    ax2: plt.Axes = ax.twiny()
    # Offset the twin axis below the host
    ax2.spines[position].set_position(offset_position)

    ax2.xaxis.set_ticks_position(position)
    ax2.xaxis.set_label_position(position)

    # Turn on the frame for the twin axis, but then hide all
    # but the bottom spine
    ax2.set_frame_on(False)
    ax2.patch.set_visible(False)

    # as @ali14 pointed out, for python3, use this
    # for sp in ax2.spines.values():
    # and for python2, use this
    for key, sp in ax2.spines.items():
        sp.set_visible(False)
    ax2.spines[position].set_visible(True)

    new_xticks = new_xticks_func(xticks)

    ax2.set_xticks(xticks, new_xticks)
    ax2.set_xlabel(xlabel)
    ax2.set_xlim(xlim)
    # ax2.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
