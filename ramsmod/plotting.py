import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm, expon, lognorm, weibull_min, chi2
from ramsmod.utils import convert_to_pd_series

__all__ = ['plot_right_censored', 'plot_interval_censored', 'plot_np_reliability',
           'plot_exponential_prob_plot', 'plot_weibull_prob_plot', 'plot_lognormal_prob_plot',
           'plot_exponential_relative_likelihoods', 'plot_weibull_relative_likelihoods',
           'plot_lognormal_relative_likelihoods']


def plot_right_censored(t, d, ax=None, show_legend=True):
    """
    Returns a plot of observations from right-censored failure data.
    :param t: Survival times for each observation.
    :param d: Indicator variable value for each observation, where
    value 1 indicates exact failure observed and 0 indicates failure was right-censored.
    :param ax: Matplotlib axes on which to plot, if None then one will be created.
    :param show_legend: Boolean that is True if legend should be added to axes and
    False otherwise.

    :return: Matplotlib axes containing the plot.
    """
    t = convert_to_pd_series(t)
    d = convert_to_pd_series(d)

    if ax is None:
        ax = plt.gca()

    # Add the observations to the axes.
    for i in range(len(t)):
        if d.iloc[i]:
            # Failure observation.
            ax.scatter(t.iloc[i], i+1, color='b', marker='o')
        else:
            # Right-censored observation.
            ax.scatter(t.iloc[i], i+1, color='b', marker='>')

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel('Observation #')

    # Add legend to axes.
    if show_legend:
        legend_elements = [
            Line2D([0], [0], color='w', markeredgecolor='b',
                   markerfacecolor='b', marker='>',
                   label='Right-censored'),
            Line2D([0], [0], color='w',markeredgecolor='b',
                   markerfacecolor='b', marker='o',
                   label='Exact')]
        ax.legend(handles=legend_elements)

    return ax


def plot_interval_censored(tmin, tmax, ax=None, show_legend=True):
    """
    Returns a plot of observations from interval-censored failure data.
    :param tmin: The exclusive lower bounds of the failure time intervals.
    :param tmax: The inclusive upper bound of the failure time intervals
    (use np.infty for right-censored observations).
    :param ax: Matplotlib axes on which to plot, if None then one will be created.
    :param show_legend: Boolean that is True if legend should be added to axes and
    False otherwise.
    :return: Matplotlib axes containing the plot.
    """
    tmin = convert_to_pd_series(tmin)
    tmax = convert_to_pd_series(tmax)

    if ax is None:
        fig = plt.figure()  # Create plot figure.
        ax = fig.add_subplot()  # Create the axes to plot on.

    # Add the observations to the axes.
    for i in range(len(tmin)):
        if tmin.iloc[i] == tmax.iloc[i]:
            ax.scatter(tmin.iloc[i], i + 1, color='b', marker='o')
        elif tmax.iloc[i] == np.infty:
            ax.scatter(tmin.iloc[i], i + 1, color='b', marker='>')
        else:
            ax.hlines(i + 1, tmin.iloc[i], tmax.iloc[i], color='b')

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel('Observation #')
    ax.set_xlabel('Time')
    ax.set_xlim(0)
    ax.set_ylim(0)

    # Add legend to axes.
    if show_legend:
        legend_elements = [Line2D([0], [0], color='b',
                                  label='Interval-censored'),
                           Line2D([0], [0], color='w', markeredgecolor='b',
                                  markerfacecolor='b', marker='>',
                                  label='Right-censored'),
                           Line2D([0], [0], color='w', markeredgecolor='b',
                                  markerfacecolor='b', marker='o',
                                  label='Exact')]
        ax.legend(handles=legend_elements, loc='upper right')

    return ax


def plot_np_reliability(t, r, ci_lb=None, ci_ub=None, ax=None, linestyle='-',
                        color='blue', label=None):
    """
    Produces a step plot of reliability estimates against time.
    :param t: The time values (Panda's series, list etc.).
    :param r: The reliability estimates (Panda's series, list etc.) corresponding
    to the time values.
    :param ci_lb: The reliability estimates corresponding to the time values
    at the lower bound of the confidence interval.
    :param ci_ub: The reliability estimates corresponding to the time values
    at the upper bound of the confidence interval.
    :param ax: The Matplotlib axes on which to draw the plot.
    :param linestyle: The Matplotlib linestyle option for the plot.
    :param color: The Matplotlib color style option for the plot.
    :param label: The label for the plot.
    :return: The Matplotlib axes containing the plot.
    """
    if ax is None:
        ax = plt.gca()

    # Plot r against t as step function.
    ax.step(t, r, where='post', color=color,
            label=label, linestyle=linestyle)

    # Add confidence intervals if provided.
    if ci_lb is not None:
        ax.step(t, ci_lb, where='post', color=color, alpha=0.5,
                linestyle=linestyle)
    if ci_ub is not None:
        ax.step(t, ci_ub, where='post', color=color, alpha=0.5,
                linestyle=linestyle)

    # Add axis-labels.
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\widehat R(t)$')

    ax.set_xlim(0)

    return ax


def plot_exponential_prob_plot(t, r, ci_lb=None, ci_ub=None, ax=None, marker='o',
                        color='blue', label=None):
    """
    Produces a linearised probability plot for the exponential distribution.
    :param t: The time values.
    :param r: The reliability estimates corresponding to the time values.
    :param ci_lb: The reliability estimates corresponding to the time values
    at the lower bound of the confidence interval.
    :param ci_ub: The reliability estimates corresponding to the time values
    at the upper bound of the confidence interval.
    :param ax: The Matplotlib axes on which to draw the plot.
    :param marker: The Matplotlib marker style option for the plot.
    :param color: The Matplotlib color style option for the plot.
    :param label: The label for the plot.
    :return: The Matplotlib axes containing the plot.
    """
    if ax is None:
        ax = plt.gca()

    y = -np.log(r)
    ax.scatter(t, y, color=color,
            label=label, marker=marker)
    if ci_lb is not None:
        y_lb = -np.log(ci_lb)
        ax.plot(t, y_lb, color=color, alpha=0.5,
                   label=label, linestyle='--')
    if ci_ub is not None:
        y_ub = -np.log(ci_ub)
        ax.plot(t, y_ub, color=color, alpha=0.5,
                  label=label, linestyle='--')

    # Add axis-labels.
    ax.set_xlabel(r'$t$', fontsize=12)
    ax.set_ylabel(r'$-\ln(\widehat R(t))$', fontsize=12)

    return ax


def plot_weibull_prob_plot(t, r, ci_lb=None, ci_ub=None, ax=None, marker='o',
                                   color='blue', label=None):
    """
    Produces a linearised probability plot for the Weibull distribution.
    :param t: The time values.
    :param r: The reliability estimates corresponding to the time values.
    :param ci_lb: The reliability estimates corresponding to the time values
    at the lower bound of the confidence interval.
    :param ci_ub: The reliability estimates corresponding to the time values
    at the upper bound of the confidence interval.
    :param ax: The Matplotlib axes on which to draw the plot.
    :param marker: The Matplotlib marker style option for the plot.
    :param color: The Matplotlib color style option for the plot.
    :param label: The label for the plot.
    :return: The Matplotlib axes containing the plot.
    """
    if ax is None:
        ax = plt.gca()

    y = np.log(-np.log(r))
    x = np.log(t)
    ax.scatter(x, y, color=color,
               label=label, marker=marker)
    if ci_lb is not None:
        y_lb = np.log(-np.log(ci_lb))
        ax.plot(x, y_lb, color=color, alpha=0.5,
                   label=label, linestyle='--')
    if ci_ub is not None:
        y_ub = np.log(-np.log(ci_ub))
        ax.plot(x, y_ub, color=color, alpha=0.5,
                   label=label, linestyle='--')

    # Add axis-labels.
    ax.set_xlabel(r'$\ln(t)$', fontsize=12)
    ax.set_ylabel(r'$\ln(-\ln(\widehat R(t)))$', fontsize=12)

    return ax


def plot_lognormal_prob_plot(t, r, ci_lb=None, ci_ub=None, ax=None, marker='o',
                           color='blue', label=None):
    """
    Produces a linearised probability plot for the log-normal distribution.
    :param t: The time values.
    :param r: The reliability estimates corresponding to the time values.
    :param ci_lb: The reliability estimates corresponding to the time values
    at the lower bound of the confidence interval.
    :param ci_ub: The reliability estimates corresponding to the time values
    at the upper bound of the confidence interval.
    :param ax: The Matplotlib axes on which to draw the plot.
    :param marker: The Matplotlib marker style option for the plot.
    :param color: The Matplotlib color style option for the plot.
    :param label: The label for the plot.
    :return: The Matplotlib axes containing the plot.
    """
    if ax is None:
        ax = plt.gca()

    y = norm.ppf(1-r)
    x = np.log(t)
    ax.scatter(x, y, color=color,
               label=label, marker=marker)
    if ci_lb is not None:
        y_lb = norm.ppf(1-ci_lb)
        ax.plot(x, y_lb, color=color, alpha=0.5,
                   label=label, linestyle='--')
    if ci_ub is not None:
        y_ub = norm.ppf(1-ci_ub)
        ax.plot(x, y_ub, color=color, alpha=0.5,
                   label=label, linestyle='--')

    # Add axis-labels.
    ax.set_xlabel(r'$\ln(t)$', fontsize=12)
    ax.set_ylabel(r'$\sigma^{-1}(1-\widehat R(t))$', fontsize=12)

    return ax


def _get_exponential_relative_log_likelihoods(tmin, tmax, scales):

    total_lls = np.zeros(scales.shape)

    interval_t = tmin != tmax
    exact_t = tmin == tmax

    for i in range(len(scales)):
        # Compute likelihood.
        scale = scales[i]

        interval_log_likelihoods = np.log(expon.cdf(tmax[interval_t], scale=scale) -
                                 expon.cdf(tmin[interval_t], scale=scale))
        exact_log_likelihoods = np.log(expon.pdf(tmin[exact_t], scale=scale))
        total_log_likelihood = interval_log_likelihoods.sum() + exact_log_likelihoods.sum()
        total_lls[i] = total_log_likelihood

    max_ll = max(total_lls)
    relative_likelihoods = np.exp(total_lls - max_ll)

    return relative_likelihoods


def _get_total_log_likelihood(ttf_rv, tmin, tmax):
    interval_t = tmin != tmax
    exact_t = tmin == tmax

    interval_log_likelihoods = np.log(ttf_rv.cdf(tmax[interval_t]) - ttf_rv.cdf(tmin[interval_t]))
    exact_log_likelihoods = np.log(ttf_rv.pdf(tmin[exact_t]))
    total_log_likelihood = interval_log_likelihoods.sum() + exact_log_likelihoods.sum()

    return total_log_likelihood


def _get_shape_scale_relative_likelihoods(tmin, tmax, shapes, scales, dist_fun):

    total_log_likelihoods = np.zeros((shapes.shape[0], scales.shape[0]))

    for i in range(len(shapes)):
        for j in range(len(scales)):
            # Compute likelihood.
            shape = shapes[i]
            scale = scales[j]

            ttf_rv = dist_fun(shape, scale=scale)

            # Compute total log-likelihood of observations in the failure data.
            total_log_likelihood = _get_total_log_likelihood(ttf_rv, tmin, tmax)

            # Compute the total log-likelihood and store in array.
            total_log_likelihoods[i, j] = total_log_likelihood

    # Compute maximum log-likelihood.
    max_log_likelihood = np.max(total_log_likelihoods)

    # Compute relative likelihoods.
    relative_likelihoods = np.exp(total_log_likelihoods - max_log_likelihood)

    return relative_likelihoods


def plot_exponential_relative_likelihoods(tmin, tmax, min_scale, max_scale, step_scale, ci_alpha=0.05,
                                          ax=None, linestyle='-', color='blue', label=None):
    """
    Plots the relative likelihoods corresponding to scale parameter value of the exponential
    distribution for a set of interval censored failure data, including a confidence interval
    for the true scale parameter value.
    :param tmin: The lower bounds for the failure time observations.
    :param tmax: The upper bounds for the failure time observations.
    :param min_scale: The minimum value for the scale parameter included on the plot.
    :param max_scale: The maximum value for the scale parameter included on the plot.
    :param step_scale: The step (increment) between plotted scale parameter values.
    :param ci_alpha: The significance value for the confidence interval (i.e.
    producing (1-ci_alpha)*100% confidence interval).
    :param ax: The Matplotlib axes on which to draw the plot, new axes created if None.
    :param linestyle: The Matplotlib linestyle option for the plot.
    :param color: The Matplotlib color option for the plot.
    :param label: The label for the plot line.
    :return: The Matplotlib axes containing the plot.
    """
    if ax is None:
        ax = plt.gca()

    scales = np.arange(min_scale, max_scale, step_scale)
    relative_likelihoods = _get_exponential_relative_log_likelihoods(tmin, tmax, scales)
    ax.plot(scales, relative_likelihoods, linestyle=linestyle, color=color, label=label)

    # Confidence interval.
    ci_relative_likelihood = np.exp(-chi2.ppf(1 - ci_alpha, df=1) / 2)

    ci_region = relative_likelihoods >= ci_relative_likelihood
    ci_region_likelihoods = relative_likelihoods[ci_region]
    ci_region_scales = scales[ci_region]

    ax.fill_between(ci_region_scales, ci_region_likelihoods, color=color, alpha=0.5)
    ax.set_xlabel(r'$\Theta$')
    ax.set_ylabel(r'$R(\Theta)$')
    ax.set_ylim(0)

    return ax


def _plot_shape_scale_confidence_regions(tmin, tmax, min_shape, max_shape, step_shape,
                                         min_scale, max_scale, step_scale, dist_fun,
                                         label_as_confidence, ax=None):

    if ax is None:
        ax = plt.gca()

    shapes = np.arange(min_shape, max_shape, step_shape)
    scales = np.arange(min_scale, max_scale, step_scale)
    relative_likelihoods = _get_shape_scale_relative_likelihoods(tmin, tmax, shapes, scales, dist_fun)

    cs = ax.contour(scales, shapes, relative_likelihoods)
    levels = cs.levels

    if label_as_confidence:
        def format_confidence_levels(level):
            value = 100*chi2.cdf(-2 * np.log(level), df=2).round(2)
            return f"{value:.2f}%"

        ax.clabel(cs, levels, fmt=format_confidence_levels)
        ax.set_title('Confidence regions')
    else:
        ax.clabel(cs, levels)
        ax.set_title('Relative likelihood regions')

    return ax


def plot_weibull_relative_likelihoods(tmin, tmax, min_shape, max_shape, step_shape, min_scale,
                                      max_scale, step_scale, label_as_confidence=False, ax=None):
    """
    Plots (as a contour plot) the relative likelihoods corresponding to parameter values
    of the Weibull distribution for a set of interval censored failure data.
    :param tmin: The lower bounds for the failure time observations.
    :param tmax: The upper bounds for the failure time observations.
    :param min_shape: The minimum value for the shape parameter included on the plot.
    :param max_shape: The maximum value for the shape parameter included on the plot.
    :param step_shape: The step (increment) between plotted shape parameter values.
    :param min_scale: The minimum value for the scale parameter included on the plot.
    :param max_scale: The maximum value for the scale parameter included on the plot.
    :param step_scale: The step (increment) between plotted scale parameter values.
    :param label_as_confidence: If True then the confidence level contours are plotted
    instead of relative likelihoods.
    :param ax: The Matplotlib axes on which to draw the plot, if None then new axes created.
    :return: The Matplotlib axes containing the plot.
    """
    dist_fun = weibull_min
    ax = _plot_shape_scale_confidence_regions(tmin, tmax, min_shape, max_shape, step_shape,
                                              min_scale, max_scale, step_scale, dist_fun,
                                              label_as_confidence, ax=ax)
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$\beta$')

    return ax


def plot_lognormal_relative_likelihoods(tmin, tmax, min_shape, max_shape, step_shape,
                                        min_scale, max_scale, step_scale, label_as_confidence=False,
                                        ax=None):
    """
    Plots (as a contour plot) the relative likelihoods corresponding to parameter values
    of the log-normal distribution for a set of interval censored failure data.
    :param tmin: The lower bounds for the failure time observations.
    :param tmax: The upper bounds for the failure time observations.
    :param min_shape: The minimum value for the shape parameter included on the plot.
    :param max_shape: The maximum value for the shape parameter included on the plot.
    :param step_shape: The step (increment) between plotted shape parameter values.
    :param min_scale: The minimum value for the scale parameter included on the plot.
    :param max_scale: The maximum value for the scale parameter included on the plot.
    :param step_scale: The step (increment) between plotted scale parameter values.
    :param label_as_confidence: If True then the confidence level contours are plotted
    instead of relative likelihoods.
    :param ax: The Matplotlib axes on which to draw the plot, if None then new axes created.
    :return: The Matplotlib axes containing the plot.
    """
    dist_fun = lognorm
    ax = _plot_shape_scale_confidence_regions(tmin, tmax, min_shape, max_shape, step_shape,
                                              min_scale, max_scale, step_scale, dist_fun,
                                              label_as_confidence, ax=ax)
    plt.xlabel(r'$m$')
    plt.ylabel(r'$\sigma$')

    return ax