"""Methods for plotting ROC (receiver operating characteristic) curve."""

import numpy
import matplotlib.pyplot as pyplot

DEFAULT_LINE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
DEFAULT_LINE_WIDTH = 3
DEFAULT_RANDOM_LINE_COLOUR = numpy.full(3, 152. / 255)
DEFAULT_RANDOM_LINE_WIDTH = 2

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _get_points_in_roc_curve(observed_labels, forecast_probabilities):
    """Creates points for ROC curve.

    E = number of examples
    T = number of binarization thresholds

    :param observed_labels: length-E numpy array of class labels (integers in
        0...1).
    :param forecast_probabilities: length-E numpy array with forecast
        probabilities of label = 1.
    :return: pofd_by_threshold: length-T numpy array of POFD (probability of
        false detection) values.
    :return: pod_by_threshold: length-T numpy array of POD (probability of
        detection) values.
    """

    assert numpy.all(numpy.logical_or(
        observed_labels == 0, observed_labels == 1
    ))

    assert numpy.all(numpy.logical_and(
        forecast_probabilities >= 0, forecast_probabilities <= 1
    ))

    observed_labels = observed_labels.astype(int)
    binarization_thresholds = numpy.linspace(0, 1, num=1001, dtype=float)

    num_thresholds = len(binarization_thresholds)
    pofd_by_threshold = numpy.full(num_thresholds, numpy.nan)
    pod_by_threshold = numpy.full(num_thresholds, numpy.nan)

    for k in range(num_thresholds):
        these_forecast_labels = (
            forecast_probabilities >= binarization_thresholds[k]
        ).astype(int)

        this_num_hits = numpy.sum(numpy.logical_and(
            these_forecast_labels == 1, observed_labels == 1
        ))

        this_num_false_alarms = numpy.sum(numpy.logical_and(
            these_forecast_labels == 1, observed_labels == 0
        ))

        this_num_misses = numpy.sum(numpy.logical_and(
            these_forecast_labels == 0, observed_labels == 1
        ))

        this_num_correct_nulls = numpy.sum(numpy.logical_and(
            these_forecast_labels == 0, observed_labels == 0
        ))

        try:
            pofd_by_threshold[k] = (
                float(this_num_false_alarms) /
                (this_num_false_alarms + this_num_correct_nulls)
            )
        except ZeroDivisionError:
            pass

        try:
            pod_by_threshold[k] = (
                float(this_num_hits) / (this_num_hits + this_num_misses)
            )
        except ZeroDivisionError:
            pass

    pod_by_threshold = numpy.array([1.] + pod_by_threshold.tolist() + [0.])
    pofd_by_threshold = numpy.array([1.] + pofd_by_threshold.tolist() + [0.])

    return pofd_by_threshold, pod_by_threshold


def plot_roc_curve(
        observed_labels, forecast_probabilities,
        line_colour=DEFAULT_LINE_COLOUR, line_width=DEFAULT_LINE_WIDTH,
        random_line_colour=DEFAULT_RANDOM_LINE_COLOUR,
        random_line_width=DEFAULT_RANDOM_LINE_WIDTH):
    """Plots ROC curve.

    E = number of examples

    :param observed_labels: length-E numpy array of class labels (integers in
        0...1).
    :param forecast_probabilities: length-E numpy array with forecast
        probabilities of label = 1.
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    :param line_width: Line width (real positive number).
    :param random_line_colour: Colour of reference line (ROC curve for random
        predictor).
    :param random_line_width: Width of reference line (ROC curve for random
        predictor).
    :return: pofd_by_threshold: See doc for `_get_points_in_roc_curve`.
    :return: pod_by_threshold: Same.
    """

    pofd_by_threshold, pod_by_threshold = _get_points_in_roc_curve(
        observed_labels=observed_labels,
        forecast_probabilities=forecast_probabilities)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    random_x_coords = numpy.array([0., 1.])
    random_y_coords = numpy.array([0., 1.])
    axes_object.plot(
        random_x_coords, random_y_coords, color=random_line_colour,
        linestyle='dashed', linewidth=random_line_width)

    nan_flags = numpy.logical_or(
        numpy.isnan(pofd_by_threshold), numpy.isnan(pod_by_threshold)
    )

    if not numpy.all(nan_flags):
        real_indices = numpy.where(numpy.invert(nan_flags))[0]
        axes_object.plot(
            pofd_by_threshold[real_indices], pod_by_threshold[real_indices],
            color=line_colour, linestyle='solid', linewidth=line_width)

    axes_object.set_xlabel('POFD (probability of false detection)')
    axes_object.set_ylabel('POD (probability of detection)')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)

    return pofd_by_threshold, pod_by_threshold
