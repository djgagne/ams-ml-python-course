"""Methods for plotting ROC (receiver operating characteristic) curve."""

import numpy
import matplotlib.colors
import matplotlib.pyplot as pyplot
from module_4 import performance_diagrams

DEFAULT_LINE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
DEFAULT_LINE_WIDTH = 3
DEFAULT_RANDOM_LINE_COLOUR = numpy.full(3, 152. / 255)
DEFAULT_RANDOM_LINE_WIDTH = 2

LEVELS_FOR_PEIRCE_CONTOURS = numpy.linspace(0, 1, num=11, dtype=float)

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


def _get_pofd_pod_grid(pofd_spacing=0.01, pod_spacing=0.01):
    """Creates grid in POFD-POD space.

    M = number of rows (unique POD values) in grid
    N = number of columns (unique POFD values) in grid

    :param pofd_spacing: Spacing between grid cells in adjacent columns.
    :param pod_spacing: Spacing between grid cells in adjacent rows.
    :return: pofd_matrix: M-by-N numpy array of POFD values.
    :return: pod_matrix: M-by-N numpy array of POD values.
    """

    num_pofd_values = 1 + int(numpy.ceil(1. / pofd_spacing))
    num_pod_values = 1 + int(numpy.ceil(1. / pod_spacing))

    unique_pofd_values = numpy.linspace(0., 1., num=num_pofd_values)
    unique_pod_values = numpy.linspace(0., 1., num=num_pod_values)[::-1]
    return numpy.meshgrid(unique_pofd_values, unique_pod_values)


def _get_peirce_colour_scheme():
    """Returns colour scheme for Peirce score.

    :return: colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap`).
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.
    """

    this_colour_map_object = pyplot.cm.Blues
    this_colour_norm_object = matplotlib.colors.BoundaryNorm(
        LEVELS_FOR_PEIRCE_CONTOURS, this_colour_map_object.N)

    rgba_matrix = this_colour_map_object(this_colour_norm_object(
        LEVELS_FOR_PEIRCE_CONTOURS
    ))

    colour_list = [
        rgba_matrix[i, ..., :-1] for i in range(rgba_matrix.shape[0])
    ]

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.array([1, 1, 1]))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        LEVELS_FOR_PEIRCE_CONTOURS, colour_map_object.N)

    return colour_map_object, colour_norm_object


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

    pofd_matrix, pod_matrix = _get_pofd_pod_grid()
    peirce_score_matrix = pod_matrix - pofd_matrix

    colour_map_object, colour_norm_object = _get_peirce_colour_scheme()

    pyplot.contourf(
        pofd_matrix, pod_matrix, peirce_score_matrix,
        LEVELS_FOR_PEIRCE_CONTOURS, cmap=colour_map_object,
        norm=colour_norm_object, vmin=0., vmax=1., axes=axes_object)

    # TODO(thunderhoser): Calling private method is a HACK.
    colour_bar_object = performance_diagrams._add_colour_bar(
        axes_object=axes_object, colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        values_to_colour=peirce_score_matrix, min_colour_value=0.,
        max_colour_value=1., orientation_string='vertical',
        extend_min=False, extend_max=False)

    print(colour_bar_object)
    colour_bar_object.set_label('Peirce score')

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
