"""Methods for plotting attributes diagram."""

import numpy
from descartes import PolygonPatch
import shapely.geometry
import matplotlib.colors
import matplotlib.pyplot as pyplot

DEFAULT_NUM_BINS = 20
RELIABILITY_LINE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
RELIABILITY_LINE_WIDTH = 3
PERFECT_LINE_COLOUR = numpy.full(3, 152. / 255)
PERFECT_LINE_WIDTH = 2

NO_SKILL_LINE_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
NO_SKILL_LINE_WIDTH = 2
SKILL_AREA_TRANSPARENCY = 0.2
CLIMATOLOGY_LINE_COLOUR = numpy.full(3, 152. / 255)
CLIMATOLOGY_LINE_WIDTH = 2

HISTOGRAM_FACE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)
HISTOGRAM_EDGE_WIDTH = 2

HISTOGRAM_LEFT_EDGE_COORD = 0.575
HISTOGRAM_BOTTOM_EDGE_COORD = 0.175
HISTOGRAM_WIDTH = 0.3
HISTOGRAM_HEIGHT = 0.3

HIST_LEFT_EDGE_FOR_REGRESSION = 0.575
HIST_WIDTH_FOR_REGRESSION = 0.3
HIST_BOTTOM_EDGE_FOR_REGRESSION = 0.225
HIST_HEIGHT_FOR_REGRESSION = 0.25

HISTOGRAM_X_TICK_VALUES = numpy.linspace(0, 1, num=6, dtype=float)
HISTOGRAM_X_TICKS_FOR_REGRESSION = numpy.linspace(0, 0.02, num=11)
HISTOGRAM_Y_TICK_SPACING = 0.1

FIGURE_WIDTH_INCHES = 10
FIGURE_HEIGHT_INCHES = 10

FONT_SIZE = 20
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _get_histogram(input_values, num_bins, min_value, max_value):
    """Creates histogram with uniform bin-spacing.

    E = number of input values
    B = number of bins

    :param input_values: length-E numpy array of values to bin.
    :param num_bins: Number of bins (B).
    :param min_value: Minimum value.  Any input value < `min_value` will be
        assigned to the first bin.
    :param max_value: Max value.  Any input value > `max_value` will be
        assigned to the last bin.
    :return: inputs_to_bins: length-E numpy array of bin indices (integers).
    """

    bin_cutoffs = numpy.linspace(min_value, max_value, num=num_bins + 1)

    inputs_to_bins = numpy.digitize(
        input_values, bin_cutoffs, right=False
    ) - 1

    inputs_to_bins[inputs_to_bins < 0] = 0
    inputs_to_bins[inputs_to_bins > num_bins - 1] = num_bins - 1

    return inputs_to_bins


def _vertices_to_polygon_object(x_vertices, y_vertices):
    """Converts two arrays of vertices to `shapely.geometry.Polygon` object.

    V = number of vertices

    This method allows for simple polygons only (no disjoint polygons, no
    holes).

    :param x_vertices: length-V numpy array of x-coordinates.
    :param y_vertices: length-V numpy array of y-coordinates.
    :return: polygon_object: Instance of `shapely.geometry.Polygon`.
    """

    list_of_vertices = []

    for i in range(len(x_vertices)):
        list_of_vertices.append(
            (x_vertices[i], y_vertices[i])
        )

    return shapely.geometry.Polygon(shell=list_of_vertices)


def _plot_background(axes_object, observed_labels):
    """Plots background of attributes diagram.

    E = number of examples

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    :param observed_labels: length-E numpy array of class labels (integers in
        0...1).
    """

    # Plot positive-skill area.
    climatology = numpy.mean(observed_labels.astype(float))
    skill_area_colour = matplotlib.colors.to_rgba(
        NO_SKILL_LINE_COLOUR, SKILL_AREA_TRANSPARENCY)

    x_vertices_left = numpy.array([0, climatology, climatology, 0, 0])
    y_vertices_left = numpy.array([0, 0, climatology, climatology / 2, 0])

    left_polygon_object = _vertices_to_polygon_object(
        x_vertices=x_vertices_left, y_vertices=y_vertices_left)
    left_polygon_patch = PolygonPatch(
        left_polygon_object, lw=0, ec=skill_area_colour, fc=skill_area_colour)
    axes_object.add_patch(left_polygon_patch)

    x_vertices_right = numpy.array(
        [climatology, 1, 1, climatology, climatology]
    )
    y_vertices_right = numpy.array(
        [climatology, (1 + climatology) / 2, 1, 1, climatology]
    )

    right_polygon_object = _vertices_to_polygon_object(
        x_vertices=x_vertices_right, y_vertices=y_vertices_right)
    right_polygon_patch = PolygonPatch(
        right_polygon_object, lw=0, ec=skill_area_colour, fc=skill_area_colour)
    axes_object.add_patch(right_polygon_patch)

    # Plot no-skill line (at edge of positive-skill area).
    no_skill_x_coords = numpy.array([0, 1], dtype=float)
    no_skill_y_coords = numpy.array([climatology, 1 + climatology]) / 2
    axes_object.plot(
        no_skill_x_coords, no_skill_y_coords, color=NO_SKILL_LINE_COLOUR,
        linestyle='solid', linewidth=NO_SKILL_LINE_WIDTH)

    # Plot climatology line (vertical).
    climo_line_x_coords = numpy.full(2, climatology)
    climo_line_y_coords = numpy.array([0, 1], dtype=float)
    axes_object.plot(
        climo_line_x_coords, climo_line_y_coords, color=CLIMATOLOGY_LINE_COLOUR,
        linestyle='dashed', linewidth=CLIMATOLOGY_LINE_WIDTH)

    # Plot no-resolution line (horizontal).
    no_resolution_x_coords = climo_line_y_coords + 0.
    no_resolution_y_coords = climo_line_x_coords + 0.
    axes_object.plot(
        no_resolution_x_coords, no_resolution_y_coords,
        color=CLIMATOLOGY_LINE_COLOUR, linestyle='dashed',
        linewidth=CLIMATOLOGY_LINE_WIDTH)


def _floor_to_nearest(input_value_or_array, increment):
    """Rounds number(s) down to the nearest multiple of `increment`.

    :param input_value_or_array: Input (either scalar or numpy array).
    :param increment: Increment (or rounding base -- whatever you want to call
        it).
    :return: output_value_or_array: Rounded version of `input_value_or_array`.
    """

    return increment * numpy.floor(input_value_or_array / increment)


def _plot_forecast_histogram(figure_object, num_examples_by_bin):
    """Plots forecast histogram as inset in the attributes diagram.

    B = number of bins

    :param figure_object: Instance of `matplotlib.figure.Figure`.  Will plot in
        this figure.
    :param num_examples_by_bin: length-B numpy array, where
        num_examples_by_bin[j] = number of examples in [j]th forecast bin.
    """

    num_bins = len(num_examples_by_bin)
    bin_frequencies = (
        num_examples_by_bin.astype(float) / numpy.sum(num_examples_by_bin)
    )

    forecast_bin_edges = numpy.linspace(0, 1, num=num_bins + 1, dtype=float)
    forecast_bin_width = forecast_bin_edges[1] - forecast_bin_edges[0]
    forecast_bin_centers = forecast_bin_edges[:-1] + forecast_bin_width / 2

    inset_axes_object = figure_object.add_axes(
        [HISTOGRAM_LEFT_EDGE_COORD, HISTOGRAM_BOTTOM_EDGE_COORD,
         HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT]
    )

    inset_axes_object.bar(
        forecast_bin_centers, bin_frequencies, forecast_bin_width,
        color=HISTOGRAM_FACE_COLOUR, edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH)

    max_y_tick_value = _floor_to_nearest(
        1.05 * numpy.max(bin_frequencies), HISTOGRAM_Y_TICK_SPACING)
    num_y_ticks = 1 + int(numpy.round(
        max_y_tick_value / HISTOGRAM_Y_TICK_SPACING
    ))

    y_tick_values = numpy.linspace(0, max_y_tick_value, num=num_y_ticks)
    pyplot.yticks(y_tick_values, axes=inset_axes_object)
    pyplot.xticks(HISTOGRAM_X_TICK_VALUES, axes=inset_axes_object)

    inset_axes_object.set_xlim(0, 1)
    inset_axes_object.set_ylim(0, 1.05 * numpy.max(bin_frequencies))


def _plot_forecast_hist_for_regression(
        figure_object, mean_forecast_by_bin, num_examples_by_bin):
    """Plots forecast histogram for regression.

    B = number of bins

    :param figure_object: Will plot histogram as inset in this figure (instance
        of `matplotlib.figure.Figure`).
    :param mean_forecast_by_bin: length-B numpy array of mean forecast values.
    :param num_examples_by_bin: length-B numpy array of example counts.
    """

    bin_frequencies = (
        num_examples_by_bin.astype(float) / numpy.sum(num_examples_by_bin)
    )

    num_bins = len(num_examples_by_bin)
    forecast_bin_width = (
        (numpy.max(mean_forecast_by_bin) - numpy.min(mean_forecast_by_bin)) /
        (num_bins - 1)
    )

    inset_axes_object = figure_object.add_axes([
        HIST_LEFT_EDGE_FOR_REGRESSION, HIST_BOTTOM_EDGE_FOR_REGRESSION,
        HIST_WIDTH_FOR_REGRESSION, HIST_HEIGHT_FOR_REGRESSION
    ])

    inset_axes_object.bar(
        mean_forecast_by_bin, bin_frequencies, forecast_bin_width,
        color=HISTOGRAM_FACE_COLOUR, edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH)

    max_y_tick_value = _floor_to_nearest(
        1.05 * numpy.max(bin_frequencies), HISTOGRAM_Y_TICK_SPACING
    )
    num_y_ticks = 1 + int(numpy.round(
        max_y_tick_value / HISTOGRAM_Y_TICK_SPACING
    ))

    y_tick_values = numpy.linspace(0, max_y_tick_value, num=num_y_ticks)
    pyplot.yticks(y_tick_values, axes=inset_axes_object)
    pyplot.xticks(HISTOGRAM_X_TICKS_FOR_REGRESSION, axes=inset_axes_object,
                  rotation=90.)

    inset_axes_object.set_xlim(
        0, numpy.max(mean_forecast_by_bin) + forecast_bin_width
    )
    inset_axes_object.set_ylim(0, 1.05 * numpy.max(bin_frequencies))


def _get_points_in_regression_relia_curve(observed_values, forecast_values,
                                          num_bins):
    """Creates points for regression-based reliability curve.

    E = number of examples
    B = number of bins

    :param observed_values: length-E numpy array of observed target values.
    :param forecast_values: length-E numpy array of forecast target values.
    :param num_bins: Number of bins for forecast value.
    :return: mean_forecast_by_bin: length-B numpy array of mean forecast values.
    :return: mean_observation_by_bin: length-B numpy array of mean observed
        values.
    :return: num_examples_by_bin: length-B numpy array with number of examples
        in each forecast bin.
    """

    inputs_to_bins = _get_histogram(
        input_values=forecast_values, num_bins=num_bins,
        min_value=numpy.min(forecast_values),
        max_value=numpy.max(forecast_values)
    )

    mean_forecast_by_bin = numpy.full(num_bins, numpy.nan)
    mean_observation_by_bin = numpy.full(num_bins, numpy.nan)
    num_examples_by_bin = numpy.full(num_bins, -1, dtype=int)

    for k in range(num_bins):
        these_example_indices = numpy.where(inputs_to_bins == k)[0]
        num_examples_by_bin[k] = len(these_example_indices)

        mean_forecast_by_bin[k] = numpy.mean(
            forecast_values[these_example_indices]
        )

        mean_observation_by_bin[k] = numpy.mean(
            observed_values[these_example_indices]
        )

    return mean_forecast_by_bin, mean_observation_by_bin, num_examples_by_bin


def get_points_in_relia_curve(
        observed_labels, forecast_probabilities, num_bins):
    """Creates points for reliability curve.

    The reliability curve is the main component of the attributes diagram.

    E = number of examples
    B = number of bins

    :param observed_labels: length-E numpy array of class labels (integers in
        0...1).
    :param forecast_probabilities: length-E numpy array with forecast
        probabilities of label = 1.
    :param num_bins: Number of bins for forecast probability.
    :return: mean_forecast_probs: length-B numpy array of mean forecast
        probabilities.
    :return: mean_event_frequencies: length-B numpy array of conditional mean
        event frequencies.  mean_event_frequencies[j] = frequency of label 1
        when forecast probability is in the [j]th bin.
    :return: num_examples_by_bin: length-B numpy array with number of examples
        in each forecast bin.
    """

    assert numpy.all(numpy.logical_or(
        observed_labels == 0, observed_labels == 1
    ))

    assert numpy.all(numpy.logical_and(
        forecast_probabilities >= 0, forecast_probabilities <= 1
    ))

    assert num_bins > 1

    inputs_to_bins = _get_histogram(
        input_values=forecast_probabilities, num_bins=num_bins, min_value=0.,
        max_value=1.)

    mean_forecast_probs = numpy.full(num_bins, numpy.nan)
    mean_event_frequencies = numpy.full(num_bins, numpy.nan)
    num_examples_by_bin = numpy.full(num_bins, -1, dtype=int)

    for k in range(num_bins):
        these_example_indices = numpy.where(inputs_to_bins == k)[0]
        num_examples_by_bin[k] = len(these_example_indices)

        mean_forecast_probs[k] = numpy.mean(
            forecast_probabilities[these_example_indices])

        mean_event_frequencies[k] = numpy.mean(
            observed_labels[these_example_indices].astype(float)
        )

    return mean_forecast_probs, mean_event_frequencies, num_examples_by_bin


def plot_reliability_curve(
        observed_labels, forecast_probabilities, num_bins=DEFAULT_NUM_BINS,
        axes_object=None):
    """Plots reliability curve.

    E = number of examples

    :param observed_labels: length-E numpy array of class labels (integers in
        0...1).
    :param forecast_probabilities: length-E numpy array with forecast
        probabilities of label = 1.
    :param num_bins: Number of bins for forecast probability.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).  If `axes_object is None`,
        will create new axes.
    :return: mean_forecast_probs: See doc for `get_points_in_relia_curve`.
    :return: mean_event_frequencies: Same.
    :return: num_examples_by_bin: Same.
    """

    mean_forecast_probs, mean_event_frequencies, num_examples_by_bin = (
        get_points_in_relia_curve(
            observed_labels=observed_labels,
            forecast_probabilities=forecast_probabilities, num_bins=num_bins)
    )

    if axes_object is None:
        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    perfect_x_coords = numpy.array([0, 1], dtype=float)
    perfect_y_coords = perfect_x_coords + 0.
    axes_object.plot(
        perfect_x_coords, perfect_y_coords, color=PERFECT_LINE_COLOUR,
        linestyle='dashed', linewidth=PERFECT_LINE_WIDTH)

    real_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(mean_forecast_probs), numpy.isnan(mean_event_frequencies)
    )))[0]

    axes_object.plot(
        mean_forecast_probs[real_indices], mean_event_frequencies[real_indices],
        color=RELIABILITY_LINE_COLOUR,
        linestyle='solid', linewidth=RELIABILITY_LINE_WIDTH)

    axes_object.set_xlabel('Forecast probability')
    axes_object.set_ylabel('Conditional event frequency')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)

    return mean_forecast_probs, mean_event_frequencies, num_examples_by_bin


def plot_regression_relia_curve(
        observed_values, forecast_values, num_bins=DEFAULT_NUM_BINS,
        figure_object=None, axes_object=None):
    """Plots reliability curve for regression.

    :param observed_values: See doc for `get_points_in_regression_relia_curve`.
    :param forecast_values: Same.
    :param num_bins: Same.
    :param figure_object: See doc for `plot_attributes_diagram`.
    :param axes_object: Same.
    """

    mean_forecast_by_bin, mean_observation_by_bin, num_examples_by_bin = (
        _get_points_in_regression_relia_curve(
            observed_values=observed_values, forecast_values=forecast_values,
            num_bins=num_bins)
    )

    if figure_object is None or axes_object is None:
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    _plot_forecast_hist_for_regression(
        figure_object=figure_object, mean_forecast_by_bin=mean_forecast_by_bin,
        num_examples_by_bin=num_examples_by_bin)

    max_forecast_or_observed = max([
        numpy.max(forecast_values), numpy.max(observed_values)
    ])

    perfect_x_coords = numpy.array([0., max_forecast_or_observed])
    perfect_y_coords = perfect_x_coords + 0.
    axes_object.plot(
        perfect_x_coords, perfect_y_coords, color=PERFECT_LINE_COLOUR,
        linestyle='dashed', linewidth=PERFECT_LINE_WIDTH)

    real_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(mean_forecast_by_bin), numpy.isnan(mean_observation_by_bin)
    )))[0]

    axes_object.plot(
        mean_forecast_by_bin[real_indices],
        mean_observation_by_bin[real_indices],
        color=RELIABILITY_LINE_COLOUR,
        linestyle='solid', linewidth=RELIABILITY_LINE_WIDTH)

    axes_object.set_xlabel('Forecast value')
    axes_object.set_ylabel('Conditional mean observation')
    axes_object.set_xlim(0., max_forecast_or_observed)
    axes_object.set_ylim(0., max_forecast_or_observed)


def plot_attributes_diagram(
        observed_labels, forecast_probabilities, num_bins=DEFAULT_NUM_BINS,
        figure_object=None, axes_object=None):
    """Plots attributes diagram.

    :param observed_labels: See doc for `plot_reliability_curve`.
    :param forecast_probabilities: Same.
    :param num_bins: Same.
    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).  If `figure_object is None`, will create a
        new one.
    :param axes_object: See doc for `plot_reliability_curve`.
    :return: mean_forecast_probs: See doc for `get_points_in_relia_curve`.
    :return: mean_event_frequencies: Same.
    :return: num_examples_by_bin: Same.
    """

    mean_forecast_probs, mean_event_frequencies, num_examples_by_bin = (
        get_points_in_relia_curve(
            observed_labels=observed_labels,
            forecast_probabilities=forecast_probabilities, num_bins=num_bins)
    )

    if figure_object is None or axes_object is None:
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    _plot_background(axes_object=axes_object, observed_labels=observed_labels)
    _plot_forecast_histogram(figure_object=figure_object,
                             num_examples_by_bin=num_examples_by_bin)

    plot_reliability_curve(
        observed_labels=observed_labels,
        forecast_probabilities=forecast_probabilities, num_bins=num_bins,
        axes_object=axes_object)

    return mean_forecast_probs, mean_event_frequencies, num_examples_by_bin
