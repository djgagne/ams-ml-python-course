"""Code for AMS 2019 short course."""

import copy
import glob
import errno
import random
import os.path
import json
import pickle
import time
import calendar
import numpy
import netCDF4
from scipy.interpolate import (
    UnivariateSpline, RectBivariateSpline, RegularGridInterpolator)
import keras
from keras import backend as K
import tensorflow
from tensorflow.python.framework import ops as tensorflow_ops
from sklearn.metrics import auc as scikit_learn_auc
import matplotlib.colors
import matplotlib.pyplot as pyplot
from module_4 import keras_metrics
from module_4 import roc_curves
from module_4 import performance_diagrams
from module_4 import attributes_diagrams

# Directories.
# MODULE4_DIR_NAME = '.'
# SHORT_COURSE_DIR_NAME = '..'

MODULE4_DIR_NAME = os.path.dirname(__file__)
SHORT_COURSE_DIR_NAME = os.path.dirname(MODULE4_DIR_NAME)
DEFAULT_IMAGE_DIR_NAME = '{0:s}/data/track_data_ncar_ams_3km_nc_small'.format(
    SHORT_COURSE_DIR_NAME)

# Plotting constants.
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

BAR_GRAPH_FACE_COLOUR = numpy.array([166, 206, 227], dtype=float) / 255
BAR_GRAPH_EDGE_COLOUR = numpy.full(3, 0.)
BAR_GRAPH_EDGE_WIDTH = 2.

SALIENCY_COLOUR_MAP_OBJECT = pyplot.cm.Greys

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

# Naming constants.
CSV_METADATA_COLUMNS = [
    'Step_ID', 'Track_ID', 'Ensemble_Name', 'Ensemble_Member', 'Run_Date',
    'Valid_Date', 'Forecast_Hour', 'Valid_Hour_UTC'
]

CSV_EXTRANEOUS_COLUMNS = [
    'Duration', 'Centroid_Lon', 'Centroid_Lat', 'Centroid_X', 'Centroid_Y',
    'Storm_Motion_U', 'Storm_Motion_V', 'Matched', 'Max_Hail_Size',
    'Num_Matches', 'Shape', 'Location', 'Scale'
]

CSV_TARGET_NAME = 'RVORT1_MAX-future_max'
TARGET_NAME = 'max_future_vorticity_s01'

NETCDF_REFL_NAME = 'REFL_COM_curr'
NETCDF_TEMP_NAME = 'T2_curr'
NETCDF_U_WIND_NAME = 'U10_curr'
NETCDF_V_WIND_NAME = 'V10_curr'
NETCDF_PREDICTOR_NAMES = [
    NETCDF_REFL_NAME, NETCDF_TEMP_NAME, NETCDF_U_WIND_NAME, NETCDF_V_WIND_NAME
]

REFLECTIVITY_NAME = 'reflectivity_dbz'
TEMPERATURE_NAME = 'temperature_kelvins'
U_WIND_NAME = 'u_wind_m_s01'
V_WIND_NAME = 'v_wind_m_s01'
PREDICTOR_NAMES = [
    REFLECTIVITY_NAME, TEMPERATURE_NAME, U_WIND_NAME, V_WIND_NAME
]

NETCDF_TRACK_ID_NAME = 'track_id'
NETCDF_TRACK_STEP_NAME = 'track_step'
NETCDF_TARGET_NAME = 'RVORT1_MAX_future'

NUM_VALUES_KEY = 'num_values'
MEAN_VALUE_KEY = 'mean_value'
MEAN_OF_SQUARES_KEY = 'mean_of_squares'

STORM_IDS_KEY = 'storm_ids'
STORM_STEPS_KEY = 'storm_steps'
PREDICTOR_NAMES_KEY = 'predictor_names'
PREDICTOR_MATRIX_KEY = 'predictor_matrix'
TARGET_NAME_KEY = 'target_name'
TARGET_MATRIX_KEY = 'target_matrix'

TRAINING_FILES_KEY = 'training_file_names'
NORMALIZATION_DICT_KEY = 'normalization_dict'
BINARIZATION_THRESHOLD_KEY = 'binarization_threshold'
NUM_EXAMPLES_PER_BATCH_KEY = 'num_examples_per_batch'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
VALIDATION_FILES_KEY = 'validation_file_names'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
CNN_FILE_KEY = 'cnn_file_name'
CNN_FEATURE_LAYER_KEY = 'cnn_feature_layer_name'

PERMUTED_PREDICTORS_KEY = 'permuted_predictor_name_by_step'
HIGHEST_COSTS_KEY = 'highest_cost_by_step'
ORIGINAL_COST_KEY = 'original_cost'
STEP1_PREDICTORS_KEY = 'predictor_names_step1'
STEP1_COSTS_KEY = 'costs_step1'

EOF_MATRIX_KEY = 'eof_matrix'
FEATURE_MEANS_KEY = 'feature_means'
FEATURE_STDEVS_KEY = 'feature_standard_deviations'

NOVEL_IMAGES_ACTUAL_KEY = 'novel_image_matrix_actual'
NOVEL_IMAGES_UPCONV_KEY = 'novel_image_matrix_upconv'
NOVEL_IMAGES_UPCONV_SVD_KEY = 'novel_image_matrix_upconv_svd'

# More plotting constants.
THIS_COLOUR_LIST = [
    numpy.array([4, 233, 231]), numpy.array([1, 159, 244]),
    numpy.array([3, 0, 244]), numpy.array([2, 253, 2]),
    numpy.array([1, 197, 1]), numpy.array([0, 142, 0]),
    numpy.array([253, 248, 2]), numpy.array([229, 188, 0]),
    numpy.array([253, 149, 0]), numpy.array([253, 0, 0]),
    numpy.array([212, 0, 0]), numpy.array([188, 0, 0]),
    numpy.array([248, 0, 253]), numpy.array([152, 84, 198])
]

for p in range(len(THIS_COLOUR_LIST)):
    THIS_COLOUR_LIST[p] = THIS_COLOUR_LIST[p].astype(float) / 255

REFL_COLOUR_MAP_OBJECT = matplotlib.colors.ListedColormap(THIS_COLOUR_LIST)
REFL_COLOUR_MAP_OBJECT.set_under(numpy.ones(3))

PREDICTOR_TO_COLOUR_MAP_DICT = {
    TEMPERATURE_NAME: pyplot.cm.YlOrRd,
    REFLECTIVITY_NAME: REFL_COLOUR_MAP_OBJECT,
    U_WIND_NAME: pyplot.cm.seismic,
    V_WIND_NAME: pyplot.cm.seismic
}

THESE_COLOUR_BOUNDS = numpy.array(
    [0.1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
REFL_COLOUR_NORM_OBJECT = matplotlib.colors.BoundaryNorm(
    THESE_COLOUR_BOUNDS, REFL_COLOUR_MAP_OBJECT.N)

# Deep-learning constants.
L1_WEIGHT = 0.
L2_WEIGHT = 0.001
NUM_PREDICTORS_TO_FIRST_NUM_FILTERS = 8
NUM_CONV_LAYER_SETS = 2
NUM_CONV_LAYERS_PER_SET = 2
NUM_CONV_FILTER_ROWS = 3
NUM_CONV_FILTER_COLUMNS = 3
CONV_LAYER_DROPOUT_FRACTION = None
USE_BATCH_NORMALIZATION = True
SLOPE_FOR_RELU = 0.2
NUM_POOLING_ROWS = 2
NUM_POOLING_COLUMNS = 2
NUM_DENSE_LAYERS = 3
DENSE_LAYER_DROPOUT_FRACTION = 0.5

NUM_SMOOTHING_FILTER_ROWS = 5
NUM_SMOOTHING_FILTER_COLUMNS = 5

MIN_XENTROPY_DECREASE_FOR_EARLY_STOP = 0.005
MIN_MSE_DECREASE_FOR_EARLY_STOP = 0.005
NUM_EPOCHS_FOR_EARLY_STOPPING = 5

LIST_OF_METRIC_FUNCTIONS = [
    keras_metrics.accuracy, keras_metrics.binary_accuracy,
    keras_metrics.binary_csi, keras_metrics.binary_frequency_bias,
    keras_metrics.binary_pod, keras_metrics.binary_pofd,
    keras_metrics.binary_peirce_score, keras_metrics.binary_success_ratio,
    keras_metrics.binary_focn
]

METRIC_FUNCTION_DICT = {
    'accuracy': keras_metrics.accuracy,
    'binary_accuracy': keras_metrics.binary_accuracy,
    'binary_csi': keras_metrics.binary_csi,
    'binary_frequency_bias': keras_metrics.binary_frequency_bias,
    'binary_pod': keras_metrics.binary_pod,
    'binary_pofd': keras_metrics.binary_pofd,
    'binary_peirce_score': keras_metrics.binary_peirce_score,
    'binary_success_ratio': keras_metrics.binary_success_ratio,
    'binary_focn': keras_metrics.binary_focn
}

DEFAULT_NUM_BWO_ITERATIONS = 200
DEFAULT_BWO_LEARNING_RATE = 0.01

# Misc constants.
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

DATE_FORMAT = '%Y%m%d'
DATE_FORMAT_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]'

BACKPROP_FUNCTION_NAME = 'GuidedBackProp'

MIN_PROBABILITY = 1e-15
MAX_PROBABILITY = 1. - MIN_PROBABILITY
METRES_PER_SECOND_TO_KT = 3.6 / 1.852


def time_string_to_unix(time_string, time_format):
    """Converts time from string to Unix format.

    Unix format = seconds since 0000 UTC 1 Jan 1970.

    :param time_string: Time string.
    :param time_format: Format of time string (example: "%Y%m%d" or
        "%Y-%m-%d-%H%M%S").
    :return: unix_time_sec: Time in Unix format.
    """

    return calendar.timegm(time.strptime(time_string, time_format))


def time_unix_to_string(unix_time_sec, time_format):
    """Converts time from Unix format to string.

    Unix format = seconds since 0000 UTC 1 Jan 1970.

    :param unix_time_sec: Time in Unix format.
    :param time_format: Desired format of time string (example: "%Y%m%d" or
        "%Y-%m-%d-%H%M%S").
    :return: time_string: Time string.
    """

    return time.strftime(time_format, time.gmtime(unix_time_sec))


def _image_file_name_to_date(netcdf_file_name):
    """Parses date from name of image (NetCDF) file.

    :param netcdf_file_name: Path to input file.
    :return: date_string: Date (format "yyyymmdd").
    """

    pathless_file_name = os.path.split(netcdf_file_name)[-1]
    date_string = pathless_file_name.replace(
        'NCARSTORM_', '').replace('-0000_d01_model_patches.nc', '')

    # Verify.
    time_string_to_unix(time_string=date_string, time_format=DATE_FORMAT)
    return date_string


def find_many_image_files(first_date_string, last_date_string,
                          image_dir_name=DEFAULT_IMAGE_DIR_NAME):
    """Finds image (NetCDF) files in the given date range.

    :param first_date_string: First date ("yyyymmdd") in range.
    :param last_date_string: Last date ("yyyymmdd") in range.
    :param image_dir_name: Name of directory with image (NetCDF) files.
    :return: netcdf_file_names: 1-D list of paths to image files.
    """

    first_time_unix_sec = time_string_to_unix(
        time_string=first_date_string, time_format=DATE_FORMAT)
    last_time_unix_sec = time_string_to_unix(
        time_string=last_date_string, time_format=DATE_FORMAT)

    netcdf_file_pattern = (
        '{0:s}/NCARSTORM_{1:s}-0000_d01_model_patches.nc'
    ).format(image_dir_name, DATE_FORMAT_REGEX)

    netcdf_file_names = glob.glob(netcdf_file_pattern)
    netcdf_file_names.sort()

    file_date_strings = [_image_file_name_to_date(f) for f in netcdf_file_names]
    file_times_unix_sec = numpy.array([
        time_string_to_unix(time_string=d, time_format=DATE_FORMAT)
        for d in file_date_strings
    ], dtype=int)

    good_indices = numpy.where(numpy.logical_and(
        file_times_unix_sec >= first_time_unix_sec,
        file_times_unix_sec <= last_time_unix_sec
    ))[0]

    return [netcdf_file_names[k] for k in good_indices]


def read_image_file(netcdf_file_name):
    """Reads storm-centered images from NetCDF file.

    E = number of examples (storm objects) in file
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)

    :param netcdf_file_name: Path to input file.
    :return: image_dict: Dictionary with the following keys.
    image_dict['storm_ids']: length-E list of storm IDs (integers).
    image_dict['storm_steps']: length-E numpy array of storm steps (integers).
    image_dict['predictor_names']: length-C list of predictor names.
    image_dict['predictor_matrix']: E-by-M-by-N-by-C numpy array of predictor
        values.
    image_dict['target_name']: Name of target variable.
    image_dict['target_matrix']: E-by-M-by-N numpy array of target values.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    storm_ids = numpy.array(
        dataset_object.variables[NETCDF_TRACK_ID_NAME][:], dtype=int)
    storm_steps = numpy.array(
        dataset_object.variables[NETCDF_TRACK_STEP_NAME][:], dtype=int)

    predictor_matrix = None

    for this_predictor_name in NETCDF_PREDICTOR_NAMES:
        this_predictor_matrix = numpy.array(
            dataset_object.variables[this_predictor_name][:], dtype=float)
        this_predictor_matrix = numpy.expand_dims(
            this_predictor_matrix, axis=-1)

        if predictor_matrix is None:
            predictor_matrix = this_predictor_matrix + 0.
        else:
            predictor_matrix = numpy.concatenate(
                (predictor_matrix, this_predictor_matrix), axis=-1)

    target_matrix = numpy.array(
        dataset_object.variables[NETCDF_TARGET_NAME][:], dtype=float)

    return {
        STORM_IDS_KEY: storm_ids,
        STORM_STEPS_KEY: storm_steps,
        PREDICTOR_NAMES_KEY: PREDICTOR_NAMES,
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        TARGET_NAME_KEY: TARGET_NAME,
        TARGET_MATRIX_KEY: target_matrix
    }


def read_many_image_files(netcdf_file_names):
    """Reads storm-centered images from many NetCDF files.

    :param netcdf_file_names: 1-D list of paths to input files.
    :return: image_dict: See doc for `read_image_file`.
    """

    image_dict = None
    keys_to_concat = [
        STORM_IDS_KEY, STORM_STEPS_KEY, PREDICTOR_MATRIX_KEY, TARGET_MATRIX_KEY
    ]

    for this_file_name in netcdf_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_image_dict = read_image_file(this_file_name)

        if image_dict is None:
            image_dict = copy.deepcopy(this_image_dict)
            continue

        for this_key in keys_to_concat:
            image_dict[this_key] = numpy.concatenate(
                (image_dict[this_key], this_image_dict[this_key]), axis=0)

    return image_dict


def image_files_example1():
    """Runs Example 1 for feature files."""

    image_file_names = find_many_image_files(
        first_date_string='20150701', last_date_string='20150731')
    image_dict = read_many_image_files(image_file_names)

    print(MINOR_SEPARATOR_STRING)
    print('Variables in dictionary are as follows:')
    for this_key in image_dict.keys():
        print(this_key)

    print('\nPredictor variables are as follows:')
    predictor_names = image_dict[PREDICTOR_NAMES_KEY]
    for this_name in predictor_names:
        print(this_name)

    these_predictor_values = image_dict[PREDICTOR_MATRIX_KEY][0, :5, :5, 0]
    print(
        ('\nSome values of predictor variable "{0:s}" for first storm object:'
         '\n{1:s}'
         ).format(predictor_names[0], str(these_predictor_values))
    )

    these_target_values = image_dict[TARGET_MATRIX_KEY][0, :5, :5]
    print(
        ('\nSome values of target variable "{0:s}" for first storm object:'
         '\n{1:s}'
         ).format(image_dict[TARGET_NAME_KEY], str(these_target_values))
    )


def find_training_files_example():
    """Finds training files."""

    training_file_names = find_many_image_files(
        first_date_string='20100101', last_date_string='20141231')

    validation_file_names = find_many_image_files(
        first_date_string='20150101', last_date_string='20151231')


def _init_figure_panels(num_rows, num_columns, horizontal_space_fraction=0.1,
                        vertical_space_fraction=0.1):
    """Initializes paneled figure.

    :param num_rows: Number of panel rows.
    :param num_columns: Number of panel columns.
    :param horizontal_space_fraction: Horizontal space between panels (as
        fraction of panel size).
    :param vertical_space_fraction: Vertical space between panels (as fraction
        of panel size).
    :return: figure_object: Instance of `matplotlib.figure.Figure`.
    :return: axes_objects_2d_list: 2-D list, where axes_objects_2d_list[i][j] is
        the handle (instance of `matplotlib.axes._subplots.AxesSubplot`) for the
        [i]th row and [j]th column.
    """

    figure_object, axes_objects_2d_list = pyplot.subplots(
        num_rows, num_columns, sharex=False, sharey=False,
        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if num_rows == num_columns == 1:
        axes_objects_2d_list = [[axes_objects_2d_list]]
    elif num_columns == 1:
        axes_objects_2d_list = [[a] for a in axes_objects_2d_list]
    elif num_rows == 1:
        axes_objects_2d_list = [axes_objects_2d_list]

    pyplot.subplots_adjust(
        left=0.02, bottom=0.02, right=0.98, top=0.95,
        hspace=vertical_space_fraction, wspace=horizontal_space_fraction)

    return figure_object, axes_objects_2d_list


def _add_colour_bar(
        axes_object, colour_map_object, values_to_colour, min_colour_value,
        max_colour_value, colour_norm_object=None,
        orientation_string='vertical', extend_min=True, extend_max=True):
    """Adds colour bar to existing axes.

    :param axes_object: Existing axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param values_to_colour: numpy array of values to colour.
    :param min_colour_value: Minimum value in colour map.
    :param max_colour_value: Max value in colour map.
    :param colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.  If `colour_norm_object is None`,
        will assume that scale is linear.
    :param orientation_string: Orientation of colour bar ("vertical" or
        "horizontal").
    :param extend_min: Boolean flag.  If True, the bottom of the colour bar will
        have an arrow.  If False, it will be a flat line, suggesting that lower
        values are not possible.
    :param extend_max: Same but for top of colour bar.
    :return: colour_bar_object: Colour bar (instance of
        `matplotlib.pyplot.colorbar`) created by this method.
    """

    if colour_norm_object is None:
        colour_norm_object = matplotlib.colors.Normalize(
            vmin=min_colour_value, vmax=max_colour_value, clip=False)

    scalar_mappable_object = pyplot.cm.ScalarMappable(
        cmap=colour_map_object, norm=colour_norm_object)
    scalar_mappable_object.set_array(values_to_colour)

    if extend_min and extend_max:
        extend_string = 'both'
    elif extend_min:
        extend_string = 'min'
    elif extend_max:
        extend_string = 'max'
    else:
        extend_string = 'neither'

    if orientation_string == 'horizontal':
        padding = 0.075
    else:
        padding = 0.05

    colour_bar_object = pyplot.colorbar(
        ax=axes_object, mappable=scalar_mappable_object,
        orientation=orientation_string, pad=padding, extend=extend_string)

    colour_bar_object.ax.tick_params(labelsize=FONT_SIZE)
    return colour_bar_object


def plot_predictor_2d(
        predictor_matrix, colour_map_object, colour_norm_object=None,
        min_colour_value=None, max_colour_value=None, axes_object=None):
    """Plots predictor variable on 2-D grid.

    If `colour_norm_object is None`, both `min_colour_value` and
    `max_colour_value` must be specified.

    M = number of rows in grid
    N = number of columns in grid

    :param predictor_matrix: M-by-N numpy array of predictor values.
    :param colour_map_object: Instance of `matplotlib.pyplot.cm`.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    :return: colour_bar_object: Colour bar (instance of
        `matplotlib.pyplot.colorbar`) created by this method.
    """

    if axes_object is None:
        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    if colour_norm_object is not None:
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]

    axes_object.pcolormesh(
        predictor_matrix, cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None')

    axes_object.set_xticks([])
    axes_object.set_yticks([])

    return _add_colour_bar(
        axes_object=axes_object, colour_map_object=colour_map_object,
        values_to_colour=predictor_matrix, min_colour_value=min_colour_value,
        max_colour_value=max_colour_value)


def plot_wind_2d(u_wind_matrix_m_s01, v_wind_matrix_m_s01, axes_object=None):
    """Plots wind velocity on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param u_wind_matrix_m_s01: M-by-N numpy array of eastward components
        (metres per second).
    :param v_wind_matrix_m_s01: M-by-N numpy array of northward components
        (metres per second).
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    """

    if axes_object is None:
        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    num_grid_rows = u_wind_matrix_m_s01.shape[0]
    num_grid_columns = u_wind_matrix_m_s01.shape[1]

    x_coords_unique = numpy.linspace(
        0, num_grid_columns, num=num_grid_columns + 1, dtype=float)
    x_coords_unique = x_coords_unique[:-1]
    x_coords_unique = x_coords_unique + numpy.diff(x_coords_unique[:2]) / 2

    y_coords_unique = numpy.linspace(
        0, num_grid_rows, num=num_grid_rows + 1, dtype=float)
    y_coords_unique = y_coords_unique[:-1]
    y_coords_unique = y_coords_unique + numpy.diff(y_coords_unique[:2]) / 2

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(x_coords_unique,
                                                    y_coords_unique)

    speed_matrix_m_s01 = numpy.sqrt(u_wind_matrix_m_s01 ** 2
                                    + v_wind_matrix_m_s01 ** 2)

    axes_object.barbs(
        x_coord_matrix, y_coord_matrix,
        u_wind_matrix_m_s01 * METRES_PER_SECOND_TO_KT,
        v_wind_matrix_m_s01 * METRES_PER_SECOND_TO_KT,
        speed_matrix_m_s01 * METRES_PER_SECOND_TO_KT, color='k', length=6,
        sizes={'emptybarb': 0.1}, fill_empty=True, rounding=False)

    axes_object.set_xlim(0, num_grid_columns)
    axes_object.set_ylim(0, num_grid_rows)


def plot_many_predictors_with_barbs(
        predictor_matrix, predictor_names, min_colour_temp_kelvins,
        max_colour_temp_kelvins):
    """Plots many predictor variables on 2-D grid with wind barbs overlain.

    M = number of rows in grid
    N = number of columns in grid
    C = number of predictors

    :param predictor_matrix: M-by-N-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param min_colour_temp_kelvins: Minimum value in temperature colour scheme.
    :param max_colour_temp_kelvins: Max value in temperature colour scheme.
    :return: figure_object: See doc for `_init_figure_panels`.
    :return: axes_objects_2d_list: Same.
    """

    u_wind_matrix_m_s01 = predictor_matrix[
        ..., predictor_names.index(U_WIND_NAME)]
    v_wind_matrix_m_s01 = predictor_matrix[
        ..., predictor_names.index(V_WIND_NAME)]

    non_wind_predictor_names = [
        p for p in predictor_names if p not in [U_WIND_NAME, V_WIND_NAME]
    ]

    figure_object, axes_objects_2d_list = _init_figure_panels(
        num_rows=len(non_wind_predictor_names), num_columns=1)

    for m in range(len(non_wind_predictor_names)):
        this_predictor_index = predictor_names.index(
            non_wind_predictor_names[m])

        if non_wind_predictor_names[m] == REFLECTIVITY_NAME:
            this_colour_norm_object = REFL_COLOUR_NORM_OBJECT
            this_min_colour_value = None
            this_max_colour_value = None
        else:
            this_colour_norm_object = None
            this_min_colour_value = min_colour_temp_kelvins + 0.
            this_max_colour_value = max_colour_temp_kelvins + 0.

        this_colour_bar_object = plot_predictor_2d(
            predictor_matrix=predictor_matrix[..., this_predictor_index],
            colour_map_object=PREDICTOR_TO_COLOUR_MAP_DICT[
                non_wind_predictor_names[m]],
            colour_norm_object=this_colour_norm_object,
            min_colour_value=this_min_colour_value,
            max_colour_value=this_max_colour_value,
            axes_object=axes_objects_2d_list[m][0])

        plot_wind_2d(u_wind_matrix_m_s01=u_wind_matrix_m_s01,
                     v_wind_matrix_m_s01=v_wind_matrix_m_s01,
                     axes_object=axes_objects_2d_list[m][0])

        this_colour_bar_object.set_label(non_wind_predictor_names[m])

    return figure_object, axes_objects_2d_list


def plot_many_predictors_sans_barbs(
        predictor_matrix, predictor_names, min_colour_temp_kelvins,
        max_colour_temp_kelvins, max_colour_wind_speed_m_s01):
    """Plots many predictor variables on 2-D grid; no wind barbs overlain.

    In this case, both u-wind and v-wind are plotted as separate maps.

    M = number of rows in grid
    N = number of columns in grid
    C = number of predictors

    :param predictor_matrix: M-by-N-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param min_colour_temp_kelvins: Minimum value in temperature colour scheme.
    :param max_colour_temp_kelvins: Max value in temperature colour scheme.
    :param max_colour_wind_speed_m_s01: Max wind speed (metres per second) in
        colour maps for both u- and v-components.  The minimum wind speed be
        `-1 * max_colour_wind_speed_m_s01`, so the diverging colour scheme will
        be zero-centered.
    :return: figure_object: See doc for `_init_figure_panels`.
    :return: axes_objects_2d_list: Same.
    """

    num_predictors = len(predictor_names)
    num_panel_rows = int(numpy.floor(numpy.sqrt(num_predictors)))
    num_panel_columns = int(numpy.ceil(float(num_predictors) / num_panel_rows))

    figure_object, axes_objects_2d_list = _init_figure_panels(
        num_rows=num_panel_rows, num_columns=num_panel_columns)

    for i in range(num_panel_rows):
        for j in range(num_panel_columns):
            this_linear_index = i * num_panel_columns + j
            if this_linear_index >= num_predictors:
                break

            this_colour_map_object = PREDICTOR_TO_COLOUR_MAP_DICT[
                predictor_names[this_linear_index]]

            if predictor_names[this_linear_index] == REFLECTIVITY_NAME:
                this_colour_norm_object = REFL_COLOUR_NORM_OBJECT
                this_min_colour_value = None
                this_max_colour_value = None
            elif predictor_names[this_linear_index] == TEMPERATURE_NAME:
                this_colour_norm_object = None
                this_min_colour_value = min_colour_temp_kelvins + 0.
                this_max_colour_value = max_colour_temp_kelvins + 0.
            else:
                this_colour_norm_object = None
                this_min_colour_value = -1 * max_colour_wind_speed_m_s01
                this_max_colour_value = max_colour_wind_speed_m_s01 + 0.

            this_colour_bar_object = plot_predictor_2d(
                predictor_matrix=predictor_matrix[..., this_linear_index],
                colour_map_object=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                min_colour_value=this_min_colour_value,
                max_colour_value=this_max_colour_value,
                axes_object=axes_objects_2d_list[i][j])

            this_colour_bar_object.set_label(predictor_names[this_linear_index])

    return figure_object, axes_objects_2d_list


def plot_predictors_example1(validation_file_names):
    """Plots all predictors for random example (storm object).

    :param validation_file_names: 1-D list of paths to input files.
    """

    validation_image_dict = read_many_image_files(validation_file_names)
    print(SEPARATOR_STRING)

    predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]
    temperature_matrix_kelvins = predictor_matrix[
        ..., predictor_names.index(TEMPERATURE_NAME)]

    plot_many_predictors_with_barbs(
        predictor_matrix=predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=numpy.percentile(temperature_matrix_kelvins, 1),
        max_colour_temp_kelvins=numpy.percentile(temperature_matrix_kelvins, 99)
    )

    pyplot.show()


def plot_predictors_example2(validation_image_dict):
    """Plots all predictors for example with greatest future vorticity.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    """

    target_matrix_s01 = validation_image_dict[TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][
        example_index, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]
    temperature_matrix_kelvins = predictor_matrix[
        ..., predictor_names.index(TEMPERATURE_NAME)]

    plot_many_predictors_with_barbs(
        predictor_matrix=predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=numpy.percentile(temperature_matrix_kelvins, 1),
        max_colour_temp_kelvins=numpy.percentile(temperature_matrix_kelvins, 99)
    )

    pyplot.show()


def _update_normalization_params(intermediate_normalization_dict, new_values):
    """Updates normalization params for one predictor.

    :param intermediate_normalization_dict: Dictionary with the following keys.
    intermediate_normalization_dict['num_values']: Number of values on which
        current estimates are based.
    intermediate_normalization_dict['mean_value']: Current estimate for mean.
    intermediate_normalization_dict['mean_of_squares']: Current mean of squared
        values.

    :param new_values: numpy array of new values (will be used to update
        `intermediate_normalization_dict`).
    :return: intermediate_normalization_dict: Same as input but with updated
        values.
    """

    if MEAN_VALUE_KEY not in intermediate_normalization_dict:
        intermediate_normalization_dict = {
            NUM_VALUES_KEY: 0,
            MEAN_VALUE_KEY: 0.,
            MEAN_OF_SQUARES_KEY: 0.
        }

    these_means = numpy.array([
        intermediate_normalization_dict[MEAN_VALUE_KEY], numpy.mean(new_values)
    ])
    these_weights = numpy.array([
        intermediate_normalization_dict[NUM_VALUES_KEY], new_values.size
    ])

    intermediate_normalization_dict[MEAN_VALUE_KEY] = numpy.average(
        these_means, weights=these_weights)

    these_means = numpy.array([
        intermediate_normalization_dict[MEAN_OF_SQUARES_KEY],
        numpy.mean(new_values ** 2)
    ])

    intermediate_normalization_dict[MEAN_OF_SQUARES_KEY] = numpy.average(
        these_means, weights=these_weights)

    intermediate_normalization_dict[NUM_VALUES_KEY] += new_values.size
    return intermediate_normalization_dict


def _get_standard_deviation(intermediate_normalization_dict):
    """Computes stdev from intermediate normalization params.

    :param intermediate_normalization_dict: See doc for
        `_update_normalization_params`.
    :return: standard_deviation: Standard deviation.
    """

    num_values = float(intermediate_normalization_dict[NUM_VALUES_KEY])
    multiplier = num_values / (num_values - 1)

    return numpy.sqrt(multiplier * (
        intermediate_normalization_dict[MEAN_OF_SQUARES_KEY] -
        intermediate_normalization_dict[MEAN_VALUE_KEY] ** 2
    ))


def get_image_normalization_params(netcdf_file_names):
    """Computes normalization params (mean and stdev) for each predictor.

    :param netcdf_file_names: 1-D list of paths to input files.
    :return: normalization_dict: See input doc for `normalize_images`.
    """

    predictor_names = None
    norm_dict_by_predictor = None

    for this_file_name in netcdf_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_image_dict = read_image_file(this_file_name)

        if predictor_names is None:
            predictor_names = this_image_dict[PREDICTOR_NAMES_KEY]
            norm_dict_by_predictor = [{}] * len(predictor_names)

        for m in range(len(predictor_names)):
            norm_dict_by_predictor[m] = _update_normalization_params(
                intermediate_normalization_dict=norm_dict_by_predictor[m],
                new_values=this_image_dict[PREDICTOR_MATRIX_KEY][..., m])

    print('\n')
    normalization_dict = {}

    for m in range(len(predictor_names)):
        this_mean = norm_dict_by_predictor[m][MEAN_VALUE_KEY]
        this_stdev = _get_standard_deviation(norm_dict_by_predictor[m])
        normalization_dict[predictor_names[m]] = numpy.array(
            [this_mean, this_stdev])

        print(
            ('Mean and standard deviation for "{0:s}" = {1:.4f}, {2:.4f}'
             ).format(predictor_names[m], this_mean, this_stdev)
        )

    return normalization_dict


def get_norm_params_example(training_file_names):
    """Gets normalization parameters.

    :param training_file_names: 1-D list of paths to input files.
    """

    normalization_dict = get_image_normalization_params(training_file_names)


def normalize_images(
        predictor_matrix, predictor_names, normalization_dict=None):
    """Normalizes images to z-scores.

    E = number of examples (storm objects) in file
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)

    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param normalization_dict: Dictionary.  Each key is the name of a predictor
        value, and the corresponding value is a length-2 numpy array with
        [mean, standard deviation].  If `normalization_dict is None`, mean and
        standard deviation will be computed for each predictor.
    :return: predictor_matrix: Normalized version of input.
    :return: normalization_dict: See doc for input variable.  If input was None,
        this will be a newly created dictionary.  Otherwise, this will be the
        same dictionary passed as input.
    """

    num_predictors = len(predictor_names)

    if normalization_dict is None:
        normalization_dict = {}

        for m in range(num_predictors):
            this_mean = numpy.mean(predictor_matrix[..., m])
            this_stdev = numpy.std(predictor_matrix[..., m], ddof=1)

            normalization_dict[predictor_names[m]] = numpy.array(
                [this_mean, this_stdev])

    for m in range(num_predictors):
        this_mean = normalization_dict[predictor_names[m]][0]
        this_stdev = normalization_dict[predictor_names[m]][1]

        predictor_matrix[..., m] = (
            (predictor_matrix[..., m] - this_mean) / float(this_stdev)
        )

    return predictor_matrix, normalization_dict


def denormalize_images(predictor_matrix, predictor_names, normalization_dict):
    """Denormalizes images from z-scores back to original scales.

    :param predictor_matrix: See doc for `normalize_images`.
    :param predictor_names: Same.
    :param normalization_dict: Same.
    :return: predictor_matrix: Denormalized version of input.
    """

    num_predictors = len(predictor_names)
    for m in range(num_predictors):
        this_mean = normalization_dict[predictor_names[m]][0]
        this_stdev = normalization_dict[predictor_names[m]][1]

        predictor_matrix[..., m] = (
            this_mean + this_stdev * predictor_matrix[..., m]
        )

    return predictor_matrix


def norm_denorm_example(training_file_names, normalization_dict):
    """Normalizes and denormalizes images.

    :param training_file_names: 1-D list of paths to input files.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    """

    image_dict = read_image_file(training_file_names[0])

    predictor_names = image_dict[PREDICTOR_NAMES_KEY]
    these_predictor_values = image_dict[PREDICTOR_MATRIX_KEY][0, :5, :5, 0]

    print('\nOriginal values of "{0:s}" for first storm object:\n{1:s}'.format(
        predictor_names[0], str(these_predictor_values)
    ))

    image_dict[PREDICTOR_MATRIX_KEY], _ = normalize_images(
        predictor_matrix=image_dict[PREDICTOR_MATRIX_KEY],
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    these_predictor_values = image_dict[PREDICTOR_MATRIX_KEY][0, :5, :5, 0]
    print(
        '\nNormalized values of "{0:s}" for first storm object:\n{1:s}'.format(
            predictor_names[0], str(these_predictor_values))
    )

    image_dict[PREDICTOR_MATRIX_KEY] = denormalize_images(
        predictor_matrix=image_dict[PREDICTOR_MATRIX_KEY],
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    these_predictor_values = image_dict[PREDICTOR_MATRIX_KEY][0, :5, :5, 0]
    print(
        ('\nDenormalized values of "{0:s}" for first storm object:\n{1:s}'
         ).format(predictor_names[0], str(these_predictor_values))
    )


def get_binarization_threshold(netcdf_file_names, percentile_level):
    """Computes binarization threshold for target variable.

    Binarization threshold will be [q]th percentile of all image maxima, where
    q = `percentile_level`.

    :param netcdf_file_names: 1-D list of paths to input files.
    :param percentile_level: q in the above discussion.
    :return: binarization_threshold: Binarization threshold (used to turn each
        target image into a yes-or-no label).
    """

    max_target_values = numpy.array([])

    for this_file_name in netcdf_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_image_dict = read_image_file(this_file_name)

        this_target_matrix = this_image_dict[TARGET_MATRIX_KEY]
        this_num_examples = this_target_matrix.shape[0]
        these_max_target_values = numpy.full(this_num_examples, numpy.nan)

        for i in range(this_num_examples):
            these_max_target_values[i] = numpy.max(this_target_matrix[i, ...])

        max_target_values = numpy.concatenate((
            max_target_values, these_max_target_values))

    binarization_threshold = numpy.percentile(
        max_target_values, percentile_level)

    print('\nBinarization threshold for "{0:s}" = {1:.4e}'.format(
        TARGET_NAME, binarization_threshold))

    return binarization_threshold


def find_binarization_threshold_example(training_file_names):
    """Finds binarization threshold for target variable.

    :param training_file_names: 1-D list of paths to input files.
    """

    binarization_threshold = get_binarization_threshold(
        netcdf_file_names=training_file_names, percentile_level=90.)


def binarize_target_images(target_matrix, binarization_threshold):
    """Binarizes target images.

    Specifically, this method turns each target image into a binary label,
    depending on whether or not (max value in image) >= binarization_threshold.

    E = number of examples (storm objects) in file
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid

    :param target_matrix: E-by-M-by-N numpy array of floats.
    :param binarization_threshold: Binarization threshold.
    :return: target_values: length-E numpy array of target values (integers in
        0...1).
    """

    num_examples = target_matrix.shape[0]
    target_values = numpy.full(num_examples, -1, dtype=int)

    for i in range(num_examples):
        target_values[i] = (
            numpy.max(target_matrix[i, ...]) >= binarization_threshold
        )

    return target_values


def binarization_example(training_file_names, binarization_threshold):
    """Binarizes target images.

    :param training_file_names: 1-D list of paths to input files.
    :param binarization_threshold: Binarization threshold.
    """

    image_dict = read_image_file(training_file_names[0])
    these_max_target_values = numpy.array(
        [numpy.max(image_dict[TARGET_MATRIX_KEY][i, ...]) for i in range(10)]
    )

    print(
        ('\nSpatial maxima of "{0:s}" for the first few storm objects:\n{1:s}'
         ).format(image_dict[TARGET_NAME_KEY], str(these_max_target_values))
    )

    target_values = binarize_target_images(
        target_matrix=image_dict[TARGET_MATRIX_KEY],
        binarization_threshold=binarization_threshold)

    print(
        ('\nBinarized target values for the first few storm objects:\n{0:s}'
         ).format(str(target_values[:10]))
    )


def _get_dense_layer_dimensions(num_input_units, num_classes, num_dense_layers):
    """Returns dimensions (number of input and output units) for each dense lyr.

    D = number of dense layers

    :param num_input_units: Number of input units (features created by
        flattening layer).
    :param num_classes: Number of output classes (possible values of target
        variable).
    :param num_dense_layers: Number of dense layers.
    :return: num_inputs_by_layer: length-D numpy array with number of input
        units by dense layer.
    :return: num_outputs_by_layer: length-D numpy array with number of output
        units by dense layer.
    """

    if num_classes == 2:
        num_output_units = 1
    else:
        num_output_units = num_classes + 0

    e_folding_param = (
        float(-1 * num_dense_layers) /
        numpy.log(float(num_output_units) / num_input_units)
    )

    dense_layer_indices = numpy.linspace(
        0, num_dense_layers - 1, num=num_dense_layers, dtype=float)
    num_inputs_by_layer = num_input_units * numpy.exp(
        -1 * dense_layer_indices / e_folding_param)
    num_inputs_by_layer = numpy.round(num_inputs_by_layer).astype(int)

    num_outputs_by_layer = numpy.concatenate((
        num_inputs_by_layer[1:],
        numpy.array([num_output_units], dtype=int)
    ))

    return num_inputs_by_layer, num_outputs_by_layer


def setup_cnn(num_grid_rows, num_grid_columns):
    """Sets up (but does not train) CNN (convolutional neural net).

    :param num_grid_rows: Number of rows in each predictor image.
    :param num_grid_columns: Number of columns in each predictor image.
    :return: cnn_model_object: Untrained instance of `keras.models.Model`.
    """

    regularizer_object = keras.regularizers.l1_l2(l1=L1_WEIGHT, l2=L2_WEIGHT)

    num_predictors = len(NETCDF_PREDICTOR_NAMES)
    input_layer_object = keras.layers.Input(
        shape=(num_grid_rows, num_grid_columns, num_predictors)
    )

    current_num_filters = None
    current_layer_object = None

    # Add convolutional layers.
    for _ in range(NUM_CONV_LAYER_SETS):
        for _ in range(NUM_CONV_LAYERS_PER_SET):

            if current_num_filters is None:
                current_num_filters = (
                    num_predictors * NUM_PREDICTORS_TO_FIRST_NUM_FILTERS)
                this_input_layer_object = input_layer_object

            else:
                current_num_filters *= 2
                this_input_layer_object = current_layer_object

            current_layer_object = keras.layers.Conv2D(
                filters=current_num_filters,
                kernel_size=(NUM_CONV_FILTER_ROWS, NUM_CONV_FILTER_COLUMNS),
                strides=(1, 1), padding='valid', data_format='channels_last',
                dilation_rate=(1, 1), activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object
            )(this_input_layer_object)

            current_layer_object = keras.layers.LeakyReLU(
                alpha=SLOPE_FOR_RELU
            )(current_layer_object)

            if CONV_LAYER_DROPOUT_FRACTION is not None:
                current_layer_object = keras.layers.Dropout(
                    rate=CONV_LAYER_DROPOUT_FRACTION
                )(current_layer_object)

            if USE_BATCH_NORMALIZATION:
                current_layer_object = keras.layers.BatchNormalization(
                    axis=-1, center=True, scale=True
                )(current_layer_object)

        current_layer_object = keras.layers.MaxPooling2D(
            pool_size=(NUM_POOLING_ROWS, NUM_POOLING_COLUMNS),
            strides=(NUM_POOLING_ROWS, NUM_POOLING_COLUMNS),
            padding='valid', data_format='channels_last'
        )(current_layer_object)

    these_dimensions = numpy.array(
        current_layer_object.get_shape().as_list()[1:], dtype=int)
    num_features = numpy.prod(these_dimensions)

    current_layer_object = keras.layers.Flatten()(current_layer_object)

    # Add intermediate dense layers.
    _, num_outputs_by_dense_layer = _get_dense_layer_dimensions(
        num_input_units=num_features, num_classes=2,
        num_dense_layers=NUM_DENSE_LAYERS)

    for k in range(NUM_DENSE_LAYERS - 1):
        current_layer_object = keras.layers.Dense(
            num_outputs_by_dense_layer[k], activation=None, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=regularizer_object
        )(current_layer_object)

        current_layer_object = keras.layers.LeakyReLU(
            alpha=SLOPE_FOR_RELU
        )(current_layer_object)

        if DENSE_LAYER_DROPOUT_FRACTION is not None:
            current_layer_object = keras.layers.Dropout(
                rate=DENSE_LAYER_DROPOUT_FRACTION
            )(current_layer_object)

        if USE_BATCH_NORMALIZATION:
            current_layer_object = keras.layers.BatchNormalization(
                axis=-1, center=True, scale=True
            )(current_layer_object)

    # Add output layer (also dense).
    current_layer_object = keras.layers.Dense(
        1, activation=None, use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=regularizer_object
    )(current_layer_object)

    current_layer_object = keras.layers.Activation(
        'sigmoid'
    )(current_layer_object)

    if DENSE_LAYER_DROPOUT_FRACTION is not None and NUM_DENSE_LAYERS == 1:
        current_layer_object = keras.layers.Dropout(
            rate=DENSE_LAYER_DROPOUT_FRACTION
        )(current_layer_object)

    # Put the whole thing together and compile.
    cnn_model_object = keras.models.Model(
        inputs=input_layer_object, outputs=current_layer_object)
    cnn_model_object.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=LIST_OF_METRIC_FUNCTIONS)

    cnn_model_object.summary()
    return cnn_model_object


def setup_cnn_example(training_file_names):
    """Sets up CNN.

    :param training_file_names: 1-D list of paths to input files.
    """

    this_image_dict = read_image_file(training_file_names[0])
    cnn_model_object = setup_cnn(
        num_grid_rows=this_image_dict[PREDICTOR_MATRIX_KEY].shape[1],
        num_grid_columns=this_image_dict[PREDICTOR_MATRIX_KEY].shape[2])


def deep_learning_generator(netcdf_file_names, num_examples_per_batch,
                            normalization_dict, binarization_threshold):
    """Generates training examples for deep-learning model on the fly.

    E = number of examples (storm objects)
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)

    :param netcdf_file_names: 1-D list of paths to input (NetCDF) files.
    :param num_examples_per_batch: Number of examples per training batch.
    :param normalization_dict: See doc for `normalize_images`.  You cannot leave
        this as None.
    :param binarization_threshold: Binarization threshold for target variable.
        See `binarize_target_images` for details on what this does.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :return: target_values: length-E numpy array of target values (integers in
        0...1).
    :raises: TypeError: if `normalization_dict is None`.
    """

    # TODO(thunderhoser): Maybe add upsampling or downsampling.

    if normalization_dict is None:
        error_string = 'normalization_dict cannot be None.  Must be specified.'
        raise TypeError(error_string)

    random.shuffle(netcdf_file_names)
    num_files = len(netcdf_file_names)
    file_index = 0

    num_examples_in_memory = 0
    full_predictor_matrix = None
    full_target_matrix = None
    predictor_names = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print('Reading data from: "{0:s}"...'.format(
                netcdf_file_names[file_index]))

            this_image_dict = read_image_file(netcdf_file_names[file_index])
            predictor_names = this_image_dict[PREDICTOR_NAMES_KEY]

            file_index += 1
            if file_index >= num_files:
                file_index = 0

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_predictor_matrix = (
                    this_image_dict[PREDICTOR_MATRIX_KEY] + 0.
                )
                full_target_matrix = this_image_dict[TARGET_MATRIX_KEY] + 0.

            else:
                full_predictor_matrix = numpy.concatenate(
                    (full_predictor_matrix,
                     this_image_dict[PREDICTOR_MATRIX_KEY]),
                    axis=0)

                full_target_matrix = numpy.concatenate(
                    (full_target_matrix, this_image_dict[TARGET_MATRIX_KEY]),
                    axis=0)

            num_examples_in_memory = full_target_matrix.shape[0]

        batch_indices = numpy.linspace(
            0, num_examples_in_memory - 1, num=num_examples_in_memory,
            dtype=int)
        batch_indices = numpy.random.choice(
            batch_indices, size=num_examples_per_batch, replace=False)

        predictor_matrix, _ = normalize_images(
            predictor_matrix=full_predictor_matrix[batch_indices, ...],
            predictor_names=predictor_names,
            normalization_dict=normalization_dict)
        predictor_matrix = predictor_matrix.astype('float32')

        target_values = binarize_target_images(
            target_matrix=full_target_matrix[batch_indices, ...],
            binarization_threshold=binarization_threshold)

        print('Fraction of examples in positive class: {0:.4f}'.format(
            numpy.mean(target_values)))

        num_examples_in_memory = 0
        full_predictor_matrix = None
        full_target_matrix = None

        yield (predictor_matrix, target_values)


def train_cnn(
        cnn_model_object, training_file_names, normalization_dict,
        binarization_threshold, num_examples_per_batch, num_epochs,
        num_training_batches_per_epoch, output_model_file_name,
        validation_file_names=None, num_validation_batches_per_epoch=None):
    """Trains CNN (convolutional neural net).

    :param cnn_model_object: Untrained instance of `keras.models.Model` (may be
        created by `setup_cnn`).
    :param training_file_names: 1-D list of paths to training files (must be
        readable by `read_image_file`).
    :param normalization_dict: See doc for `deep_learning_generator`.
    :param binarization_threshold: Same.
    :param num_examples_per_batch: Same.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches furnished
        to model in each epoch.
    :param output_model_file_name: Path to output file.  The model will be saved
        as an HDF5 file (extension should be ".h5", but this is not enforced).
    :param validation_file_names: 1-D list of paths to training files (must be
        readable by `read_image_file`).  If `validation_file_names is None`,
        will omit on-the-fly validation.
    :param num_validation_batches_per_epoch:
        [used only if `validation_file_names is not None`]
        Number of validation batches furnished to model in each epoch.

    :return: cnn_metadata_dict: Dictionary with the following keys.
    cnn_metadata_dict['training_file_names']: See input doc.
    cnn_metadata_dict['normalization_dict']: Same.
    cnn_metadata_dict['binarization_threshold']: Same.
    cnn_metadata_dict['num_examples_per_batch']: Same.
    cnn_metadata_dict['num_training_batches_per_epoch']: Same.
    cnn_metadata_dict['validation_file_names']: Same.
    cnn_metadata_dict['num_validation_batches_per_epoch']: Same.
    """

    _create_directory(file_name=output_model_file_name)

    if validation_file_names is None:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=output_model_file_name, monitor='loss', verbose=1,
            save_best_only=False, save_weights_only=False, mode='min',
            period=1)
    else:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=output_model_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min',
            period=1)

    list_of_callback_objects = [checkpoint_object]

    cnn_metadata_dict = {
        TRAINING_FILES_KEY: training_file_names,
        NORMALIZATION_DICT_KEY: normalization_dict,
        BINARIZATION_THRESHOLD_KEY: binarization_threshold,
        NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        VALIDATION_FILES_KEY: validation_file_names,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch
    }

    training_generator = deep_learning_generator(
        netcdf_file_names=training_file_names,
        num_examples_per_batch=num_examples_per_batch,
        normalization_dict=normalization_dict,
        binarization_threshold=binarization_threshold)

    if validation_file_names is None:
        cnn_model_object.fit_generator(
            generator=training_generator,
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, callbacks=list_of_callback_objects, workers=0)

        return cnn_metadata_dict

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=MIN_XENTROPY_DECREASE_FOR_EARLY_STOP,
        patience=NUM_EPOCHS_FOR_EARLY_STOPPING, verbose=1, mode='min')

    list_of_callback_objects.append(early_stopping_object)

    validation_generator = deep_learning_generator(
        netcdf_file_names=validation_file_names,
        num_examples_per_batch=num_examples_per_batch,
        normalization_dict=normalization_dict,
        binarization_threshold=binarization_threshold)

    cnn_model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
        verbose=1, callbacks=list_of_callback_objects, workers=0,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch)

    return cnn_metadata_dict


def _create_directory(directory_name=None, file_name=None):
    """Creates directory (along with parents if necessary).

    This method creates directories only when necessary, so you don't have to
    worry about it overwriting anything.

    :param directory_name: Name of desired directory.
    :param file_name: [used only if `directory_name is None`]
        Path to desired file.  All directories in path will be created.
    """

    if directory_name is None:
        directory_name = os.path.split(file_name)[0]

    try:
        os.makedirs(directory_name)
    except OSError as this_error:
        if this_error.errno == errno.EEXIST and os.path.isdir(directory_name):
            pass
        else:
            raise


def read_keras_model(hdf5_file_name):
    """Reads Keras model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model`.
    """

    return keras.models.load_model(
        hdf5_file_name, custom_objects=METRIC_FUNCTION_DICT)


def find_model_metafile(model_file_name, raise_error_if_missing=False):
    """Finds metafile for machine-learning model.

    :param model_file_name: Path to file with trained model.
    :param raise_error_if_missing: Boolean flag.  If True and metafile is not
        found, this method will error out.
    :return: model_metafile_name: Path to file with metadata.  If file is not
        found and `raise_error_if_missing = False`, this will be the expected
        path.
    :raises: ValueError: if metafile is not found and
        `raise_error_if_missing = True`.
    """

    model_directory_name, pathless_model_file_name = os.path.split(
        model_file_name)
    model_metafile_name = '{0:s}/{1:s}_metadata.json'.format(
        model_directory_name, os.path.splitext(pathless_model_file_name)[0]
    )

    if not os.path.isfile(model_metafile_name) and raise_error_if_missing:
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            model_metafile_name)
        raise ValueError(error_string)

    return model_metafile_name


def _metadata_numpy_to_list(model_metadata_dict):
    """Converts numpy arrays in model metadata to lists.

    This is needed so that the metadata can be written to a JSON file (JSON does
    not handle numpy arrays).

    This method does not overwrite the original dictionary.

    :param model_metadata_dict: Dictionary created by `train_cnn` or
        `train_ucn`.
    :return: new_metadata_dict: Same but with lists instead of numpy arrays.
    """

    new_metadata_dict = copy.deepcopy(model_metadata_dict)

    if NORMALIZATION_DICT_KEY in new_metadata_dict.keys():
        this_norm_dict = new_metadata_dict[NORMALIZATION_DICT_KEY]

        for this_key in this_norm_dict.keys():
            if isinstance(this_norm_dict[this_key], numpy.ndarray):
                this_norm_dict[this_key] = this_norm_dict[this_key].tolist()

    return new_metadata_dict


def _metadata_list_to_numpy(model_metadata_dict):
    """Converts lists in model metadata to numpy arrays.

    This method is the inverse of `_metadata_numpy_to_list`.

    This method overwrites the original dictionary.

    :param model_metadata_dict: Dictionary created by `train_cnn` or
        `train_ucn`.
    :return: model_metadata_dict: Same but numpy arrays instead of lists.
    """

    if NORMALIZATION_DICT_KEY in model_metadata_dict.keys():
        this_norm_dict = model_metadata_dict[NORMALIZATION_DICT_KEY]

        for this_key in this_norm_dict.keys():
            this_norm_dict[this_key] = numpy.array(this_norm_dict[this_key])

    return model_metadata_dict


def write_model_metadata(model_metadata_dict, json_file_name):
    """Writes metadata for machine-learning model to JSON file.

    :param model_metadata_dict: Dictionary created by `train_cnn` or
        `train_ucn`.
    :param json_file_name: Path to output file.
    """

    _create_directory(file_name=json_file_name)

    new_metadata_dict = _metadata_numpy_to_list(model_metadata_dict)
    with open(json_file_name, 'w') as this_file:
        json.dump(new_metadata_dict, this_file)


def read_model_metadata(json_file_name):
    """Reads metadata for machine-learning model from JSON file.

    :param json_file_name: Path to output file.
    :return: model_metadata_dict: Dictionary with keys listed in doc for
        `train_cnn` or `train_ucn`.
    """

    with open(json_file_name) as this_file:
        model_metadata_dict = json.load(this_file)
        return _metadata_list_to_numpy(model_metadata_dict)


def train_cnn_example(
        cnn_model_object, training_file_names, validation_file_names,
        normalization_dict, binarization_threshold):
    """Actually trains the CNN.

    :param cnn_model_object: See doc for `train_cnn`.
    :param training_file_names: Same.
    :param validation_file_names: Same.
    :param normalization_dict: Same.
    :param binarization_threshold: Same.
    """

    cnn_file_name = '{0:s}/cnn_model.h5'.format(MODULE4_DIR_NAME)
    cnn_metadata_dict = train_cnn(
        cnn_model_object=cnn_model_object,
        training_file_names=training_file_names,
        normalization_dict=normalization_dict,
        binarization_threshold=binarization_threshold,
        num_examples_per_batch=256, num_epochs=10,
        num_training_batches_per_epoch=10,
        validation_file_names=validation_file_names,
        num_validation_batches_per_epoch=10,
        output_model_file_name=cnn_file_name)


def _apply_cnn(cnn_model_object, predictor_matrix, verbose=True,
               output_layer_name=None):
    """Applies trained CNN (convolutional neural net) to new data.

    E = number of examples (storm objects) in file
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)

    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param verbose: Boolean flag.  If True, progress messages will be printed.
    :param output_layer_name: Name of output layer.  If
        `output_layer_name is None`, this method will use the actual output
        layer, so will return predictions.  If `output_layer_name is not None`,
        will return "features" (outputs from the given layer).

    If `output_layer_name is None`...

    :return: forecast_probabilities: length-E numpy array with forecast
        probabilities of positive class (label = 1).

    If `output_layer_name is not None`...

    :return: feature_matrix: numpy array of features (outputs from the given
        layer).  There is no guarantee on the shape of this array, except that
        the first axis has length E.
    """

    num_examples = predictor_matrix.shape[0]
    num_examples_per_batch = 1000

    if output_layer_name is None:
        model_object_to_use = cnn_model_object
    else:
        model_object_to_use = keras.models.Model(
            inputs=cnn_model_object.input,
            outputs=cnn_model_object.get_layer(name=output_layer_name).output)

    output_array = None

    for i in range(0, num_examples, num_examples_per_batch):
        this_first_index = i
        this_last_index = min(
            [i + num_examples_per_batch - 1, num_examples - 1]
        )

        if verbose:
            print('Applying model to examples {0:d}-{1:d} of {2:d}...'.format(
                this_first_index, this_last_index, num_examples))

        these_indices = numpy.linspace(
            this_first_index, this_last_index,
            num=this_last_index - this_first_index + 1, dtype=int)

        this_output_array = model_object_to_use.predict(
            predictor_matrix[these_indices, ...],
            batch_size=num_examples_per_batch)

        if output_layer_name is None:
            this_output_array = this_output_array[:, -1]

        if output_array is None:
            output_array = this_output_array + 0.
        else:
            output_array = numpy.concatenate(
                (output_array, this_output_array), axis=0)

    return output_array


def evaluate_cnn(
        cnn_model_object, image_dict, cnn_metadata_dict, output_dir_name):
    """Evaluates trained CNN (convolutional neural net).

    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param image_dict: Dictionary created by `read_image_file` or
        `read_many_image_files`.  Should contain validation or testing data (not
        training data), but this is not enforced.
    :param cnn_metadata_dict: Dictionary created by `train_cnn`.  This will
        ensure that data in `image_dict` are processed the exact same way as the
        training data for `cnn_model_object`.
    :param output_dir_name: Path to output directory.  Figures will be saved
        here.
    """

    predictor_matrix, _ = normalize_images(
        predictor_matrix=image_dict[PREDICTOR_MATRIX_KEY] + 0.,
        predictor_names=image_dict[PREDICTOR_NAMES_KEY],
        normalization_dict=cnn_metadata_dict[NORMALIZATION_DICT_KEY])
    predictor_matrix = predictor_matrix.astype('float32')

    target_values = binarize_target_images(
        target_matrix=image_dict[TARGET_MATRIX_KEY],
        binarization_threshold=cnn_metadata_dict[BINARIZATION_THRESHOLD_KEY])

    forecast_probabilities = _apply_cnn(cnn_model_object=cnn_model_object,
                                        predictor_matrix=predictor_matrix)
    print(MINOR_SEPARATOR_STRING)

    pofd_by_threshold, pod_by_threshold = roc_curves.plot_roc_curve(
        observed_labels=target_values,
        forecast_probabilities=forecast_probabilities)

    area_under_roc_curve = scikit_learn_auc(pofd_by_threshold, pod_by_threshold)
    title_string = 'Area under ROC curve: {0:.4f}'.format(area_under_roc_curve)

    pyplot.title(title_string)
    pyplot.show()

    _create_directory(directory_name=output_dir_name)
    roc_curve_file_name = '{0:s}/roc_curve.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(roc_curve_file_name))
    pyplot.savefig(roc_curve_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    performance_diagrams.plot_performance_diagram(
        observed_labels=target_values,
        forecast_probabilities=forecast_probabilities)
    pyplot.show()

    perf_diagram_file_name = '{0:s}/performance_diagram.jpg'.format(
        output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(perf_diagram_file_name))
    pyplot.savefig(perf_diagram_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    attributes_diagrams.plot_attributes_diagram(
        observed_labels=target_values,
        forecast_probabilities=forecast_probabilities, num_bins=20)
    pyplot.show()

    attr_diagram_file_name = '{0:s}/attributes_diagram.jpg'.format(
        output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(attr_diagram_file_name))
    pyplot.savefig(attr_diagram_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


def evaluate_cnn_example(validation_image_dict):
    """Evaluates CNN on validation data.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    """

    cnn_file_name = '{0:s}/pretrained_cnn/pretrained_cnn.h5'.format(
        MODULE4_DIR_NAME)
    cnn_metafile_name = find_model_metafile(model_file_name=cnn_file_name)

    cnn_model_object = read_keras_model(cnn_file_name)
    cnn_metadata_dict = read_model_metadata(cnn_metafile_name)
    validation_dir_name = '{0:s}/validation'.format(MODULE4_DIR_NAME)

    evaluate_cnn(
        cnn_model_object=cnn_model_object, image_dict=validation_image_dict,
        cnn_metadata_dict=cnn_metadata_dict,
        output_dir_name=validation_dir_name)
    print(SEPARATOR_STRING)


def _get_binary_xentropy(target_values, forecast_probabilities):
    """Computes binary cross-entropy.

    This function satisfies the requirements for `cost_function` in the input to
    `run_permutation_test`.

    E = number of examples

    :param: target_values: length-E numpy array of target values (integer class
        labels).
    :param: forecast_probabilities: length-E numpy array with predicted
        probabilities of positive class (target value = 1).
    :return: cross_entropy: Cross-entropy.
    """

    forecast_probabilities[
        forecast_probabilities < MIN_PROBABILITY] = MIN_PROBABILITY
    forecast_probabilities[
        forecast_probabilities > MAX_PROBABILITY] = MAX_PROBABILITY

    return -1 * numpy.nanmean(
        target_values * numpy.log2(forecast_probabilities) +
        (1 - target_values) * numpy.log2(1 - forecast_probabilities)
    )


def permutation_test_for_cnn(
        cnn_model_object, image_dict, cnn_metadata_dict,
        output_pickle_file_name, cost_function=_get_binary_xentropy):
    """Runs permutation test on CNN (convolutional neural net).

    E = number of examples (storm objects)
    C = number of channels (predictor variables)

    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param image_dict: Dictionary created by `read_image_file` or
        `read_many_image_files`.  Should contain validation data (rather than
        training data), but this is not enforced.
    :param cnn_metadata_dict: Dictionary created by `train_cnn`.  This will
        ensure that data in `image_dict` are processed the exact same way as the
        training data for `cnn_model_object`.
    :param output_pickle_file_name: Path to output file.  `result_dict` (the
        output variable) will be saved here.

    :param cost_function: Cost function (used to evaluate model predictions).
        Must be negatively oriented (lower values are better).  Must have the
        following inputs and outputs.
    Input: target_values: length-E numpy array of target values (integer class
        labels).
    Input: forecast_probabilities: length-E numpy array with predicted
        probabilities of positive class (target value = 1).
    Output: cost: Scalar value.

    :return: result_dict: Dictionary with the following keys.
    result_dict['permuted_predictor_name_by_step']: length-C list with name of
        predictor permuted at each step.
    result_dict['highest_cost_by_step']: length-C numpy array with corresponding
        cost at each step.  highest_cost_by_step[m] = cost after permuting
        permuted_predictor_name_by_step[m].
    result_dict['original_cost']: Original cost (before any permutation).
    result_dict['predictor_names_step1']: length-C list of predictor names.
    result_dict['costs_step1']: length-C numpy array of corresponding costs.
        costs_step1[m] = cost after permuting only predictor_names_step1[m].
        This key and "predictor_names_step1" correspond to the Breiman version
        of the permutation test, while "permuted_predictor_name_by_step" and
        "highest_cost_by_step" correspond to the Lakshmanan version.
    """

    predictor_names = image_dict[PREDICTOR_NAMES_KEY]

    predictor_matrix, _ = normalize_images(
        predictor_matrix=image_dict[PREDICTOR_MATRIX_KEY] + 0.,
        predictor_names=image_dict[PREDICTOR_NAMES_KEY],
        normalization_dict=cnn_metadata_dict[NORMALIZATION_DICT_KEY])
    predictor_matrix = predictor_matrix.astype('float32')

    target_values = binarize_target_images(
        target_matrix=image_dict[TARGET_MATRIX_KEY],
        binarization_threshold=cnn_metadata_dict[BINARIZATION_THRESHOLD_KEY])

    # Get original cost (before permutation).
    these_probabilities = _apply_cnn(cnn_model_object=cnn_model_object,
                                     predictor_matrix=predictor_matrix)
    print(MINOR_SEPARATOR_STRING)

    original_cost = cost_function(target_values, these_probabilities)
    print('Original cost (no permutation): {0:.4e}\n'.format(original_cost))

    num_examples = len(target_values)
    remaining_predictor_names = predictor_names + []
    current_step_num = 0

    permuted_predictor_name_by_step = []
    highest_cost_by_step = []
    predictor_names_step1 = []
    costs_step1 = []

    while len(remaining_predictor_names) > 0:
        current_step_num += 1

        highest_cost = -numpy.inf
        best_predictor_name = None
        best_predictor_permuted_values = None

        for this_predictor_name in remaining_predictor_names:
            print(
                ('Trying predictor "{0:s}" at step {1:d} of permutation test...'
                 ).format(this_predictor_name, current_step_num)
            )

            this_predictor_index = predictor_names.index(this_predictor_name)
            this_predictor_matrix = predictor_matrix + 0.

            for i in range(num_examples):
                this_predictor_matrix[i, ..., this_predictor_index] = (
                    numpy.random.permutation(
                        this_predictor_matrix[i, ..., this_predictor_index])
                )

            print(MINOR_SEPARATOR_STRING)
            these_probabilities = _apply_cnn(
                cnn_model_object=cnn_model_object,
                predictor_matrix=this_predictor_matrix)
            print(MINOR_SEPARATOR_STRING)

            this_cost = cost_function(target_values, these_probabilities)
            print('Resulting cost = {0:.4e}'.format(this_cost))

            if current_step_num == 1:
                predictor_names_step1.append(this_predictor_name)
                costs_step1.append(this_cost)

            if this_cost < highest_cost:
                continue

            highest_cost = this_cost + 0.
            best_predictor_name = this_predictor_name + ''
            best_predictor_permuted_values = this_predictor_matrix[
                ..., this_predictor_index]

        permuted_predictor_name_by_step.append(best_predictor_name)
        highest_cost_by_step.append(highest_cost)

        # Remove best predictor from list.
        remaining_predictor_names.remove(best_predictor_name)

        # Leave values of best predictor permuted.
        this_predictor_index = predictor_names.index(best_predictor_name)
        predictor_matrix[
            ..., this_predictor_index] = best_predictor_permuted_values

        print('\nBest predictor = "{0:s}" ... new cost = {1:.4e}\n'.format(
            best_predictor_name, highest_cost))

    result_dict = {
        PERMUTED_PREDICTORS_KEY: permuted_predictor_name_by_step,
        HIGHEST_COSTS_KEY: numpy.array(highest_cost_by_step),
        ORIGINAL_COST_KEY: original_cost,
        STEP1_PREDICTORS_KEY: predictor_names_step1,
        STEP1_COSTS_KEY: numpy.array(costs_step1)
    }

    _create_directory(file_name=output_pickle_file_name)

    print('Writing results to: "{0:s}"...'.format(output_pickle_file_name))
    file_handle = open(output_pickle_file_name, 'wb')
    pickle.dump(result_dict, file_handle)
    file_handle.close()

    return result_dict


def permutation_test_example(cnn_model_object, validation_image_dict,
                             cnn_metadata_dict):
    """Runs the permutation test on validation data.

    :param cnn_model_object: See doc for `permutation_test_for_cnn`.
    :param validation_image_dict: Same.
    :param cnn_metadata_dict: Same.
    """

    permutation_dir_name = '{0:s}/permutation_test'.format(MODULE4_DIR_NAME)
    main_permutation_file_name = '{0:s}/permutation_results.p'.format(
        permutation_dir_name)

    permutation_dict = permutation_test_for_cnn(
        cnn_model_object=cnn_model_object, image_dict=validation_image_dict,
        cnn_metadata_dict=cnn_metadata_dict,
        output_pickle_file_name=main_permutation_file_name)


def _label_bars_in_graph(axes_object, y_coords, y_strings):
    """Labels bars in graph.

    J = number of bars

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    :param y_coords: length-J numpy array with y-coordinates of bars.
    :param y_strings: length-J list of labels.
    """

    x_min, x_max = pyplot.xlim()
    x_coord_for_text = x_min + 0.01 * (x_max - x_min)

    for j in range(len(y_coords)):
        axes_object.text(
            x_coord_for_text, y_coords[j], y_strings[j], color='k',
            horizontalalignment='left', verticalalignment='center')


def plot_breiman_results(
        result_dict, output_file_name, plot_percent_increase=False):
    """Plots results of Breiman (single-pass) permutation test.

    :param result_dict: Dictionary created by `permutation_test_for_cnn`.
    :param output_file_name: Path to output file.  Figure will be saved here.
    :param plot_percent_increase: Boolean flag.  If True, x-axis will be
        percentage of original cost (before permutation).  If False, will be
        actual cost.
    """

    cost_values = result_dict[STEP1_COSTS_KEY]
    predictor_names = result_dict[STEP1_PREDICTORS_KEY]

    sort_indices = numpy.argsort(cost_values)
    cost_values = cost_values[sort_indices]
    predictor_names = [predictor_names[k] for k in sort_indices]

    x_coords = numpy.concatenate((
        numpy.array([result_dict[ORIGINAL_COST_KEY]]), cost_values
    ))

    if plot_percent_increase:
        x_coords = 100 * x_coords / x_coords[0]

    y_strings = ['No permutation'] + predictor_names
    y_coords = numpy.linspace(
        0, len(y_strings) - 1, num=len(y_strings), dtype=float)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.barh(
        y_coords, x_coords, color=BAR_GRAPH_FACE_COLOUR,
        edgecolor=BAR_GRAPH_EDGE_COLOUR, linewidth=BAR_GRAPH_EDGE_WIDTH)

    pyplot.yticks([], [])
    pyplot.ylabel('Predictor permuted')

    if plot_percent_increase:
        pyplot.xlabel('Cost (percentage of original)')
    else:
        pyplot.xlabel('Cost')

    _label_bars_in_graph(
        axes_object=axes_object, y_coords=y_coords, y_strings=y_strings)
    pyplot.show()

    _create_directory(file_name=output_file_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


def plot_lakshmanan_results(
        result_dict, output_file_name, plot_percent_increase=False):
    """Plots results of Lakshmanan (multi-pass) permutation test.

    :param result_dict: See doc for `plot_breiman_results`.
    :param output_file_name: Same.
    :param plot_percent_increase: Same.
    """

    x_coords = numpy.concatenate((
        numpy.array([result_dict[ORIGINAL_COST_KEY]]),
        result_dict[HIGHEST_COSTS_KEY]
    ))

    if plot_percent_increase:
        x_coords = 100 * x_coords / x_coords[0]

    y_strings = ['No permutation'] + result_dict[PERMUTED_PREDICTORS_KEY]
    y_coords = numpy.linspace(
        0, len(y_strings) - 1, num=len(y_strings), dtype=float
    )[::-1]

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.barh(
        y_coords, x_coords, color=BAR_GRAPH_FACE_COLOUR,
        edgecolor=BAR_GRAPH_EDGE_COLOUR, linewidth=BAR_GRAPH_EDGE_WIDTH)

    pyplot.yticks([], [])
    pyplot.ylabel('Predictor permuted')

    if plot_percent_increase:
        pyplot.xlabel('Cost (percentage of original)')
    else:
        pyplot.xlabel('Cost')

    _label_bars_in_graph(
        axes_object=axes_object, y_coords=y_coords, y_strings=y_strings)
    pyplot.show()

    _create_directory(file_name=output_file_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


def plot_breiman_results_example(permutation_dir_name, permutation_dict):
    """Plots results of Breiman permutation test.

    :param permutation_dir_name: Name of output directory.
    :param permutation_dict: Dictionary created by `permutation_test_for_cnn`.
    """

    breiman_file_name = '{0:s}/breiman_results.jpg'.format(permutation_dir_name)
    plot_breiman_results(
        result_dict=permutation_dict, output_file_name=breiman_file_name,
        plot_percent_increase=False)


def plot_lakshmanan_results_example(permutation_dir_name, permutation_dict):
    """Plots results of Lakshmanan permutation test.

    :param permutation_dir_name: Name of output directory.
    :param permutation_dict: Dictionary created by `permutation_test_for_cnn`.
    """

    lakshmanan_file_name = '{0:s}/lakshmanan_results.jpg'.format(
        permutation_dir_name)
    plot_lakshmanan_results(
        result_dict=permutation_dict, output_file_name=lakshmanan_file_name,
        plot_percent_increase=False)


def _gradient_descent_for_bwo(
        cnn_model_object, loss_tensor, init_function_or_matrices,
        num_iterations, learning_rate):
    """Does gradient descent (the nitty-gritty part) for backwards optimization.

    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param loss_tensor: Keras tensor, defining the loss function to be
        minimized.
    :param init_function_or_matrices: Either a function or list of numpy arrays.

    If function, will be used to initialize input matrices.  See
    `create_gaussian_initializer` for an example.

    If list of numpy arrays, these are the input matrices themselves.  Matrices
    should be processed in the exact same way that training data were processed
    (e.g., normalization method).  Matrices must also be in the same order as
    training matrices, and the [q]th matrix in this list must have the same
    shape as the [q]th training matrix.

    :param num_iterations: Number of gradient-descent iterations (number of
        times that the input matrices are adjusted).
    :param learning_rate: Learning rate.  At each iteration, each input value x
        will be decremented by `learning_rate * gradient`, where `gradient` is
        the gradient of the loss function with respect to x.
    :return: list_of_optimized_input_matrices: length-T list of optimized input
        matrices (numpy arrays), where T = number of input tensors to the model.
        If the input arg `init_function_or_matrices` is a list of numpy arrays
        (rather than a function), `list_of_optimized_input_matrices` will have
        the exact same shape, just with different values.
    """

    if isinstance(cnn_model_object.input, list):
        list_of_input_tensors = cnn_model_object.input
    else:
        list_of_input_tensors = [cnn_model_object.input]

    num_input_tensors = len(list_of_input_tensors)
    list_of_gradient_tensors = K.gradients(loss_tensor, list_of_input_tensors)

    for i in range(num_input_tensors):
        list_of_gradient_tensors[i] /= K.maximum(
            K.sqrt(K.mean(list_of_gradient_tensors[i] ** 2)),
            K.epsilon()
        )

    inputs_to_loss_and_gradients = K.function(
        list_of_input_tensors + [K.learning_phase()],
        ([loss_tensor] + list_of_gradient_tensors)
    )

    if isinstance(init_function_or_matrices, list):
        list_of_optimized_input_matrices = copy.deepcopy(
            init_function_or_matrices)
    else:
        list_of_optimized_input_matrices = [None] * num_input_tensors

        for i in range(num_input_tensors):
            these_dimensions = numpy.array(
                [1] + list_of_input_tensors[i].get_shape().as_list()[1:],
                dtype=int)

            list_of_optimized_input_matrices[i] = init_function_or_matrices(
                these_dimensions)

    for j in range(num_iterations):
        these_outputs = inputs_to_loss_and_gradients(
            list_of_optimized_input_matrices + [0])

        if numpy.mod(j, 100) == 0:
            print('Loss after {0:d} of {1:d} iterations: {2:.2e}'.format(
                j, num_iterations, these_outputs[0]))

        for i in range(num_input_tensors):
            list_of_optimized_input_matrices[i] -= (
                these_outputs[i + 1] * learning_rate)

    print('Loss after {0:d} iterations: {1:.2e}'.format(
        num_iterations, these_outputs[0]))
    return list_of_optimized_input_matrices


def bwo_for_class(
        cnn_model_object, target_class, init_function_or_matrices,
        num_iterations=DEFAULT_NUM_BWO_ITERATIONS,
        learning_rate=DEFAULT_BWO_LEARNING_RATE):
    """Does backwards optimization to maximize probability of target class.

    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param target_class: Synthetic input data will be created to maximize
        probability of this class.
    :param init_function_or_matrices: See doc for `_gradient_descent_for_bwo`.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :return: list_of_optimized_input_matrices: Same.
    """

    target_class = int(numpy.round(target_class))
    num_iterations = int(numpy.round(num_iterations))

    assert target_class >= 0
    assert num_iterations > 0
    assert learning_rate > 0.
    assert  learning_rate < 1.

    num_output_neurons = (
        cnn_model_object.layers[-1].output.get_shape().as_list()[-1]
    )

    if num_output_neurons == 1:
        assert target_class <= 1

        if target_class == 1:
            loss_tensor = K.mean(
                (cnn_model_object.layers[-1].output[..., 0] - 1) ** 2
            )
        else:
            loss_tensor = K.mean(
                cnn_model_object.layers[-1].output[..., 0] ** 2
            )
    else:
        assert target_class < num_output_neurons

        loss_tensor = K.mean(
            (cnn_model_object.layers[-1].output[..., target_class] - 1) ** 2
        )

    return _gradient_descent_for_bwo(
        cnn_model_object=cnn_model_object, loss_tensor=loss_tensor,
        init_function_or_matrices=init_function_or_matrices,
        num_iterations=num_iterations, learning_rate=learning_rate)


def bwo_example1(validation_image_dict, normalization_dict, cnn_model_object):
    """Optimizes random example (storm object) for positive class.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    orig_predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    orig_predictor_matrix_norm, _ = normalize_images(
        predictor_matrix=orig_predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    orig_predictor_matrix_norm = numpy.expand_dims(
        orig_predictor_matrix_norm, axis=0)

    optimized_predictor_matrix_norm = bwo_for_class(
        cnn_model_object=cnn_model_object, target_class=1,
        init_function_or_matrices=[orig_predictor_matrix_norm]
    )[0][0, ...]

    optimized_predictor_matrix = denormalize_images(
        predictor_matrix=optimized_predictor_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (orig_predictor_matrix[..., temperature_index],
         optimized_predictor_matrix[..., temperature_index]),
        axis=0)

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=orig_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Real example (before optimization)')
    pyplot.show()

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=optimized_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Synthetic example (after optimization)')
    pyplot.show()


def bwo_example2(validation_image_dict, normalization_dict, cnn_model_object):
    """Optimizes random example (storm object) for negative class.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    orig_predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    orig_predictor_matrix_norm, _ = normalize_images(
        predictor_matrix=orig_predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    orig_predictor_matrix_norm = numpy.expand_dims(
        orig_predictor_matrix_norm, axis=0)

    optimized_predictor_matrix_norm = bwo_for_class(
        cnn_model_object=cnn_model_object, target_class=0,
        init_function_or_matrices=[orig_predictor_matrix_norm]
    )[0][0, ...]

    optimized_predictor_matrix = denormalize_images(
        predictor_matrix=optimized_predictor_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (orig_predictor_matrix[..., temperature_index],
         optimized_predictor_matrix[..., temperature_index]),
        axis=0)

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=orig_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Real example (before optimization)')
    pyplot.show()

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=optimized_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Synthetic example (after optimization)')
    pyplot.show()


def bwo_example3(validation_image_dict, normalization_dict, cnn_model_object):
    """Optimizes extreme example (storm object) for positive class.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    target_matrix_s01 = validation_image_dict[TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    orig_predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][
        example_index, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    orig_predictor_matrix_norm, _ = normalize_images(
        predictor_matrix=orig_predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    orig_predictor_matrix_norm = numpy.expand_dims(
        orig_predictor_matrix_norm, axis=0)

    optimized_predictor_matrix_norm = bwo_for_class(
        cnn_model_object=cnn_model_object, target_class=1,
        init_function_or_matrices=[orig_predictor_matrix_norm]
    )[0][0, ...]

    optimized_predictor_matrix = denormalize_images(
        predictor_matrix=optimized_predictor_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (orig_predictor_matrix[..., temperature_index],
         optimized_predictor_matrix[..., temperature_index]),
        axis=0)

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=orig_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Real example (before optimization)')
    pyplot.show()

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=optimized_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Synthetic example (after optimization)')
    pyplot.show()


def bwo_example4(validation_image_dict, normalization_dict, cnn_model_object):
    """Optimizes extreme example (storm object) for negative class.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    target_matrix_s01 = validation_image_dict[TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    orig_predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][
        example_index, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    orig_predictor_matrix_norm, _ = normalize_images(
        predictor_matrix=orig_predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    orig_predictor_matrix_norm = numpy.expand_dims(
        orig_predictor_matrix_norm, axis=0)

    optimized_predictor_matrix_norm = bwo_for_class(
        cnn_model_object=cnn_model_object, target_class=0,
        init_function_or_matrices=[orig_predictor_matrix_norm]
    )[0][0, ...]

    optimized_predictor_matrix = denormalize_images(
        predictor_matrix=optimized_predictor_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (orig_predictor_matrix[..., temperature_index],
         optimized_predictor_matrix[..., temperature_index]),
        axis=0)

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=orig_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Real example (before optimization)')
    pyplot.show()

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=optimized_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Synthetic example (after optimization)')
    pyplot.show()


def _do_saliency_calculations(
        cnn_model_object, loss_tensor, list_of_input_matrices):
    """Does the nitty-gritty part of computing saliency maps.

    T = number of input tensors to the model
    E = number of examples (storm objects)

    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param loss_tensor: Keras tensor defining the loss function.
    :param list_of_input_matrices: length-T list of numpy arrays, comprising one
        or more examples (storm objects).  list_of_input_matrices[i] must have
        the same dimensions as the [i]th input tensor to the model.
    :return: list_of_saliency_matrices: length-T list of numpy arrays,
        comprising the saliency map for each example.
        list_of_saliency_matrices[i] has the same dimensions as
        list_of_input_matrices[i] and defines the "saliency" of each value x,
        which is the gradient of the loss function with respect to x.
    """

    if isinstance(cnn_model_object.input, list):
        list_of_input_tensors = cnn_model_object.input
    else:
        list_of_input_tensors = [cnn_model_object.input]

    list_of_gradient_tensors = K.gradients(loss_tensor, list_of_input_tensors)
    num_input_tensors = len(list_of_input_tensors)

    for i in range(num_input_tensors):
        list_of_gradient_tensors[i] /= K.maximum(
            K.std(list_of_gradient_tensors[i]), K.epsilon()
        )

    inputs_to_gradients_function = K.function(
        list_of_input_tensors + [K.learning_phase()],
        list_of_gradient_tensors)

    list_of_saliency_matrices = inputs_to_gradients_function(
        list_of_input_matrices + [0])

    for i in range(num_input_tensors):
        list_of_saliency_matrices[i] *= -1

    return list_of_saliency_matrices


def saliency_for_class(cnn_model_object, target_class, list_of_input_matrices):
    """For each input example, creates saliency map for prob of given class.

    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param target_class: Saliency maps will be created for probability of this
        class.
    :param list_of_input_matrices: See doc for `_do_saliency_calculations`.
    :return: list_of_saliency_matrices: Same.
    """

    target_class = int(numpy.round(target_class))
    assert target_class >= 0

    num_output_neurons = (
        cnn_model_object.layers[-1].output.get_shape().as_list()[-1]
    )

    if num_output_neurons == 1:
        assert target_class <= 1

        if target_class == 1:
            loss_tensor = K.mean(
                (cnn_model_object.layers[-1].output[..., 0] - 1) ** 2
            )
        else:
            loss_tensor = K.mean(
                cnn_model_object.layers[-1].output[..., 0] ** 2
            )
    else:
        assert target_class < num_output_neurons

        loss_tensor = K.mean(
            (cnn_model_object.layers[-1].output[..., target_class] - 1) ** 2
        )

    return _do_saliency_calculations(
        cnn_model_object=cnn_model_object, loss_tensor=loss_tensor,
        list_of_input_matrices=list_of_input_matrices)


def plot_saliency_2d(
        saliency_matrix, axes_object, colour_map_object,
        max_absolute_contour_level, contour_interval, line_width=2):
    """Plots saliency map over 2-D grid (for one predictor).

    M = number of rows in grid
    N = number of columns in grid

    :param saliency_matrix: M-by-N numpy array of saliency values.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param max_absolute_contour_level: Max saliency to plot.  The minimum
        saliency plotted will be `-1 * max_absolute_contour_level`.
    :param max_absolute_contour_level: Max absolute saliency value to plot.  The
        min and max values, respectively, will be
        `-1 * max_absolute_contour_level` and `max_absolute_contour_level`.
    :param contour_interval: Saliency interval between successive contours.
    :param line_width: Width of contour lines.
    """

    num_grid_rows = saliency_matrix.shape[0]
    num_grid_columns = saliency_matrix.shape[1]

    x_coords_unique = numpy.linspace(
        0, num_grid_columns, num=num_grid_columns + 1, dtype=float)
    x_coords_unique = x_coords_unique[:-1]
    x_coords_unique = x_coords_unique + numpy.diff(x_coords_unique[:2]) / 2

    y_coords_unique = numpy.linspace(
        0, num_grid_rows, num=num_grid_rows + 1, dtype=float)
    y_coords_unique = y_coords_unique[:-1]
    y_coords_unique = y_coords_unique + numpy.diff(y_coords_unique[:2]) / 2

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(x_coords_unique,
                                                    y_coords_unique)

    half_num_contours = int(numpy.round(
        1 + max_absolute_contour_level / contour_interval
    ))

    # Plot positive values.
    these_contour_levels = numpy.linspace(
        0., max_absolute_contour_level, num=half_num_contours)

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, saliency_matrix,
        these_contour_levels, cmap=colour_map_object,
        vmin=numpy.min(these_contour_levels),
        vmax=numpy.max(these_contour_levels), linewidths=line_width,
        linestyles='solid', zorder=1e6)

    # Plot negative values.
    these_contour_levels = these_contour_levels[1:]

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, -saliency_matrix,
        these_contour_levels, cmap=colour_map_object,
        vmin=numpy.min(these_contour_levels),
        vmax=numpy.max(these_contour_levels), linewidths=line_width,
        linestyles='dashed', zorder=1e6)


def plot_many_saliency_maps(
        saliency_matrix, axes_objects_2d_list, colour_map_object,
        max_absolute_contour_level, contour_interval, line_width=2):
    """Plots 2-D saliency map for each predictor.

    M = number of rows in grid
    N = number of columns in grid
    C = number of predictors

    :param saliency_matrix: M-by-N-by-C numpy array of saliency values.
    :param axes_objects_2d_list: See doc for `_init_figure_panels`.
    :param colour_map_object: See doc for `plot_saliency_2d`.
    :param max_absolute_contour_level: Same.
    :param max_absolute_contour_level: Same.
    :param contour_interval: Same.
    :param line_width: Same.
    """

    num_predictors = saliency_matrix.shape[-1]
    num_panel_rows = len(axes_objects_2d_list)
    num_panel_columns = len(axes_objects_2d_list[0])

    for m in range(num_predictors):
        this_panel_row, this_panel_column = numpy.unravel_index(
            m, (num_panel_rows, num_panel_columns)
        )

        plot_saliency_2d(
            saliency_matrix=saliency_matrix[..., m],
            axes_object=axes_objects_2d_list[this_panel_row][this_panel_column],
            colour_map_object=colour_map_object,
            max_absolute_contour_level=max_absolute_contour_level,
            contour_interval=contour_interval, line_width=line_width)


def saliency_example1(validation_image_dict, normalization_dict,
                      cnn_model_object):
    """Computes saliency map for random example wrt positive-class probability.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = normalize_images(
        predictor_matrix=predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    saliency_matrix = saliency_for_class(
        cnn_model_object=cnn_model_object, target_class=1,
        list_of_input_matrices=[predictor_matrix_norm]
    )[0][0, ...]

    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    min_colour_temp_kelvins = numpy.percentile(
        predictor_matrix[..., temperature_index], 1)
    max_colour_temp_kelvins = numpy.percentile(
        predictor_matrix[..., temperature_index], 99)

    wind_indices = numpy.array([
        predictor_names.index(U_WIND_NAME), predictor_names.index(V_WIND_NAME)
    ], dtype=int)

    max_colour_wind_speed_m_s01 = numpy.percentile(
        numpy.absolute(predictor_matrix[..., wind_indices]), 99)

    _, axes_objects_2d_list = plot_many_predictors_sans_barbs(
        predictor_matrix=predictor_matrix, predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins,
        max_colour_wind_speed_m_s01=max_colour_wind_speed_m_s01)

    max_absolute_contour_level = numpy.percentile(
        numpy.absolute(saliency_matrix), 99)
    contour_interval = max_absolute_contour_level / 10

    plot_many_saliency_maps(
        saliency_matrix=saliency_matrix,
        axes_objects_2d_list=axes_objects_2d_list,
        colour_map_object=SALIENCY_COLOUR_MAP_OBJECT,
        max_absolute_contour_level=max_absolute_contour_level,
        contour_interval=contour_interval)

    pyplot.show()


def saliency_example2(validation_image_dict, normalization_dict,
                      cnn_model_object):
    """Computes saliency map for random example wrt negative-class probability.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = normalize_images(
        predictor_matrix=predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    saliency_matrix = saliency_for_class(
        cnn_model_object=cnn_model_object, target_class=0,
        list_of_input_matrices=[predictor_matrix_norm]
    )[0][0, ...]

    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    min_colour_temp_kelvins = numpy.percentile(
        predictor_matrix[..., temperature_index], 1)
    max_colour_temp_kelvins = numpy.percentile(
        predictor_matrix[..., temperature_index], 99)

    wind_indices = numpy.array([
        predictor_names.index(U_WIND_NAME), predictor_names.index(V_WIND_NAME)
    ], dtype=int)

    max_colour_wind_speed_m_s01 = numpy.percentile(
        numpy.absolute(predictor_matrix[..., wind_indices]), 99)

    _, axes_objects_2d_list = plot_many_predictors_sans_barbs(
        predictor_matrix=predictor_matrix, predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins,
        max_colour_wind_speed_m_s01=max_colour_wind_speed_m_s01)

    max_absolute_contour_level = numpy.percentile(
        numpy.absolute(saliency_matrix), 99)
    contour_interval = max_absolute_contour_level / 10

    plot_many_saliency_maps(
        saliency_matrix=saliency_matrix,
        axes_objects_2d_list=axes_objects_2d_list,
        colour_map_object=SALIENCY_COLOUR_MAP_OBJECT,
        max_absolute_contour_level=max_absolute_contour_level,
        contour_interval=contour_interval)

    pyplot.show()


def saliency_example3(validation_image_dict, normalization_dict,
                      cnn_model_object):
    """Computes saliency map for extreme example wrt positive-class probability.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    target_matrix_s01 = validation_image_dict[TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][
        example_index, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = normalize_images(
        predictor_matrix=predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    saliency_matrix = saliency_for_class(
        cnn_model_object=cnn_model_object, target_class=1,
        list_of_input_matrices=[predictor_matrix_norm]
    )[0][0, ...]

    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    min_colour_temp_kelvins = numpy.percentile(
        predictor_matrix[..., temperature_index], 1)
    max_colour_temp_kelvins = numpy.percentile(
        predictor_matrix[..., temperature_index], 99)

    wind_indices = numpy.array([
        predictor_names.index(U_WIND_NAME), predictor_names.index(V_WIND_NAME)
    ], dtype=int)

    max_colour_wind_speed_m_s01 = numpy.percentile(
        numpy.absolute(predictor_matrix[..., wind_indices]), 99)

    _, axes_objects_2d_list = plot_many_predictors_sans_barbs(
        predictor_matrix=predictor_matrix, predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins,
        max_colour_wind_speed_m_s01=max_colour_wind_speed_m_s01)

    max_absolute_contour_level = numpy.percentile(
        numpy.absolute(saliency_matrix), 99)
    contour_interval = max_absolute_contour_level / 10

    plot_many_saliency_maps(
        saliency_matrix=saliency_matrix,
        axes_objects_2d_list=axes_objects_2d_list,
        colour_map_object=SALIENCY_COLOUR_MAP_OBJECT,
        max_absolute_contour_level=max_absolute_contour_level,
        contour_interval=contour_interval)

    pyplot.show()


def saliency_example4(validation_image_dict, normalization_dict,
                      cnn_model_object):
    """Computes saliency map for extreme example wrt negative-class probability.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    target_matrix_s01 = validation_image_dict[TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][
        example_index, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = normalize_images(
        predictor_matrix=predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    saliency_matrix = saliency_for_class(
        cnn_model_object=cnn_model_object, target_class=0,
        list_of_input_matrices=[predictor_matrix_norm]
    )[0][0, ...]

    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    min_colour_temp_kelvins = numpy.percentile(
        predictor_matrix[..., temperature_index], 1)
    max_colour_temp_kelvins = numpy.percentile(
        predictor_matrix[..., temperature_index], 99)

    wind_indices = numpy.array([
        predictor_names.index(U_WIND_NAME), predictor_names.index(V_WIND_NAME)
    ], dtype=int)

    max_colour_wind_speed_m_s01 = numpy.percentile(
        numpy.absolute(predictor_matrix[..., wind_indices]), 99)

    _, axes_objects_2d_list = plot_many_predictors_sans_barbs(
        predictor_matrix=predictor_matrix, predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins,
        max_colour_wind_speed_m_s01=max_colour_wind_speed_m_s01)

    max_absolute_contour_level = numpy.percentile(
        numpy.absolute(saliency_matrix), 99)
    contour_interval = max_absolute_contour_level / 10

    plot_many_saliency_maps(
        saliency_matrix=saliency_matrix,
        axes_objects_2d_list=axes_objects_2d_list,
        colour_map_object=SALIENCY_COLOUR_MAP_OBJECT,
        max_absolute_contour_level=max_absolute_contour_level,
        contour_interval=contour_interval)

    pyplot.show()


def _create_smoothing_filter(
        smoothing_radius_px, num_half_filter_rows, num_half_filter_columns,
        num_channels):
    """Creates convolution filter for Gaussian smoothing.

    M = number of rows in filter
    N = number of columns in filter
    C = number of channels (or "variables" or "features") to smooth.  Each
        channel will be smoothed independently.

    :param smoothing_radius_px: e-folding radius (pixels).
    :param num_half_filter_rows: Number of rows in one half of filter.  Total
        number of rows will be 2 * `num_half_filter_rows` + 1.
    :param num_half_filter_columns: Same but for columns.
    :param num_channels: C in the above discussion.
    :return: weight_matrix: M-by-N-by-C-by-C numpy array of convolution weights.
    """

    num_filter_rows = 2 * num_half_filter_rows + 1
    num_filter_columns = 2 * num_half_filter_columns + 1

    row_offsets_unique = numpy.linspace(
        -num_half_filter_rows, num_half_filter_rows, num=num_filter_rows,
        dtype=float)
    column_offsets_unique = numpy.linspace(
        -num_half_filter_columns, num_half_filter_columns,
        num=num_filter_columns, dtype=float)

    column_offset_matrix, row_offset_matrix = numpy.meshgrid(
        column_offsets_unique, row_offsets_unique)

    pixel_offset_matrix = numpy.sqrt(
        row_offset_matrix ** 2 + column_offset_matrix ** 2)

    small_weight_matrix = numpy.exp(
        -pixel_offset_matrix ** 2 / (2 * smoothing_radius_px ** 2)
    )
    small_weight_matrix = small_weight_matrix / numpy.sum(small_weight_matrix)

    weight_matrix = numpy.zeros(
        (num_filter_rows, num_filter_columns, num_channels, num_channels)
    )

    for k in range(num_channels):
        weight_matrix[..., k, k] = small_weight_matrix

    return weight_matrix


def setup_ucn(
        num_input_features, first_num_rows, first_num_columns,
        upsampling_factors, num_output_channels,
        use_activation_for_out_layer=False, use_bn_for_out_layer=True,
        use_transposed_conv=False, smoothing_radius_px=None):
    """Creates (but does not train) upconvnet.

    L = number of conv or deconv layers

    :param num_input_features: Number of input features.
    :param first_num_rows: Number of rows in input to first deconv layer.  The
        input features will be reshaped into a grid with this many rows.
    :param first_num_columns: Same but for columns.
    :param upsampling_factors: length-L numpy array of upsampling factors.  Must
        all be positive integers.
    :param num_output_channels: Number of channels in output images.
    :param use_activation_for_out_layer: Boolean flag.  If True, activation will
        be applied to output layer.
    :param use_bn_for_out_layer: Boolean flag.  If True, batch normalization
        will be applied to output layer.
    :param use_transposed_conv: Boolean flag.  If True, upsampling will be done
        with transposed-convolution layers.  If False, each upsampling will be
        done with an upsampling layer followed by a conv layer.
    :param smoothing_radius_px: Smoothing radius (pixels).  Gaussian smoothing
        with this e-folding radius will be done after each upsampling.  If
        `smoothing_radius_px is None`, no smoothing will be done.
    :return: ucn_model_object: Untrained instance of `keras.models.Model`.
    """

    if smoothing_radius_px is not None:
        num_half_smoothing_rows = int(numpy.round(
            (NUM_SMOOTHING_FILTER_ROWS - 1) / 2
        ))
        num_half_smoothing_columns = int(numpy.round(
            (NUM_SMOOTHING_FILTER_COLUMNS - 1) / 2
        ))

    regularizer_object = keras.regularizers.l1_l2(l1=L1_WEIGHT, l2=L2_WEIGHT)
    input_layer_object = keras.layers.Input(shape=(num_input_features,))

    current_num_filters = int(numpy.round(
        num_input_features / (first_num_rows * first_num_columns)
    ))

    layer_object = keras.layers.Reshape(
        target_shape=(first_num_rows, first_num_columns, current_num_filters)
    )(input_layer_object)

    num_main_layers = len(upsampling_factors)

    for i in range(num_main_layers):
        this_upsampling_factor = upsampling_factors[i]

        if i == num_main_layers - 1:
            current_num_filters = num_output_channels + 0
        elif this_upsampling_factor == 1:
            current_num_filters = int(numpy.round(current_num_filters / 2))

        if use_transposed_conv:
            if this_upsampling_factor > 1:
                this_padding_arg = 'same'
            else:
                this_padding_arg = 'valid'

            layer_object = keras.layers.Conv2DTranspose(
                filters=current_num_filters,
                kernel_size=(NUM_CONV_FILTER_ROWS, NUM_CONV_FILTER_COLUMNS),
                strides=(this_upsampling_factor, this_upsampling_factor),
                padding=this_padding_arg, data_format='channels_last',
                dilation_rate=(1, 1), activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object
            )(layer_object)

        else:
            if this_upsampling_factor > 1:
                try:
                    layer_object = keras.layers.UpSampling2D(
                        size=(this_upsampling_factor, this_upsampling_factor),
                        data_format='channels_last', interpolation='nearest'
                    )(layer_object)
                except:
                    layer_object = keras.layers.UpSampling2D(
                        size=(this_upsampling_factor, this_upsampling_factor),
                        data_format='channels_last'
                    )(layer_object)

            layer_object = keras.layers.Conv2D(
                filters=current_num_filters,
                kernel_size=(NUM_CONV_FILTER_ROWS, NUM_CONV_FILTER_COLUMNS),
                strides=(1, 1), padding='same', data_format='channels_last',
                dilation_rate=(1, 1), activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object
            )(layer_object)

            if this_upsampling_factor == 1:
                layer_object = keras.layers.ZeroPadding2D(
                    padding=(1, 1), data_format='channels_last'
                )(layer_object)

        if smoothing_radius_px is not None:
            this_weight_matrix = _create_smoothing_filter(
                smoothing_radius_px=smoothing_radius_px,
                num_half_filter_rows=num_half_smoothing_rows,
                num_half_filter_columns=num_half_smoothing_columns,
                num_channels=current_num_filters)

            this_bias_vector = numpy.zeros(current_num_filters)

            layer_object = keras.layers.Conv2D(
                filters=current_num_filters,
                kernel_size=(NUM_SMOOTHING_FILTER_ROWS,
                             NUM_SMOOTHING_FILTER_COLUMNS),
                strides=(1, 1), padding='same', data_format='channels_last',
                dilation_rate=(1, 1), activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object, trainable=False,
                weights=[this_weight_matrix, this_bias_vector]
            )(layer_object)

        if i < num_main_layers - 1 or use_activation_for_out_layer:
            layer_object = keras.layers.LeakyReLU(
                alpha=SLOPE_FOR_RELU
            )(layer_object)

        if i < num_main_layers - 1 or use_bn_for_out_layer:
            layer_object = keras.layers.BatchNormalization(
                axis=-1, center=True, scale=True
            )(layer_object)

    ucn_model_object = keras.models.Model(
        inputs=input_layer_object, outputs=layer_object)
    ucn_model_object.compile(
        loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam())

    ucn_model_object.summary()
    return ucn_model_object


def get_cnn_flatten_layer(cnn_model_object):
    """Finds flattening layer in CNN.

    This method assumes that there is only one flattening layer.  If there are
    several, this method will return the first (shallowest).

    :param cnn_model_object: Instance of `keras.models.Model`.
    :return: layer_name: Name of flattening layer.
    :raises: TypeError: if flattening layer cannot be found.
    """

    layer_names = [lyr.name for lyr in cnn_model_object.layers]

    flattening_flags = numpy.array(
        ['flatten' in n for n in layer_names], dtype=bool)
    flattening_indices = numpy.where(flattening_flags)[0]

    if len(flattening_indices) == 0:
        error_string = (
            'Cannot find flattening layer in model.  Layer names are listed '
            'below.\n{0:s}'
        ).format(str(layer_names))

        raise TypeError(error_string)

    return layer_names[flattening_indices[0]]


def setup_ucn_example(cnn_model_object):
    """Example of UCN architecture (with transposed conv, no smoothing).

    :param cnn_model_object: Trained CNN (instance of `keras.models.Model`).
    """

    cnn_feature_layer_name = get_cnn_flatten_layer(cnn_model_object)
    cnn_feature_layer_object = cnn_model_object.get_layer(
        name=cnn_feature_layer_name)
    cnn_feature_dimensions = numpy.array(
        cnn_feature_layer_object.input.shape[1:], dtype=int)

    num_input_features = numpy.prod(cnn_feature_dimensions)
    first_num_rows = cnn_feature_dimensions[0]
    first_num_columns = cnn_feature_dimensions[1]
    num_output_channels = numpy.array(
        cnn_model_object.input.shape[1:], dtype=int
    )[-1]

    upsampling_factors = numpy.array([2, 1, 1, 2, 1, 1], dtype=int)

    ucn_model_object = setup_ucn(
        num_input_features=num_input_features, first_num_rows=first_num_rows,
        first_num_columns=first_num_columns,
        upsampling_factors=upsampling_factors,
        num_output_channels=num_output_channels,
        use_transposed_conv=True, smoothing_radius_px=None)


def ucn_generator(netcdf_file_names, num_examples_per_batch, normalization_dict,
                  cnn_model_object, cnn_feature_layer_name):
    """Generates training examples for UCN (upconvolutional network) on the fly.

    E = number of examples (storm objects)
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)
    Z = number of scalar features (neurons in layer `cnn_feature_layer_name` of
        the CNN specified by `cnn_model_object`)

    :param netcdf_file_names: 1-D list of paths to input (NetCDF) files.
    :param num_examples_per_batch: Number of examples per training batch.
    :param normalization_dict: See doc for `normalize_images`.  You cannot leave
        this as None.
    :param cnn_model_object: Trained CNN model (instance of
        `keras.models.Model`).  This will be used to turn images stored in
        `netcdf_file_names` into scalar features.
    :param cnn_feature_layer_name: The "scalar features" will be the set of
        activations from this layer.
    :return: feature_matrix: E-by-Z numpy array of scalar features.  These are
        the "predictors" for the upconv network.
    :return: target_matrix: E-by-M-by-N-by-C numpy array of target images.
        These are the predictors for the CNN and the targets for the upconv
        network.
    :raises: TypeError: if `normalization_dict is None`.
    """

    if normalization_dict is None:
        error_string = 'normalization_dict cannot be None.  Must be specified.'
        raise TypeError(error_string)

    random.shuffle(netcdf_file_names)
    num_files = len(netcdf_file_names)
    file_index = 0

    num_examples_in_memory = 0
    full_target_matrix = None
    predictor_names = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print('Reading data from: "{0:s}"...'.format(
                netcdf_file_names[file_index]))

            this_image_dict = read_image_file(netcdf_file_names[file_index])
            predictor_names = this_image_dict[PREDICTOR_NAMES_KEY]

            file_index += 1
            if file_index >= num_files:
                file_index = 0

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_target_matrix = this_image_dict[PREDICTOR_MATRIX_KEY] + 0.
            else:
                full_target_matrix = numpy.concatenate(
                    (full_target_matrix, this_image_dict[PREDICTOR_MATRIX_KEY]),
                    axis=0)

            num_examples_in_memory = full_target_matrix.shape[0]

        batch_indices = numpy.linspace(
            0, num_examples_in_memory - 1, num=num_examples_in_memory,
            dtype=int)
        batch_indices = numpy.random.choice(
            batch_indices, size=num_examples_per_batch, replace=False)

        target_matrix, _ = normalize_images(
            predictor_matrix=full_target_matrix[batch_indices, ...],
            predictor_names=predictor_names,
            normalization_dict=normalization_dict)
        target_matrix = target_matrix.astype('float32')

        feature_matrix = _apply_cnn(
            cnn_model_object=cnn_model_object, predictor_matrix=target_matrix,
            verbose=False, output_layer_name=cnn_feature_layer_name)

        num_examples_in_memory = 0
        full_target_matrix = None

        yield (feature_matrix, target_matrix)


def train_ucn(
        ucn_model_object, training_file_names, normalization_dict,
        cnn_model_object, cnn_file_name, cnn_feature_layer_name,
        num_examples_per_batch, num_epochs, num_training_batches_per_epoch,
        output_model_file_name, validation_file_names=None,
        num_validation_batches_per_epoch=None):
    """Trains UCN (upconvolutional network).

    :param ucn_model_object: Untrained instance of `keras.models.Model` (may be
        created by `setup_ucn`), representing the upconv network.
    :param training_file_names: 1-D list of paths to training files (must be
        readable by `read_image_file`).
    :param normalization_dict: See doc for `ucn_generator`.
    :param cnn_model_object: Same.
    :param cnn_file_name: Path to file with trained CNN (represented by
        `cnn_model_object`).  This is needed only for the output dictionary
        (metadata).
    :param cnn_feature_layer_name: Same.
    :param num_examples_per_batch: Same.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches furnished
        to model in each epoch.
    :param output_model_file_name: Path to output file.  The model will be saved
        as an HDF5 file (extension should be ".h5", but this is not enforced).
    :param validation_file_names: 1-D list of paths to training files (must be
        readable by `read_image_file`).  If `validation_file_names is None`,
        will omit on-the-fly validation.
    :param num_validation_batches_per_epoch:
        [used only if `validation_file_names is not None`]
        Number of validation batches furnished to model in each epoch.

    :return: ucn_metadata_dict: Dictionary with the following keys.
    ucn_metadata_dict['training_file_names']: See input doc.
    ucn_metadata_dict['normalization_dict']: Same.
    ucn_metadata_dict['cnn_file_name']: Same.
    ucn_metadata_dict['cnn_feature_layer_name']: Same.
    ucn_metadata_dict['num_examples_per_batch']: Same.
    ucn_metadata_dict['num_training_batches_per_epoch']: Same.
    ucn_metadata_dict['validation_file_names']: Same.
    ucn_metadata_dict['num_validation_batches_per_epoch']: Same.
    """

    _create_directory(file_name=output_model_file_name)

    if validation_file_names is None:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=output_model_file_name, monitor='loss', verbose=1,
            save_best_only=False, save_weights_only=False, mode='min',
            period=1)
    else:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=output_model_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min',
            period=1)

    list_of_callback_objects = [checkpoint_object]

    ucn_metadata_dict = {
        TRAINING_FILES_KEY: training_file_names,
        NORMALIZATION_DICT_KEY: normalization_dict,
        CNN_FILE_KEY: cnn_file_name,
        CNN_FEATURE_LAYER_KEY: cnn_feature_layer_name,
        NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        VALIDATION_FILES_KEY: validation_file_names,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch
    }

    training_generator = ucn_generator(
        netcdf_file_names=training_file_names,
        num_examples_per_batch=num_examples_per_batch,
        normalization_dict=normalization_dict,
        cnn_model_object=cnn_model_object,
        cnn_feature_layer_name=cnn_feature_layer_name)

    if validation_file_names is None:
        ucn_model_object.fit_generator(
            generator=training_generator,
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, callbacks=list_of_callback_objects, workers=0)

        return ucn_metadata_dict

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=MIN_MSE_DECREASE_FOR_EARLY_STOP,
        patience=NUM_EPOCHS_FOR_EARLY_STOPPING, verbose=1, mode='min')

    list_of_callback_objects.append(early_stopping_object)

    validation_generator = ucn_generator(
        netcdf_file_names=validation_file_names,
        num_examples_per_batch=num_examples_per_batch,
        normalization_dict=normalization_dict,
        cnn_model_object=cnn_model_object,
        cnn_feature_layer_name=cnn_feature_layer_name)

    ucn_model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
        verbose=1, callbacks=list_of_callback_objects, workers=0,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch)

    return ucn_metadata_dict


def train_ucn_example(ucn_model_object, training_file_names, normalization_dict,
                      cnn_model_object, cnn_file_name):
    """Actually trains the UCN (upconvolutional network).

    :param ucn_model_object: See doc for `train_ucn`.
    :param training_file_names: Same.
    :param normalization_dict: Same.
    :param cnn_model_object: See doc for `cnn_model_object` in `train_ucn`.
    :param cnn_file_name: See doc for `train_ucn`.
    """

    validation_file_names = find_many_image_files(
        first_date_string='20150101', last_date_string='20151231')

    ucn_file_name = '{0:s}/ucn_model.h5'.format(MODULE4_DIR_NAME)
    ucn_metadata_dict = train_ucn(
        ucn_model_object=ucn_model_object,
        training_file_names=training_file_names,
        normalization_dict=normalization_dict,
        cnn_model_object=cnn_model_object, cnn_file_name=cnn_file_name,
        cnn_feature_layer_name=get_cnn_flatten_layer(cnn_model_object),
        num_examples_per_batch=100, num_epochs=10,
        num_training_batches_per_epoch=10, output_model_file_name=ucn_file_name,
        validation_file_names=validation_file_names,
        num_validation_batches_per_epoch=10)


def apply_ucn_example1(
        validation_image_dict, normalization_dict, cnn_model_object):
    """Uses upconvnet to reconstruct random validation example.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`,
        representing the CNN that goes with the upconvnet.
    """

    ucn_file_name = '{0:s}/pretrained_cnn/pretrained_ucn.h5'.format(
        MODULE4_DIR_NAME)
    ucn_metafile_name = find_model_metafile(model_file_name=ucn_file_name)

    ucn_model_object = read_keras_model(ucn_file_name)
    ucn_metadata_dict = read_model_metadata(ucn_metafile_name)

    image_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    image_matrix_norm, _ = normalize_images(
        predictor_matrix=image_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    image_matrix_norm = numpy.expand_dims(image_matrix_norm, axis=0)

    feature_matrix = _apply_cnn(
        cnn_model_object=cnn_model_object, predictor_matrix=image_matrix_norm,
        output_layer_name=get_cnn_flatten_layer(cnn_model_object),
        verbose=False)

    reconstructed_image_matrix_norm = ucn_model_object.predict(
        feature_matrix, batch_size=1)

    reconstructed_image_matrix = denormalize_images(
        predictor_matrix=reconstructed_image_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict
    )[0, ...]

    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (image_matrix[..., temperature_index],
         reconstructed_image_matrix[..., temperature_index]),
        axis=0)

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=image_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Original image (CNN input)')
    pyplot.show()

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=reconstructed_image_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Reconstructed image (upconvnet output)')
    pyplot.show()


def apply_ucn_example2(
        validation_image_dict, normalization_dict, ucn_model_object,
        cnn_model_object):
    """Uses upconvnet to reconstruct extreme validation example.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param ucn_model_object: Trained instance of `keras.models.Model`,
        representing the upconvnet.
    :param cnn_model_object: Trained instance of `keras.models.Model`,
        representing the CNN that goes with the upconvnet.
    """

    target_matrix_s01 = validation_image_dict[TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    image_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][
        example_index, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    image_matrix_norm, _ = normalize_images(
        predictor_matrix=image_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    image_matrix_norm = numpy.expand_dims(image_matrix_norm, axis=0)

    feature_matrix = _apply_cnn(
        cnn_model_object=cnn_model_object, predictor_matrix=image_matrix_norm,
        output_layer_name=get_cnn_flatten_layer(cnn_model_object),
        verbose=False)

    reconstructed_image_matrix_norm = ucn_model_object.predict(
        feature_matrix, batch_size=1)

    reconstructed_image_matrix = denormalize_images(
        predictor_matrix=reconstructed_image_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict
    )[0, ...]

    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (image_matrix[..., temperature_index],
         reconstructed_image_matrix[..., temperature_index]),
        axis=0)

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=image_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Original image (CNN input)')
    pyplot.show()

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=reconstructed_image_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Reconstructed image (upconvnet output)')
    pyplot.show()


def _normalize_features(feature_matrix, feature_means=None,
                        feature_standard_deviations=None):
    """Normalizes scalar features to z-scores.

    E = number of examples (storm objects)
    Z = number of features

    :param feature_matrix: E-by-Z numpy array of features.
    :param feature_means: length-Z numpy array of mean values.  If
        `feature_means is None`, these will be computed on the fly from
        `feature_matrix`.
    :param feature_standard_deviations: Same but with standard deviations.
    :return: feature_matrix: Normalized version of input.
    :return: feature_means: See input doc.
    :return: feature_standard_deviations: See input doc.
    """

    if feature_means is None or feature_standard_deviations is None:
        feature_means = numpy.mean(feature_matrix, axis=0)
        feature_standard_deviations = numpy.std(feature_matrix, axis=0, ddof=1)

    num_examples = feature_matrix.shape[0]
    num_features = feature_matrix.shape[1]

    mean_matrix = numpy.reshape(feature_means, (1, num_features))
    mean_matrix = numpy.repeat(mean_matrix, repeats=num_examples, axis=0)

    stdev_matrix = numpy.reshape(feature_standard_deviations, (1, num_features))
    stdev_matrix = numpy.repeat(stdev_matrix, repeats=num_examples, axis=0)

    feature_matrix = (feature_matrix - mean_matrix) / stdev_matrix
    return feature_matrix, feature_means, feature_standard_deviations


def _fit_svd(baseline_feature_matrix, test_feature_matrix,
             percent_variance_to_keep):
    """Fits SVD (singular-value decomposition) model.

    B = number of baseline examples (storm objects)
    T = number of testing examples (storm objects)
    Z = number of scalar features (produced by dense layer of a CNN)
    K = number of modes (top eigenvectors) retained

    The SVD model will be fit only to the baseline set, but both the baseline
    and testing sets will be used to compute normalization parameters (means and
    standard deviations).  Before, when only the baseline set was used to
    compute normalization params, the testing set had huge standard deviations,
    which caused the results of novelty detection to be physically unrealistic.

    :param baseline_feature_matrix: B-by-Z numpy array of features.
    :param test_feature_matrix: T-by-Z numpy array of features.
    :param percent_variance_to_keep: Percentage of variance to keep.  Determines
        how many eigenvectors (K in the above discussion) will be used in the
        SVD model.
    :return: svd_dictionary: Dictionary with the following keys.
    svd_dictionary['eof_matrix']: Z-by-K numpy array, where each column is an
        EOF (empirical orthogonal function).
    svd_dictionary['feature_means']: length-Z numpy array with mean value of
        each feature (before transformation).
    svd_dictionary['feature_standard_deviations']: length-Z numpy array with
        standard deviation of each feature (before transformation).
    """

    combined_feature_matrix = numpy.concatenate(
        (baseline_feature_matrix, test_feature_matrix), axis=0)

    combined_feature_matrix, feature_means, feature_standard_deviations = (
        _normalize_features(feature_matrix=combined_feature_matrix)
    )

    num_baseline_examples = baseline_feature_matrix.shape[0]
    baseline_feature_matrix = combined_feature_matrix[
        :num_baseline_examples, ...]

    eigenvalues, eof_matrix = numpy.linalg.svd(baseline_feature_matrix)[1:]
    eigenvalues = eigenvalues ** 2

    explained_variances = eigenvalues / numpy.sum(eigenvalues)
    cumulative_explained_variances = numpy.cumsum(explained_variances)

    fraction_of_variance_to_keep = 0.01 * percent_variance_to_keep
    num_modes_to_keep = 1 + numpy.where(
        cumulative_explained_variances >= fraction_of_variance_to_keep
    )[0][0]

    print(
        ('Number of modes required to explain {0:f}% of variance: {1:d}'
         ).format(percent_variance_to_keep, num_modes_to_keep)
    )

    return {
        EOF_MATRIX_KEY: numpy.transpose(eof_matrix)[..., :num_modes_to_keep],
        FEATURE_MEANS_KEY: feature_means,
        FEATURE_STDEVS_KEY: feature_standard_deviations
    }


def _apply_svd(feature_vector, svd_dictionary):
    """Applies SVD (singular-value decomposition) model to new example.

    Z = number of features

    :param feature_vector: length-Z numpy array with feature values for one
        example (storm object).
    :param svd_dictionary: Dictionary created by `_fit_svd`.
    :return: reconstructed_feature_vector: Reconstructed version of input.
    """

    this_matrix = numpy.dot(
        svd_dictionary[EOF_MATRIX_KEY],
        numpy.transpose(svd_dictionary[EOF_MATRIX_KEY])
    )

    feature_vector_norm = (
        (feature_vector - svd_dictionary[FEATURE_MEANS_KEY]) /
        svd_dictionary[FEATURE_STDEVS_KEY]
    )

    reconstructed_feature_vector_norm = numpy.dot(
        this_matrix, feature_vector_norm)

    return (
        svd_dictionary[FEATURE_MEANS_KEY] +
        reconstructed_feature_vector_norm * svd_dictionary[FEATURE_STDEVS_KEY]
    )


def do_novelty_detection(
        baseline_image_matrix, test_image_matrix, image_normalization_dict,
        predictor_names, cnn_model_object, cnn_feature_layer_name,
        ucn_model_object, num_novel_test_images,
        percent_svd_variance_to_keep=97.5):
    """Does novelty detection.

    Specifically, this method follows the procedure in Wagstaff et al. (2018)
    to determine which images in the test set are most novel with respect to the
    baseline set.

    NOTE: Both input and output images are (assumed to be) denormalized.

    B = number of baseline examples (storm objects)
    T = number of test examples (storm objects)
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)

    :param baseline_image_matrix: B-by-M-by-N-by-C numpy array of baseline
        images.
    :param test_image_matrix: T-by-M-by-N-by-C numpy array of test images.
    :param image_normalization_dict: See doc for `normalize_images`.
    :param predictor_names: length-C list of predictor names.
    :param cnn_model_object: Trained CNN model (instance of
        `keras.models.Model`).  Will be used to turn images into scalar
        features.
    :param cnn_feature_layer_name: The "scalar features" will be the set of
        activations from this layer.
    :param ucn_model_object: Trained UCN model (instance of
        `keras.models.Model`).  Will be used to turn scalar features into
        images.
    :param num_novel_test_images: Number of novel test images to find.
    :param percent_svd_variance_to_keep: See doc for `_fit_svd`.

    :return: novelty_dict: Dictionary with the following keys.  In the following
        discussion, Q = number of novel test images found.
    novelty_dict['novel_image_matrix_actual']: Q-by-M-by-N-by-C numpy array of
        novel test images.
    novelty_dict['novel_image_matrix_upconv']: Same as
        "novel_image_matrix_actual" but reconstructed by the upconvnet.
    novelty_dict['novel_image_matrix_upconv_svd']: Same as
        "novel_image_matrix_actual" but reconstructed by SVD (singular-value
        decomposition) and the upconvnet.

    :raises: TypeError: if `image_normalization_dict is None`.
    """

    if image_normalization_dict is None:
        error_string = (
            'image_normalization_dict cannot be None.  Must be specified.')
        raise TypeError(error_string)

    num_test_examples = test_image_matrix.shape[0]

    baseline_image_matrix_norm, _ = normalize_images(
        predictor_matrix=baseline_image_matrix + 0.,
        predictor_names=predictor_names,
        normalization_dict=image_normalization_dict)

    test_image_matrix_norm, _ = normalize_images(
        predictor_matrix=test_image_matrix + 0.,
        predictor_names=predictor_names,
        normalization_dict=image_normalization_dict)

    baseline_feature_matrix = _apply_cnn(
        cnn_model_object=cnn_model_object,
        predictor_matrix=baseline_image_matrix_norm, verbose=False,
        output_layer_name=cnn_feature_layer_name)

    test_feature_matrix = _apply_cnn(
        cnn_model_object=cnn_model_object,
        predictor_matrix=test_image_matrix_norm, verbose=False,
        output_layer_name=cnn_feature_layer_name)

    novel_indices = []
    novel_image_matrix_upconv = None
    novel_image_matrix_upconv_svd = None

    for k in range(num_novel_test_images):
        print('Finding {0:d}th of {1:d} novel test images...'.format(
            k + 1, num_novel_test_images))

        if len(novel_indices) == 0:
            this_baseline_feature_matrix = baseline_feature_matrix + 0.
            this_test_feature_matrix = test_feature_matrix + 0.
        else:
            novel_indices_numpy = numpy.array(novel_indices, dtype=int)
            this_baseline_feature_matrix = numpy.concatenate(
                (baseline_feature_matrix,
                 test_feature_matrix[novel_indices_numpy, ...]),
                axis=0)

            this_test_feature_matrix = numpy.delete(
                test_feature_matrix, obj=novel_indices_numpy, axis=0)

        svd_dictionary = _fit_svd(
            baseline_feature_matrix=this_baseline_feature_matrix,
            test_feature_matrix=this_test_feature_matrix,
            percent_variance_to_keep=percent_svd_variance_to_keep)

        svd_errors = numpy.full(num_test_examples, numpy.nan)
        test_feature_matrix_svd = numpy.full(
            test_feature_matrix.shape, numpy.nan)

        for i in range(num_test_examples):
            print(i)
            if i in novel_indices:
                continue

            test_feature_matrix_svd[i, ...] = _apply_svd(
                feature_vector=test_feature_matrix[i, ...],
                svd_dictionary=svd_dictionary)

            svd_errors[i] = numpy.linalg.norm(
                test_feature_matrix_svd[i, ...] - test_feature_matrix[i, ...]
            )

        new_novel_index = numpy.nanargmax(svd_errors)
        novel_indices.append(new_novel_index)

        new_image_matrix_upconv = ucn_model_object.predict(
            test_feature_matrix[[new_novel_index], ...], batch_size=1)

        new_image_matrix_upconv_svd = ucn_model_object.predict(
            test_feature_matrix_svd[[new_novel_index], ...], batch_size=1)

        if novel_image_matrix_upconv is None:
            novel_image_matrix_upconv = new_image_matrix_upconv + 0.
            novel_image_matrix_upconv_svd = new_image_matrix_upconv_svd + 0.
        else:
            novel_image_matrix_upconv = numpy.concatenate(
                (novel_image_matrix_upconv, new_image_matrix_upconv), axis=0)
            novel_image_matrix_upconv_svd = numpy.concatenate(
                (novel_image_matrix_upconv_svd, new_image_matrix_upconv_svd),
                axis=0)

    novel_indices = numpy.array(novel_indices, dtype=int)

    novel_image_matrix_upconv = denormalize_images(
        predictor_matrix=novel_image_matrix_upconv,
        predictor_names=predictor_names,
        normalization_dict=image_normalization_dict)

    novel_image_matrix_upconv_svd = denormalize_images(
        predictor_matrix=novel_image_matrix_upconv_svd,
        predictor_names=predictor_names,
        normalization_dict=image_normalization_dict)

    return {
        NOVEL_IMAGES_ACTUAL_KEY: test_image_matrix[novel_indices, ...],
        NOVEL_IMAGES_UPCONV_KEY: novel_image_matrix_upconv,
        NOVEL_IMAGES_UPCONV_SVD_KEY: novel_image_matrix_upconv_svd
    }


def _plot_novelty_for_many_predictors(
        novelty_matrix, predictor_names, max_absolute_temp_kelvins,
        max_absolute_refl_dbz):
    """Plots novelty for each predictor on 2-D grid with wind barbs overlain.

    M = number of rows in grid
    N = number of columns in grid
    C = number of predictors

    :param novelty_matrix: M-by-N-by-C numpy array of novelty values.
    :param predictor_names: length-C list of predictor names.
    :param max_absolute_temp_kelvins: Max absolute temperature in colour scheme.
        Minimum temperature in colour scheme will be
        -1 * `max_absolute_temp_kelvins`, and this will be a diverging scheme
        centered at zero.
    :param max_absolute_refl_dbz: Same but for reflectivity.
    :return: figure_object: See doc for `_init_figure_panels`.
    :return: axes_objects_2d_list: Same.
    """

    u_wind_matrix_m_s01 = novelty_matrix[
        ..., predictor_names.index(U_WIND_NAME)]
    v_wind_matrix_m_s01 = novelty_matrix[
        ..., predictor_names.index(V_WIND_NAME)]

    non_wind_predictor_names = [
        p for p in predictor_names if p not in [U_WIND_NAME, V_WIND_NAME]
    ]

    figure_object, axes_objects_2d_list = _init_figure_panels(
        num_rows=len(non_wind_predictor_names), num_columns=1)

    for m in range(len(non_wind_predictor_names)):
        this_predictor_index = predictor_names.index(
            non_wind_predictor_names[m])

        if non_wind_predictor_names[m] == REFLECTIVITY_NAME:
            this_min_colour_value = -1 * max_absolute_refl_dbz
            this_max_colour_value = max_absolute_refl_dbz + 0.
            this_colour_map_object = pyplot.cm.PuOr
        else:
            this_min_colour_value = -1 * max_absolute_temp_kelvins
            this_max_colour_value = max_absolute_temp_kelvins + 0.
            this_colour_map_object = pyplot.cm.bwr

        this_colour_bar_object = plot_predictor_2d(
            predictor_matrix=novelty_matrix[..., this_predictor_index],
            colour_map_object=this_colour_map_object, colour_norm_object=None,
            min_colour_value=this_min_colour_value,
            max_colour_value=this_max_colour_value,
            axes_object=axes_objects_2d_list[m][0])

        plot_wind_2d(u_wind_matrix_m_s01=u_wind_matrix_m_s01,
                     v_wind_matrix_m_s01=v_wind_matrix_m_s01,
                     axes_object=axes_objects_2d_list[m][0])

        this_colour_bar_object.set_label(non_wind_predictor_names[m])

    return figure_object, axes_objects_2d_list


def plot_novelty_detection(image_dict, novelty_dict, test_index):
    """Plots results of novelty detection.

    :param image_dict: Dictionary created by `read_many_image_files`, containing
        input data for novelty detection.
    :param novelty_dict: Dictionary created by `do_novelty_detection`,
        containing results.
    :param test_index: Array index.  The [i]th-most novel test example will be
        plotted, where i = `test_index`.
    """

    predictor_names = image_dict[PREDICTOR_NAMES_KEY]
    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    reflectivity_index = predictor_names.index(REFLECTIVITY_NAME)

    image_matrix_actual = novelty_dict[NOVEL_IMAGES_ACTUAL_KEY][test_index, ...]
    image_matrix_upconv = novelty_dict[NOVEL_IMAGES_UPCONV_KEY][test_index, ...]
    image_matrix_upconv_svd = novelty_dict[
        NOVEL_IMAGES_UPCONV_SVD_KEY][test_index, ...]

    combined_matrix_kelvins = numpy.concatenate(
        (image_matrix_actual[..., temperature_index],
         image_matrix_upconv[..., temperature_index]),
        axis=0)

    min_colour_temp_kelvins = numpy.percentile(combined_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_matrix_kelvins, 99)

    this_figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=image_matrix_actual, predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    base_title_string = '{0:d}th-most novel example'.format(test_index + 1)
    this_title_string = '{0:s}: actual'.format(base_title_string)
    this_figure_object.suptitle(this_title_string)
    pyplot.show()

    this_figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=image_matrix_upconv,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    this_title_string = r'{0:s}: upconvnet reconstruction'.format(
        base_title_string)
    this_title_string += r' ($\mathbf{X}_{up}$)'
    this_figure_object.suptitle(this_title_string)
    pyplot.show()

    novelty_matrix = image_matrix_upconv - image_matrix_upconv_svd
    max_absolute_temp_kelvins = numpy.percentile(
        numpy.absolute(novelty_matrix[..., temperature_index]), 99)
    max_absolute_refl_dbz = numpy.percentile(
        numpy.absolute(novelty_matrix[..., reflectivity_index]), 99)

    this_figure_object, _ = _plot_novelty_for_many_predictors(
        novelty_matrix=novelty_matrix, predictor_names=predictor_names,
        max_absolute_temp_kelvins=max_absolute_temp_kelvins,
        max_absolute_refl_dbz=max_absolute_refl_dbz)

    this_title_string = r'{0:s}: novelty'.format(
        base_title_string)
    this_title_string += r' ($\mathbf{X}_{up} - \mathbf{X}_{up,svd}$)'
    this_figure_object.suptitle(this_title_string)
    pyplot.show()


def do_novelty_detection_example(
        validation_image_dict, normalization_dict, cnn_model_object,
        ucn_model_object):
    """Runs novelty detection.

    The baseline images are a random set of 100 from the validation set, and the
    test images are the 100 storm objects with greatest vorticity in the
    validation set.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`,
        representing the CNN or "encoder".
    :param ucn_model_object: Trained instance of `keras.models.Model`,
        representing the UCN or "decoder".
    """

    target_matrix_s01 = validation_image_dict[TARGET_MATRIX_KEY]
    num_examples = target_matrix_s01.shape[0]

    max_target_by_example_s01 = numpy.array(
        [numpy.max(target_matrix_s01[i, ...]) for i in range(num_examples)]
    )

    test_indices = numpy.argsort(-1 * max_target_by_example_s01)[:100]
    test_indices = test_indices[test_indices >= 100]
    baseline_indices = numpy.linspace(0, 100, num=100, dtype=int)

    novelty_dict = do_novelty_detection(
        baseline_image_matrix=validation_image_dict[
            PREDICTOR_MATRIX_KEY][baseline_indices, ...],
        test_image_matrix=validation_image_dict[
            PREDICTOR_MATRIX_KEY][test_indices, ...],
        image_normalization_dict=normalization_dict,
        predictor_names=validation_image_dict[PREDICTOR_NAMES_KEY],
        cnn_model_object=cnn_model_object,
        cnn_feature_layer_name=get_cnn_flatten_layer(cnn_model_object),
        ucn_model_object=ucn_model_object,
        num_novel_test_images=4)


def plot_novelty_detection_example1(validation_image_dict, novelty_dict):
    """Plots first-most novel example, selon novelty detection.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param novelty_dict: Dictionary created by `do_novelty_detection`.
    """

    plot_novelty_detection(image_dict=validation_image_dict,
                           novelty_dict=novelty_dict, test_index=0)


def plot_novelty_detection_example2(validation_image_dict, novelty_dict):
    """Plots second-most novel example, selon novelty detection.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param novelty_dict: Dictionary created by `do_novelty_detection`.
    """

    plot_novelty_detection(image_dict=validation_image_dict,
                           novelty_dict=novelty_dict, test_index=1)


def plot_novelty_detection_example3(validation_image_dict, novelty_dict):
    """Plots third-most novel example, selon novelty detection.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param novelty_dict: Dictionary created by `do_novelty_detection`.
    """

    plot_novelty_detection(image_dict=validation_image_dict,
                           novelty_dict=novelty_dict, test_index=2)


def plot_novelty_detection_example4(validation_image_dict, novelty_dict):
    """Plots fourth-most novel example, selon novelty detection.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param novelty_dict: Dictionary created by `do_novelty_detection`.
    """

    plot_novelty_detection(image_dict=validation_image_dict,
                           novelty_dict=novelty_dict, test_index=3)


def _compute_gradients(loss_tensor, list_of_input_tensors):
    """Computes gradient of each input tensor with respect to loss tensor.

    :param loss_tensor: Loss tensor.
    :param list_of_input_tensors: 1-D list of input tensors.
    :return: list_of_gradient_tensors: 1-D list of gradient tensors.
    """

    list_of_gradient_tensors = tensorflow.gradients(
        loss_tensor, list_of_input_tensors)

    for i in range(len(list_of_gradient_tensors)):
        if list_of_gradient_tensors[i] is not None:
            continue

        list_of_gradient_tensors[i] = tensorflow.zeros_like(
            list_of_input_tensors[i])

    return list_of_gradient_tensors


def _normalize_tensor(input_tensor):
    """Normalizes tensor by its L2 norm.

    :param input_tensor: Unnormalized tensor.
    :return: output_tensor: Normalized tensor.
    """

    rms_tensor = K.sqrt(K.mean(K.square(input_tensor)))
    return input_tensor / (rms_tensor + K.epsilon())


def _upsample_cam(class_activation_matrix, new_dimensions):
    """Upsamples class-activation matrix (CAM).

    CAM may be 1-D, 2-D, or 3-D.

    :param class_activation_matrix: numpy array containing 1-D, 2-D, or 3-D
        class-activation matrix.
    :param new_dimensions: numpy array of new dimensions.  If matrix is
        {1D, 2D, 3D}, this must be a length-{1, 2, 3} array, respectively.
    :return: class_activation_matrix: Upsampled version of input.
    """

    num_rows_new = new_dimensions[0]
    row_indices_new = numpy.linspace(
        1, num_rows_new, num=num_rows_new, dtype=float)
    row_indices_orig = numpy.linspace(
        1, num_rows_new, num=class_activation_matrix.shape[0], dtype=float)

    if len(new_dimensions) == 1:
        interp_object = UnivariateSpline(
            x=row_indices_orig, y=numpy.ravel(class_activation_matrix),
            k=1, s=0)

        return interp_object(row_indices_new)

    num_columns_new = new_dimensions[1]
    column_indices_new = numpy.linspace(
        1, num_columns_new, num=num_columns_new, dtype=float)
    column_indices_orig = numpy.linspace(
        1, num_columns_new, num=class_activation_matrix.shape[1],
        dtype=float)

    if len(new_dimensions) == 2:
        interp_object = RectBivariateSpline(
            x=row_indices_orig, y=column_indices_orig,
            z=class_activation_matrix, kx=1, ky=1, s=0)

        return interp_object(x=row_indices_new, y=column_indices_new, grid=True)

    num_heights_new = new_dimensions[2]
    height_indices_new = numpy.linspace(
        1, num_heights_new, num=num_heights_new, dtype=float)
    height_indices_orig = numpy.linspace(
        1, num_heights_new, num=class_activation_matrix.shape[2],
        dtype=float)

    interp_object = RegularGridInterpolator(
        points=(row_indices_orig, column_indices_orig, height_indices_orig),
        values=class_activation_matrix, method='linear')

    row_index_matrix, column_index_matrix, height_index_matrix = (
        numpy.meshgrid(row_indices_new, column_indices_new, height_indices_new)
    )
    query_point_matrix = numpy.stack(
        (row_index_matrix, column_index_matrix, height_index_matrix), axis=-1)

    return interp_object(query_point_matrix)


def run_gradcam(model_object, list_of_input_matrices, target_class,
                target_layer_name):
    """Runs Grad-CAM.

    T = number of input tensors to the model

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param list_of_input_matrices: length-T list of numpy arrays, containing
        only one example (storm object).  list_of_input_matrices[i] must have
        the same dimensions as the [i]th input tensor to the model.
    :param target_class: Activation maps will be created for this class.  Must
        be an integer in 0...(K - 1), where K = number of classes.
    :param target_layer_name: Name of target layer.  Neuron-importance weights
        will be based on activations in this layer.
    :return: class_activation_matrix: Class-activation matrix.  Dimensions of
        this numpy array will be the spatial dimensions of whichever input
        tensor feeds into the target layer.  For example, if the given input
        tensor is 2-dimensional with M rows and N columns, this array will be
        M x N.
    """

    # Check input args.
    for q in range(len(list_of_input_matrices)):
        if list_of_input_matrices[q].shape[0] != 1:
            list_of_input_matrices[q] = numpy.expand_dims(
                list_of_input_matrices[q], axis=0)

    # Create loss tensor.
    output_layer_object = model_object.layers[-1].output
    num_output_neurons = output_layer_object.get_shape().as_list()[-1]

    if num_output_neurons == 1:
        if target_class == 1:
            loss_tensor = model_object.layers[-1].output[..., 0]
        else:
            loss_tensor = 1 - model_object.layers[-1].output[..., 0]
    else:
        loss_tensor = model_object.layers[-1].output[..., target_class]

    # Create gradient function.
    target_layer_activation_tensor = model_object.get_layer(
        name=target_layer_name
    ).output

    gradient_tensor = _compute_gradients(
        loss_tensor, [target_layer_activation_tensor]
    )[0]
    gradient_tensor = _normalize_tensor(gradient_tensor)

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    gradient_function = K.function(
        list_of_input_tensors, [target_layer_activation_tensor, gradient_tensor]
    )

    # Evaluate gradient function.
    target_layer_activation_matrix, gradient_matrix = gradient_function(
        list_of_input_matrices)
    target_layer_activation_matrix = target_layer_activation_matrix[0, ...]
    gradient_matrix = gradient_matrix[0, ...]

    # Compute class-activation matrix.
    mean_weight_by_filter = numpy.mean(gradient_matrix, axis=(0, 1))
    class_activation_matrix = numpy.ones(
        target_layer_activation_matrix.shape[:-1])

    num_filters = len(mean_weight_by_filter)
    for m in range(num_filters):
        class_activation_matrix += (
            mean_weight_by_filter[m] * target_layer_activation_matrix[..., m]
        )

    spatial_dimensions = numpy.array(
        list_of_input_matrices[0].shape[1:-1], dtype=int)
    class_activation_matrix = _upsample_cam(
        class_activation_matrix=class_activation_matrix,
        new_dimensions=spatial_dimensions)

    class_activation_matrix[class_activation_matrix < 0.] = 0.
    denominator = numpy.maximum(numpy.max(class_activation_matrix), K.epsilon())
    return class_activation_matrix / denominator


def gradcam_example1(validation_image_dict, normalization_dict,
                     cnn_model_object):
    """Runs Grad-CAM for random example wrt positive-class probability.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = normalize_images(
        predictor_matrix=predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    target_layer_names = [
        'batch_normalization_1', 'batch_normalization_2',
        'batch_normalization_3', 'batch_normalization_4'
    ]

    for this_layer_name in target_layer_names:
        class_activation_matrix = run_gradcam(
            model_object=cnn_model_object,
            list_of_input_matrices=[predictor_matrix_norm],
            target_class=1, target_layer_name=this_layer_name)

        temperature_index = predictor_names.index(TEMPERATURE_NAME)
        min_colour_temp_kelvins = numpy.percentile(
            predictor_matrix[..., temperature_index], 1)
        max_colour_temp_kelvins = numpy.percentile(
            predictor_matrix[..., temperature_index], 99)

        wind_indices = numpy.array([
            predictor_names.index(U_WIND_NAME),
            predictor_names.index(V_WIND_NAME)
        ], dtype=int)

        max_colour_wind_speed_m_s01 = numpy.percentile(
            numpy.absolute(predictor_matrix[..., wind_indices]), 99)

        figure_object, axes_objects_2d_list = plot_many_predictors_sans_barbs(
            predictor_matrix=predictor_matrix, predictor_names=predictor_names,
            min_colour_temp_kelvins=min_colour_temp_kelvins,
            max_colour_temp_kelvins=max_colour_temp_kelvins,
            max_colour_wind_speed_m_s01=max_colour_wind_speed_m_s01)

        dummy_saliency_matrix = numpy.expand_dims(
            class_activation_matrix, axis=-1)
        dummy_saliency_matrix = numpy.repeat(
            dummy_saliency_matrix, repeats=4, axis=-1)

        max_absolute_contour_level = numpy.percentile(
            numpy.absolute(dummy_saliency_matrix), 99)
        if max_absolute_contour_level == 0:
            max_absolute_contour_level = 10.

        contour_interval = max_absolute_contour_level / 10

        plot_many_saliency_maps(
            saliency_matrix=dummy_saliency_matrix,
            axes_objects_2d_list=axes_objects_2d_list,
            colour_map_object=SALIENCY_COLOUR_MAP_OBJECT,
            max_absolute_contour_level=max_absolute_contour_level,
            contour_interval=contour_interval)

        figure_object.suptitle(
            'Class-activation map for layer "{0:s}"'.format(this_layer_name)
        )
        pyplot.show()


def gradcam_example2(validation_image_dict, normalization_dict,
                     cnn_model_object):
    """Runs Grad-CAM for random example wrt negative-class probability.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = normalize_images(
        predictor_matrix=predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    target_layer_names = [
        'batch_normalization_1', 'batch_normalization_2',
        'batch_normalization_3', 'batch_normalization_4'
    ]

    for this_layer_name in target_layer_names:
        class_activation_matrix = run_gradcam(
            model_object=cnn_model_object,
            list_of_input_matrices=[predictor_matrix_norm],
            target_class=0, target_layer_name=this_layer_name)

        temperature_index = predictor_names.index(TEMPERATURE_NAME)
        min_colour_temp_kelvins = numpy.percentile(
            predictor_matrix[..., temperature_index], 1)
        max_colour_temp_kelvins = numpy.percentile(
            predictor_matrix[..., temperature_index], 99)

        wind_indices = numpy.array([
            predictor_names.index(U_WIND_NAME),
            predictor_names.index(V_WIND_NAME)
        ], dtype=int)

        max_colour_wind_speed_m_s01 = numpy.percentile(
            numpy.absolute(predictor_matrix[..., wind_indices]), 99)

        figure_object, axes_objects_2d_list = plot_many_predictors_sans_barbs(
            predictor_matrix=predictor_matrix, predictor_names=predictor_names,
            min_colour_temp_kelvins=min_colour_temp_kelvins,
            max_colour_temp_kelvins=max_colour_temp_kelvins,
            max_colour_wind_speed_m_s01=max_colour_wind_speed_m_s01)

        dummy_saliency_matrix = numpy.expand_dims(
            class_activation_matrix, axis=-1)
        dummy_saliency_matrix = numpy.repeat(
            dummy_saliency_matrix, repeats=4, axis=-1)

        max_absolute_contour_level = numpy.percentile(
            numpy.absolute(dummy_saliency_matrix), 99)
        contour_interval = max_absolute_contour_level / 10

        plot_many_saliency_maps(
            saliency_matrix=dummy_saliency_matrix,
            axes_objects_2d_list=axes_objects_2d_list,
            colour_map_object=SALIENCY_COLOUR_MAP_OBJECT,
            max_absolute_contour_level=max_absolute_contour_level,
            contour_interval=contour_interval)

        figure_object.suptitle(
            'Class-activation map for layer "{0:s}"'.format(this_layer_name)
        )
        pyplot.show()


def gradcam_example3(validation_image_dict, normalization_dict,
                     cnn_model_object):
    """Runs Grad-CAM for extreme example wrt positive-class probability.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    target_matrix_s01 = validation_image_dict[TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][
        example_index, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = normalize_images(
        predictor_matrix=predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    target_layer_names = [
        'batch_normalization_1', 'batch_normalization_2',
        'batch_normalization_3', 'batch_normalization_4'
    ]

    for this_layer_name in target_layer_names:
        class_activation_matrix = run_gradcam(
            model_object=cnn_model_object,
            list_of_input_matrices=[predictor_matrix_norm],
            target_class=1, target_layer_name=this_layer_name)

        temperature_index = predictor_names.index(TEMPERATURE_NAME)
        min_colour_temp_kelvins = numpy.percentile(
            predictor_matrix[..., temperature_index], 1)
        max_colour_temp_kelvins = numpy.percentile(
            predictor_matrix[..., temperature_index], 99)

        wind_indices = numpy.array([
            predictor_names.index(U_WIND_NAME),
            predictor_names.index(V_WIND_NAME)
        ], dtype=int)

        max_colour_wind_speed_m_s01 = numpy.percentile(
            numpy.absolute(predictor_matrix[..., wind_indices]), 99)

        figure_object, axes_objects_2d_list = plot_many_predictors_sans_barbs(
            predictor_matrix=predictor_matrix, predictor_names=predictor_names,
            min_colour_temp_kelvins=min_colour_temp_kelvins,
            max_colour_temp_kelvins=max_colour_temp_kelvins,
            max_colour_wind_speed_m_s01=max_colour_wind_speed_m_s01)

        dummy_saliency_matrix = numpy.expand_dims(
            class_activation_matrix, axis=-1)
        dummy_saliency_matrix = numpy.repeat(
            dummy_saliency_matrix, repeats=4, axis=-1)

        max_absolute_contour_level = numpy.percentile(
            numpy.absolute(dummy_saliency_matrix), 99)
        contour_interval = max_absolute_contour_level / 10

        plot_many_saliency_maps(
            saliency_matrix=dummy_saliency_matrix,
            axes_objects_2d_list=axes_objects_2d_list,
            colour_map_object=SALIENCY_COLOUR_MAP_OBJECT,
            max_absolute_contour_level=max_absolute_contour_level,
            contour_interval=contour_interval)

        figure_object.suptitle(
            'Class-activation map for layer "{0:s}"'.format(this_layer_name)
        )
        pyplot.show()


def gradcam_example4(validation_image_dict, normalization_dict,
                     cnn_model_object):
    """Runs Grad-CAM for extreme example wrt negative-class probability.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    target_matrix_s01 = validation_image_dict[TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][
        example_index, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = normalize_images(
        predictor_matrix=predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    target_layer_names = [
        'batch_normalization_1', 'batch_normalization_2',
        'batch_normalization_3', 'batch_normalization_4'
    ]

    for this_layer_name in target_layer_names:
        class_activation_matrix = run_gradcam(
            model_object=cnn_model_object,
            list_of_input_matrices=[predictor_matrix_norm],
            target_class=0, target_layer_name=this_layer_name)

        temperature_index = predictor_names.index(TEMPERATURE_NAME)
        min_colour_temp_kelvins = numpy.percentile(
            predictor_matrix[..., temperature_index], 1)
        max_colour_temp_kelvins = numpy.percentile(
            predictor_matrix[..., temperature_index], 99)

        wind_indices = numpy.array([
            predictor_names.index(U_WIND_NAME),
            predictor_names.index(V_WIND_NAME)
        ], dtype=int)

        max_colour_wind_speed_m_s01 = numpy.percentile(
            numpy.absolute(predictor_matrix[..., wind_indices]), 99)

        figure_object, axes_objects_2d_list = plot_many_predictors_sans_barbs(
            predictor_matrix=predictor_matrix, predictor_names=predictor_names,
            min_colour_temp_kelvins=min_colour_temp_kelvins,
            max_colour_temp_kelvins=max_colour_temp_kelvins,
            max_colour_wind_speed_m_s01=max_colour_wind_speed_m_s01)

        dummy_saliency_matrix = numpy.expand_dims(
            class_activation_matrix, axis=-1)
        dummy_saliency_matrix = numpy.repeat(
            dummy_saliency_matrix, repeats=4, axis=-1)

        max_absolute_contour_level = numpy.percentile(
            numpy.absolute(dummy_saliency_matrix), 99)
        contour_interval = max_absolute_contour_level / 10

        plot_many_saliency_maps(
            saliency_matrix=dummy_saliency_matrix,
            axes_objects_2d_list=axes_objects_2d_list,
            colour_map_object=SALIENCY_COLOUR_MAP_OBJECT,
            max_absolute_contour_level=max_absolute_contour_level,
            contour_interval=contour_interval)

        figure_object.suptitle(
            'Class-activation map for layer "{0:s}"'.format(this_layer_name)
        )
        pyplot.show()


def _register_guided_backprop():
    """Registers guided-backprop method with TensorFlow backend."""

    if (BACKPROP_FUNCTION_NAME not in
            tensorflow_ops._gradient_registry._registry):

        @tensorflow_ops.RegisterGradient(BACKPROP_FUNCTION_NAME)
        def _GuidedBackProp(operation, gradient_tensor):
            input_type = operation.inputs[0].dtype

            return (
                gradient_tensor *
                tensorflow.cast(gradient_tensor > 0., input_type) *
                tensorflow.cast(operation.inputs[0] > 0., input_type)
            )


def _change_backprop_function(model_object):
    """Changes backpropagation function for Keras model.

    :param model_object: Instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :return: new_model_object: Same as `model_object` but with new backprop
        function.
    """

    # TODO(thunderhoser): I know that "Relu" is a valid operation name, but I
    # have no clue about the last three.
    orig_to_new_operation_dict = {
        'Relu': BACKPROP_FUNCTION_NAME,
        'LeakyRelu': BACKPROP_FUNCTION_NAME,
        'Elu': BACKPROP_FUNCTION_NAME,
        'Selu': BACKPROP_FUNCTION_NAME
    }

    graph_object = tensorflow.get_default_graph()

    with graph_object.gradient_override_map(orig_to_new_operation_dict):
        new_model_object = keras.models.clone_model(model_object)
        new_model_object.set_weights(model_object.get_weights())
        new_model_object.summary()

    return new_model_object


def _make_saliency_function(model_object, layer_name):
    """Creates saliency function.

    :param model_object: Instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param layer_name: Saliency will be computed with respect to activations in
        this layer.
    :return: saliency_function: Instance of `keras.backend.function`.
    """

    output_tensor = model_object.get_layer(name=layer_name).output
    filter_maxxed_output_tensor = K.max(output_tensor, axis=-1)

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    list_of_saliency_tensors = K.gradients(
        K.sum(filter_maxxed_output_tensor), list_of_input_tensors)

    return K.function(
        list_of_input_tensors + [K.learning_phase()],
        list_of_saliency_tensors
    )


def _normalize_guided_gradcam_output(gradient_matrix):
    """Normalizes image produced by guided Grad-CAM.

    :param gradient_matrix: numpy array with output of guided Grad-CAM.
    :return: gradient_matrix: Normalized version of input.  If the first axis
        had length 1, it has been removed ("squeezed out").
    """

    if gradient_matrix.shape[0] == 1:
        gradient_matrix = gradient_matrix[0, ...]

    # Standardize.
    gradient_matrix -= numpy.mean(gradient_matrix)
    gradient_matrix /= (numpy.std(gradient_matrix, ddof=0) + K.epsilon())

    # Force standard deviation of 0.1 and mean of 0.5.
    gradient_matrix = 0.5 + gradient_matrix * 0.1
    gradient_matrix[gradient_matrix < 0.] = 0.
    gradient_matrix[gradient_matrix > 1.] = 1.

    return gradient_matrix


def run_guided_gradcam(model_object, list_of_input_matrices, target_layer_name,
                       class_activation_matrix):
    """Runs guided Grad-CAM.

    M = number of rows in grid
    N = number of columns in grid
    C = number of channels

    :param model_object: See doc for `run_gradcam`.
    :param list_of_input_matrices: Same.
    :param target_layer_name: Same.
    :param class_activation_matrix: Matrix created by `run_gradcam`.
    :return: gradient_matrix: M-by-N-by-C numpy array of gradients.
    """

    _register_guided_backprop()

    new_model_object = _change_backprop_function(model_object=model_object)
    saliency_function = _make_saliency_function(
        model_object=new_model_object, layer_name=target_layer_name)

    saliency_matrix = saliency_function(list_of_input_matrices + [0])[0]
    gradient_matrix = saliency_matrix * class_activation_matrix[
        ..., numpy.newaxis]
    return _normalize_guided_gradcam_output(gradient_matrix)


def guided_gradcam_example3(validation_image_dict, normalization_dict,
                            cnn_model_object):
    """Runs Grad-CAM for extreme example wrt positive-class probability.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    target_matrix_s01 = validation_image_dict[TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][
        example_index, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = normalize_images(
        predictor_matrix=predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    target_layer_names = [
        'batch_normalization_1', 'batch_normalization_2',
        'batch_normalization_3', 'batch_normalization_4'
    ]

    for this_layer_name in target_layer_names:
        class_activation_matrix = run_gradcam(
            model_object=cnn_model_object,
            list_of_input_matrices=[predictor_matrix_norm],
            target_class=1, target_layer_name=this_layer_name)

        gradient_matrix = run_guided_gradcam(
            model_object=cnn_model_object,
            list_of_input_matrices=[predictor_matrix_norm],
            target_layer_name=this_layer_name,
            class_activation_matrix=class_activation_matrix)

        temperature_index = predictor_names.index(TEMPERATURE_NAME)
        min_colour_temp_kelvins = numpy.percentile(
            predictor_matrix[..., temperature_index], 1)
        max_colour_temp_kelvins = numpy.percentile(
            predictor_matrix[..., temperature_index], 99)

        wind_indices = numpy.array([
            predictor_names.index(U_WIND_NAME),
            predictor_names.index(V_WIND_NAME)
        ], dtype=int)

        max_colour_wind_speed_m_s01 = numpy.percentile(
            numpy.absolute(predictor_matrix[..., wind_indices]), 99)

        figure_object, axes_objects_2d_list = plot_many_predictors_sans_barbs(
            predictor_matrix=predictor_matrix, predictor_names=predictor_names,
            min_colour_temp_kelvins=min_colour_temp_kelvins,
            max_colour_temp_kelvins=max_colour_temp_kelvins,
            max_colour_wind_speed_m_s01=max_colour_wind_speed_m_s01)

        max_absolute_contour_level = numpy.percentile(
            numpy.absolute(gradient_matrix), 99)
        contour_interval = max_absolute_contour_level / 10

        plot_many_saliency_maps(
            saliency_matrix=gradient_matrix,
            axes_objects_2d_list=axes_objects_2d_list,
            colour_map_object=SALIENCY_COLOUR_MAP_OBJECT,
            max_absolute_contour_level=max_absolute_contour_level,
            contour_interval=contour_interval)

        figure_object.suptitle(
            'Guided Grad-CAM for layer "{0:s}"'.format(this_layer_name)
        )
        pyplot.show()
