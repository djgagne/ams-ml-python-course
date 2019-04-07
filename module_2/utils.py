"""Helper methods for Module 2."""

import errno
import glob
import os.path
import pickle
import time
import calendar
import numpy
import pandas
import matplotlib.colors
import matplotlib.pyplot as pyplot
import sklearn.metrics
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
from module_4 import roc_curves
from module_4 import performance_diagrams as perf_diagrams
from module_4 import attributes_diagrams as attr_diagrams

# Directories.
MODULE2_DIR_NAME = '.'
SHORT_COURSE_DIR_NAME = '..'
DEFAULT_FEATURE_DIR_NAME = (
    '{0:s}/data/track_data_ncar_ams_3km_csv_small'
).format(SHORT_COURSE_DIR_NAME)

# Variable names.
METADATA_COLUMNS = [
    'Step_ID', 'Track_ID', 'Ensemble_Name', 'Ensemble_Member', 'Run_Date',
    'Valid_Date', 'Forecast_Hour', 'Valid_Hour_UTC'
]

EXTRANEOUS_COLUMNS = [
    'Duration', 'Centroid_Lon', 'Centroid_Lat', 'Centroid_X', 'Centroid_Y',
    'Storm_Motion_U', 'Storm_Motion_V', 'Matched', 'Max_Hail_Size',
    'Num_Matches', 'Shape', 'Location', 'Scale'
]

TARGET_NAME = 'RVORT1_MAX-future_max'
BINARIZED_TARGET_NAME = 'strong_future_rotation_flag'

NUM_VALUES_KEY = 'num_values'
MEAN_VALUE_KEY = 'mean_value'
MEAN_OF_SQUARES_KEY = 'mean_of_squares'

MAE_KEY = 'mean_absolute_error'
MSE_KEY = 'mean_squared_error'
MEAN_BIAS_KEY = 'mean_bias'
MAE_SKILL_SCORE_KEY = 'mae_skill_score'
MSE_SKILL_SCORE_KEY = 'mse_skill_score'

MAX_PEIRCE_SCORE_KEY = 'max_peirce_score'
AUC_KEY = 'area_under_roc_curve'
MAX_CSI_KEY = 'max_csi'
BRIER_SCORE_KEY = 'brier_score'
BRIER_SKILL_SCORE_KEY = 'brier_skill_score'

# Plotting constants.
DEFAULT_FIG_WIDTH_INCHES = 10
DEFAULT_FIG_HEIGHT_INCHES = 10
SMALL_FIG_WIDTH_INCHES = 10
SMALL_FIG_HEIGHT_INCHES = 10
FIGURE_RESOLUTION_DPI = 300

BAR_GRAPH_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
BAR_GRAPH_EDGE_WIDTH = 2
BAR_GRAPH_FONT_SIZE = 14
BAR_GRAPH_FONT_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

FONT_SIZE = 20
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

# Misc constants.
DATE_FORMAT = '%Y%m%d'
DATE_FORMAT_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]'

RANDOM_SEED = 6695
LAMBDA_TOLERANCE = 1e-10


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


def _remove_future_data(predictor_table):
    """Removes future data from predictors.

    :param predictor_table: pandas DataFrame with predictor values.  Each row is
        one storm object.
    :return: predictor_table: Same but with fewer columns.
    """

    predictor_names = list(predictor_table)
    columns_to_remove = [p for p in predictor_names if 'future' in p]

    return predictor_table.drop(columns_to_remove, axis=1, inplace=False)


def _feature_file_name_to_date(csv_file_name):
    """Parses date from name of feature (CSV) file.

    :param csv_file_name: Path to input file.
    :return: date_string: Date (format "yyyymmdd").
    """

    pathless_file_name = os.path.split(csv_file_name)[-1]
    date_string = pathless_file_name.replace(
        'track_step_NCARSTORM_d01_', ''
    ).replace('-0000.csv', '')

    # Verify.
    time_string_to_unix(time_string=date_string, time_format=DATE_FORMAT)
    return date_string


def find_many_feature_files(first_date_string, last_date_string,
                            feature_dir_name=DEFAULT_FEATURE_DIR_NAME):
    """Finds feature files in the given date range.

    :param first_date_string: First date ("yyyymmdd") in range.
    :param last_date_string: Last date ("yyyymmdd") in range.
    :param feature_dir_name: Name of directory with feature (CSV) files.
    :return: csv_file_names: 1-D list of paths to feature files.
    """

    first_time_unix_sec = time_string_to_unix(
        time_string=first_date_string, time_format=DATE_FORMAT)
    last_time_unix_sec = time_string_to_unix(
        time_string=last_date_string, time_format=DATE_FORMAT)

    csv_file_pattern = '{0:s}/track_step_NCARSTORM_d01_{1:s}-0000.csv'.format(
        feature_dir_name, DATE_FORMAT_REGEX)
    csv_file_names = glob.glob(csv_file_pattern)
    csv_file_names.sort()

    file_date_strings = [_feature_file_name_to_date(f) for f in csv_file_names]

    file_times_unix_sec = numpy.array([
        time_string_to_unix(time_string=d, time_format=DATE_FORMAT)
        for d in file_date_strings
    ], dtype=int)

    good_indices = numpy.where(numpy.logical_and(
        file_times_unix_sec >= first_time_unix_sec,
        file_times_unix_sec <= last_time_unix_sec
    ))[0]

    return [csv_file_names[k] for k in good_indices]


def read_feature_file(csv_file_name):
    """Reads features from CSV file.

    :param csv_file_name: Path to input file.
    :return: metadata_table: pandas DataFrame with metadata.  Each row is one
        storm object.
    :return: predictor_table: pandas DataFrame with predictor values.  Each row
        is one storm object.
    :return: target_table: pandas DataFrame with target values.  Each row is one
        storm object.
    """

    predictor_table = pandas.read_csv(csv_file_name, header=0, sep=',')
    predictor_table.drop(EXTRANEOUS_COLUMNS, axis=1, inplace=True)

    metadata_table = predictor_table[METADATA_COLUMNS]
    predictor_table.drop(METADATA_COLUMNS, axis=1, inplace=True)

    target_table = predictor_table[[TARGET_NAME]]
    predictor_table.drop([TARGET_NAME], axis=1, inplace=True)
    predictor_table = _remove_future_data(predictor_table)

    return metadata_table, predictor_table, target_table


def read_many_feature_files(csv_file_names):
    """Reads features from many CSV files.

    :param csv_file_names: 1-D list of paths to input files.
    :return: metadata_table: See doc for `read_feature_file`.
    :return: predictor_table: Same.
    :return: target_table: Same.
    """

    num_files = len(csv_file_names)
    list_of_metadata_tables = [pandas.DataFrame()] * num_files
    list_of_predictor_tables = [pandas.DataFrame()] * num_files
    list_of_target_tables = [pandas.DataFrame()] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(csv_file_names[i]))

        (list_of_metadata_tables[i], list_of_predictor_tables[i],
         list_of_target_tables[i]
        ) = read_feature_file(csv_file_names[i])

        if i == 0:
            continue

        list_of_metadata_tables[i] = list_of_metadata_tables[i].align(
            list_of_metadata_tables[0], axis=1
        )[0]

        list_of_predictor_tables[i] = list_of_predictor_tables[i].align(
            list_of_predictor_tables[0], axis=1
        )[0]

        list_of_target_tables[i] = list_of_target_tables[i].align(
            list_of_target_tables[0], axis=1
        )[0]

    metadata_table = pandas.concat(
        list_of_metadata_tables, axis=0, ignore_index=True)
    predictor_table = pandas.concat(
        list_of_predictor_tables, axis=0, ignore_index=True)
    target_table = pandas.concat(
        list_of_target_tables, axis=0, ignore_index=True)

    return metadata_table, predictor_table, target_table


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


def get_normalization_params(csv_file_names):
    """Computes normalization params (mean and stdev) for each predictor.

    :param csv_file_names: 1-D list of paths to input files.
    :return: normalization_dict: See input doc for `normalize_images`.
    """

    predictor_names = None
    norm_dict_by_predictor = None

    for this_file_name in csv_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_predictor_table = read_feature_file(this_file_name)[1]

        if predictor_names is None:
            predictor_names = list(this_predictor_table)
            norm_dict_by_predictor = [{}] * len(predictor_names)

        for m in range(len(predictor_names)):
            norm_dict_by_predictor[m] = _update_normalization_params(
                intermediate_normalization_dict=norm_dict_by_predictor[m],
                new_values=this_predictor_table[predictor_names[m]].values
            )

    print('\n')
    normalization_dict = {}

    for m in range(len(predictor_names)):
        this_mean = norm_dict_by_predictor[m][MEAN_VALUE_KEY]
        this_stdev = _get_standard_deviation(norm_dict_by_predictor[m])

        normalization_dict[predictor_names[m]] = numpy.array(
            [this_mean, this_stdev]
        )

        message_string = (
            'Mean and standard deviation for "{0:s}" = {1:.4f}, {2:.4f}'
        ).format(predictor_names[m], this_mean, this_stdev)
        print(message_string)

    return normalization_dict


def normalize_predictors(predictor_table, normalization_dict=None):
    """Normalizes predictors to z-scores.

    :param predictor_table: See doc for `read_feature_file`.
    :param normalization_dict: Dictionary.  Each key is the name of a predictor
        value, and the corresponding value is a length-2 numpy array with
        [mean, standard deviation].  If `normalization_dict is None`, mean and
        standard deviation will be computed for each predictor.
    :return: predictor_table: Normalized version of input.
    :return: normalization_dict: See doc for input variable.  If input was None,
        this will be a newly created dictionary.  Otherwise, this will be the
        same dictionary passed as input.
    """

    predictor_names = list(predictor_table)
    num_predictors = len(predictor_names)

    if normalization_dict is None:
        normalization_dict = {}

        for m in range(num_predictors):
            this_mean = numpy.mean(predictor_table[predictor_names[m]].values)
            this_stdev = numpy.std(
                predictor_table[predictor_names[m]].values, ddof=1
            )

            normalization_dict[predictor_names[m]] = numpy.array(
                [this_mean, this_stdev]
            )

    for m in range(num_predictors):
        this_mean = normalization_dict[predictor_names[m]][0]
        this_stdev = normalization_dict[predictor_names[m]][1]
        these_norm_values = (
            predictor_table[predictor_names[m]].values - this_mean
        ) / this_stdev

        predictor_table = predictor_table.assign(**{
            predictor_names[m]: these_norm_values
        })

    return predictor_table, normalization_dict


def denormalize_predictors(predictor_table, normalization_dict):
    """Denormalizes predictors from z-scores back to original scales.

    :param predictor_table: See doc for `normalize_predictors`.
    :param normalization_dict: Same.
    :return: predictor_table: Denormalized version of input.
    """

    predictor_names = list(predictor_table)
    num_predictors = len(predictor_names)

    for m in range(num_predictors):
        this_mean = normalization_dict[predictor_names[m]][0]
        this_stdev = normalization_dict[predictor_names[m]][1]
        these_denorm_values = (
            this_mean + this_stdev * predictor_table[predictor_names[m]].values
        )

        predictor_table = predictor_table.assign(**{
            predictor_names[m]: these_denorm_values
        })

    return predictor_table


def get_binarization_threshold(csv_file_names, percentile_level):
    """Computes binarization threshold for target variable.

    Binarization threshold will be [q]th percentile of all target values, where
    q = `percentile_level`.

    :param csv_file_names: 1-D list of paths to input files.
    :param percentile_level: q in the above discussion.
    :return: binarization_threshold: Binarization threshold (used to turn each
        target value into a yes-or-no label).
    """

    max_target_values = numpy.array([])

    for this_file_name in csv_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_target_table = read_feature_file(this_file_name)[-1]

        max_target_values = numpy.concatenate((
            max_target_values, this_target_table[TARGET_NAME].values
        ))

    binarization_threshold = numpy.percentile(
        max_target_values, percentile_level)

    print('\nBinarization threshold for "{0:s}" = {1:.4e}'.format(
        TARGET_NAME, binarization_threshold
    ))

    return binarization_threshold


def binarize_target_values(target_values, binarization_threshold):
    """Binarizes target values.

    E = number of examples (storm objects)

    :param target_values: length-E numpy array of real-number target values.
    :param binarization_threshold: Binarization threshold.
    :return: target_values: length-E numpy array of binarized target values
        (integers in 0...1).
    """

    return (target_values >= binarization_threshold).astype(int)


def _lambdas_to_sklearn_inputs(lambda1, lambda2):
    """Converts lambdas to input arguments for scikit-learn.

    :param lambda1: L1-regularization weight.
    :param lambda2: L2-regularization weight.
    :return: alpha: Input arg for scikit-learn model.
    :return: l1_ratio: Input arg for scikit-learn model.
    """

    return lambda1 + lambda2, lambda1 / (lambda1 + lambda2)


def setup_linear_regression(lambda1=0., lambda2=0.):
    """Sets up (but does not train) linear-regression model.

    :param lambda1: L1-regularization weight.
    :param lambda2: L2-regularization weight.
    :return: model_object: Instance of `sklearn.linear_model`.
    """

    assert lambda1 >= 0
    assert lambda2 >= 0

    if lambda1 < LAMBDA_TOLERANCE and lambda2 < LAMBDA_TOLERANCE:
        return sklearn.linear_model.LinearRegression(
            fit_intercept=True, normalize=False)

    if lambda1 < LAMBDA_TOLERANCE:
        return sklearn.linear_model.Ridge(
            alpha=lambda2, fit_intercept=True, normalize=False,
            random_state=RANDOM_SEED)

    if lambda2 < LAMBDA_TOLERANCE:
        return sklearn.linear_model.Lasso(
            alpha=lambda1, fit_intercept=True, normalize=False,
            random_state=RANDOM_SEED)

    alpha, l1_ratio = _lambdas_to_sklearn_inputs(
        lambda1=lambda1, lambda2=lambda2)

    return sklearn.linear_model.ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, normalize=False,
        random_state=RANDOM_SEED)


def train_linear_regression(model_object, training_predictor_table,
                            training_target_table):
    """Trains linear-regression model.

    :param model_object: Untrained model created by `setup_linear_regression`.
    :param training_predictor_table: See doc for `read_feature_file`.
    :param training_target_table: Same.
    :return: model_object: Trained version of input.
    """

    model_object.fit(
        X=training_predictor_table.as_matrix(),
        y=training_target_table[TARGET_NAME].values
    )

    return model_object


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


def write_model(model_object, pickle_file_name):
    """Writes model to Pickle file.

    :param model_object: Trained model (instance of `sklearn.linear_model`, for
        example).
    :param pickle_file_name: Path to output file.
    """

    print('Writing model to: "{0:s}"...'.format(pickle_file_name))
    _create_directory(file_name=pickle_file_name)

    file_handle = open(pickle_file_name, 'wb')
    pickle.dump(model_object, file_handle)
    file_handle.close()


def evaluate_regression(
        target_values, predicted_target_values, mean_training_target_value,
        verbose=True, create_plots=True, dataset_name=None):
    """Evaluates regression model.

    E = number of examples

    :param target_values: length-E numpy array of actual target values.
    :param predicted_target_values: length-E numpy array of predictions.
    :param mean_training_target_value: Mean target value in training data.
    :param verbose: Boolean flag.  If True, will print results to command
        window.
    :param create_plots: Boolean flag.  If True, will create plots.
    :param dataset_name: Dataset name (e.g., "validation").  Used only if
        `create_plots == True or verbose == True`.
    :return: evaluation_dict: Dictionary with the following keys.
    evaluation_dict['mean_absolute_error']: Mean absolute error (MAE).
    evaluation_dict['mean_squared_error']: Mean squared error (MSE).
    evaluation_dict['mean_bias']: Mean bias (signed error).
    evaluation_dict['mae_skill_score']: MAE skill score (fractional improvement
        over climatology, in range -1...1).
    evaluation_dict['mse_skill_score']: MSE skill score (fractional improvement
        over climatology, in range -1...1).
    """

    signed_errors = predicted_target_values - target_values
    mean_bias = numpy.mean(signed_errors)
    mean_absolute_error = numpy.mean(numpy.absolute(signed_errors))
    mean_squared_error = numpy.mean(signed_errors ** 2)

    climo_signed_errors = mean_training_target_value - target_values
    climo_mae = numpy.mean(numpy.absolute(climo_signed_errors))
    climo_mse = numpy.mean(climo_signed_errors ** 2)

    mae_skill_score = (climo_mae - mean_absolute_error) / climo_mae
    mse_skill_score = (climo_mse - mean_squared_error) / climo_mse

    evaluation_dict = {
        MAE_KEY: mean_absolute_error,
        MSE_KEY: mean_squared_error,
        MEAN_BIAS_KEY: mean_bias,
        MAE_SKILL_SCORE_KEY: mae_skill_score,
        MSE_SKILL_SCORE_KEY: mse_skill_score
    }

    if verbose or create_plots:
        dataset_name = dataset_name[0].upper() + dataset_name[1:]

    if verbose:
        print('{0:s} MAE (mean absolute error) = {1:.3e} s^-1'.format(
            dataset_name, evaluation_dict[MAE_KEY]
        ))
        print('{0:s} MSE (mean squared error) = {1:.3e} s^-2'.format(
            dataset_name, evaluation_dict[MSE_KEY]
        ))
        print('{0:s} bias (mean signed error) = {1:.3e} s^-1'.format(
            dataset_name, evaluation_dict[MEAN_BIAS_KEY]
        ))

        message_string = (
            '{0:s} MAE skill score (improvement over climatology) = {1:.3f}'
        ).format(dataset_name, evaluation_dict[MAE_SKILL_SCORE_KEY])
        print(message_string)

        message_string = (
            '{0:s} MSE skill score (improvement over climatology) = {1:.3f}'
        ).format(dataset_name, evaluation_dict[MSE_SKILL_SCORE_KEY])
        print(message_string)

    if not create_plots:
        return evaluation_dict

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(SMALL_FIG_WIDTH_INCHES, SMALL_FIG_HEIGHT_INCHES)
    )

    attr_diagrams.plot_regression_relia_curve(
        observed_values=target_values, forecast_values=predicted_target_values,
        num_bins=20, figure_object=figure_object, axes_object=axes_object)

    axes_object.set_xlabel(r'Forecast value (s$^{-1}$)')
    axes_object.set_ylabel(r'Conditional mean observation (s$^{-1}$)')

    title_string = '{0:s} reliability curve for max future vorticity'.format(
        dataset_name)
    axes_object.set_title(title_string)
    pyplot.show()

    return evaluation_dict


def plot_model_coefficients(model_object, predictor_names):
    """Plots coefficients for linear- or logistic-regression model.

    :param model_object: Trained instance of `sklearn.linear_model`.
    :param predictor_names: 1-D list of predictor names, in the same order used
        to train the model.
    """

    coefficients = model_object.coef_
    num_dimensions = len(coefficients.shape)
    if num_dimensions > 1:
        coefficients = coefficients[0, ...]

    num_predictors = len(predictor_names)
    y_coords = numpy.linspace(
        0, num_predictors - 1, num=num_predictors, dtype=float)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(DEFAULT_FIG_WIDTH_INCHES, DEFAULT_FIG_HEIGHT_INCHES)
    )

    axes_object.barh(
        y_coords, coefficients, color=BAR_GRAPH_COLOUR,
        edgecolor=BAR_GRAPH_COLOUR, linewidth=BAR_GRAPH_EDGE_WIDTH)

    pyplot.xlabel('Coefficient')
    pyplot.ylabel('Predictor variable')

    pyplot.yticks([], [])
    x_tick_values, _ = pyplot.xticks()
    pyplot.xticks(x_tick_values, rotation=90)

    x_min = numpy.percentile(coefficients, 1.)
    x_max = numpy.percentile(coefficients, 99.)
    pyplot.xlim([x_min, x_max])

    for j in range(num_predictors):
        axes_object.text(
            0, y_coords[j], predictor_names[j], color=BAR_GRAPH_FONT_COLOUR,
            horizontalalignment='center', verticalalignment='center',
            fontsize=BAR_GRAPH_FONT_SIZE)


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
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
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
        orientation=orientation_string, pad=padding, extend=extend_string,
        shrink=0.8)

    colour_bar_object.ax.tick_params(labelsize=FONT_SIZE)
    return colour_bar_object


def plot_scores_2d(
        score_matrix, min_colour_value, max_colour_value, x_tick_labels,
        y_tick_labels, colour_map_object=pyplot.cm.plasma):
    """Plots scores on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param x_tick_labels: length-N numpy array of tick values.
    :param y_tick_labels: length-M numpy array of tick values.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    """

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(DEFAULT_FIG_WIDTH_INCHES, DEFAULT_FIG_HEIGHT_INCHES)
    )

    pyplot.imshow(
        score_matrix, cmap=colour_map_object, origin='lower',
        vmin=min_colour_value, vmax=max_colour_value)

    x_tick_values = numpy.linspace(
        0, score_matrix.shape[1] - 1, num=score_matrix.shape[1], dtype=float
    )
    y_tick_values = numpy.linspace(
        0, score_matrix.shape[0] - 1, num=score_matrix.shape[0], dtype=float
    )

    pyplot.xticks(x_tick_values, x_tick_labels)
    pyplot.yticks(y_tick_values, y_tick_labels)

    _add_colour_bar(
        axes_object=axes_object, colour_map_object=colour_map_object,
        values_to_colour=score_matrix, min_colour_value=min_colour_value,
        max_colour_value=max_colour_value)


def setup_logistic_regression(lambda1=0., lambda2=0.):
    """Sets up (but does not train) logistic-regression model.

    :param lambda1: L1-regularization weight.
    :param lambda2: L2-regularization weight.
    :return: model_object: Instance of `sklearn.linear_model.SGDClassifier`.
    """

    assert lambda1 >= 0
    assert lambda2 >= 0

    if lambda1 < LAMBDA_TOLERANCE and lambda2 < LAMBDA_TOLERANCE:
        return sklearn.linear_model.SGDClassifier(
            loss='log', penalty='none', fit_intercept=True, verbose=0,
            random_state=RANDOM_SEED)

    if lambda1 < LAMBDA_TOLERANCE:
        return sklearn.linear_model.SGDClassifier(
            loss='log', penalty='l2', alpha=lambda2, fit_intercept=True,
            verbose=0, random_state=RANDOM_SEED)

    if lambda2 < LAMBDA_TOLERANCE:
        return sklearn.linear_model.SGDClassifier(
            loss='log', penalty='l1', alpha=lambda1, fit_intercept=True,
            verbose=0, random_state=RANDOM_SEED)

    alpha, l1_ratio = _lambdas_to_sklearn_inputs(
        lambda1=lambda1, lambda2=lambda2)

    return sklearn.linear_model.SGDClassifier(
        loss='log', penalty='elasticnet', alpha=alpha, l1_ratio=l1_ratio,
        fit_intercept=True, verbose=0, random_state=RANDOM_SEED)


def train_logistic_regression(model_object, training_predictor_table,
                              training_target_table):
    """Trains logistic-regression model.

    :param model_object: Untrained model created by `setup_logistic_regression`.
    :param training_predictor_table: See doc for `read_feature_file`.
    :param training_target_table: Same.
    :return: model_object: Trained version of input.
    """

    model_object.fit(
        X=training_predictor_table.as_matrix(),
        y=training_target_table[BINARIZED_TARGET_NAME].values
    )

    return model_object


def eval_binary_classifn(
        observed_labels, forecast_probabilities, training_event_frequency,
        verbose=True, create_plots=True, dataset_name=None):
    """Evaluates binary-classification model.

    E = number of examples

    :param observed_labels: length-E numpy array of observed labels (integers in
        0...1, where 1 means that event occurred).
    :param forecast_probabilities: length-E numpy array with forecast
        probabilities of event (positive class).
    :param training_event_frequency: Frequency of event in training data.
    :param verbose: Boolean flag.  If True, will print results to command
        window.
    :param create_plots: Boolean flag.  If True, will create plots.
    :param dataset_name: Dataset name (e.g., "validation").  Used only if
        `create_plots == True or verbose == True`.
    """

    pofd_by_threshold, pod_by_threshold = roc_curves.get_points_in_roc_curve(
        observed_labels=observed_labels,
        forecast_probabilities=forecast_probabilities)

    max_peirce_score = numpy.nanmax(pod_by_threshold - pofd_by_threshold)
    area_under_roc_curve = sklearn.metrics.auc(
        x=pofd_by_threshold, y=pod_by_threshold)

    pod_by_threshold, success_ratio_by_threshold = (
        perf_diagrams.get_points_in_perf_diagram(
            observed_labels=observed_labels,
            forecast_probabilities=forecast_probabilities)
    )

    csi_by_threshold = (
        (pod_by_threshold ** -1 + success_ratio_by_threshold ** -1 - 1) ** -1
    )
    max_csi = numpy.nanmax(csi_by_threshold)

    mean_forecast_by_bin, event_freq_by_bin, num_examples_by_bin = (
        attr_diagrams.get_points_in_relia_curve(
            observed_labels=observed_labels,
            forecast_probabilities=forecast_probabilities, num_bins=20)
    )

    uncertainty = training_event_frequency * (1. - training_event_frequency)

    this_numerator = numpy.nansum(
        num_examples_by_bin *
        (mean_forecast_by_bin - event_freq_by_bin) ** 2
    )
    reliability = this_numerator / numpy.sum(num_examples_by_bin)

    this_numerator = numpy.nansum(
        num_examples_by_bin *
        (event_freq_by_bin - training_event_frequency) ** 2
    )
    resolution = this_numerator / numpy.sum(num_examples_by_bin)

    brier_score = uncertainty + reliability - resolution
    brier_skill_score = (resolution - reliability) / uncertainty

    evaluation_dict = {
        MAX_PEIRCE_SCORE_KEY: max_peirce_score,
        AUC_KEY: area_under_roc_curve,
        MAX_CSI_KEY: max_csi,
        BRIER_SCORE_KEY: brier_score,
        BRIER_SKILL_SCORE_KEY: brier_skill_score
    }

    if verbose or create_plots:
        dataset_name = dataset_name[0].upper() + dataset_name[1:]

    if verbose:
        print('{0:s} Max Peirce score (POD - POFD) = {1:.3f}'.format(
            dataset_name, evaluation_dict[MAX_PEIRCE_SCORE_KEY]
        ))
        print('{0:s} AUC (area under ROC curve) = {1:.3f}'.format(
            dataset_name, evaluation_dict[AUC_KEY]
        ))
        print('{0:s} Max CSI (critical success index) = {1:.3f}'.format(
            dataset_name, evaluation_dict[MAX_CSI_KEY]
        ))
        print('{0:s} Brier score = {1:.3f}'.format(
            dataset_name, evaluation_dict[BRIER_SCORE_KEY]
        ))

        message_string = (
            '{0:s} Brier skill score (improvement over climatology) = {1:.3f}'
        ).format(dataset_name, evaluation_dict[BRIER_SKILL_SCORE_KEY])
        print(message_string)

    if not create_plots:
        return evaluation_dict

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(SMALL_FIG_WIDTH_INCHES, SMALL_FIG_HEIGHT_INCHES)
    )

    roc_curves.plot_roc_curve(
        observed_labels=observed_labels,
        forecast_probabilities=forecast_probabilities,
        axes_object=axes_object)

    title_string = '{0:s} ROC curve (AUC = {1:.3f})'.format(
        dataset_name, evaluation_dict[AUC_KEY]
    )

    pyplot.title(title_string)
    pyplot.show()

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(SMALL_FIG_WIDTH_INCHES, SMALL_FIG_HEIGHT_INCHES)
    )

    perf_diagrams.plot_performance_diagram(
        observed_labels=observed_labels,
        forecast_probabilities=forecast_probabilities,
        axes_object=axes_object)

    title_string = '{0:s} performance diagram (max CSI = {1:.3f})'.format(
        dataset_name, evaluation_dict[MAX_CSI_KEY]
    )

    pyplot.title(title_string)
    pyplot.show()

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(SMALL_FIG_WIDTH_INCHES, SMALL_FIG_HEIGHT_INCHES)
    )

    attr_diagrams.plot_attributes_diagram(
        observed_labels=observed_labels,
        forecast_probabilities=forecast_probabilities, num_bins=20,
        figure_object=figure_object, axes_object=axes_object)

    title_string = (
        '{0:s} attributes diagram (Brier skill score = {1:.3f})'
    ).format(dataset_name, evaluation_dict[BRIER_SKILL_SCORE_KEY])

    axes_object.set_title(title_string)
    pyplot.show()

    return evaluation_dict


def setup_classification_tree(min_examples_at_split=30,
                              min_examples_at_leaf=30):
    """Sets up (but does not train) decision tree for classification.

    :param min_examples_at_split: Minimum number of examples at split node.
    :param min_examples_at_leaf: Minimum number of examples at leaf node.
    :return: model_object: Instance of `sklearn.tree.DecisionTreeClassifier`.
    """

    return sklearn.tree.DecisionTreeClassifier(
        criterion='entropy', min_samples_split=min_examples_at_split,
        min_samples_leaf=min_examples_at_leaf, random_state=RANDOM_SEED)


def train_classification_tree(model_object, training_predictor_table,
                              training_target_table):
    """Trains decision tree for classification.

    :param model_object: Untrained model created by `setup_classification_tree`.
    :param training_predictor_table: See doc for `read_feature_file`.
    :param training_target_table: Same.
    :return: model_object: Trained version of input.
    """

    model_object.fit(
        X=training_predictor_table.as_matrix(),
        y=training_target_table[BINARIZED_TARGET_NAME].values
    )

    return model_object


def setup_classification_forest(
        max_predictors_per_split, num_trees=100, min_examples_at_split=30,
        min_examples_at_leaf=30):
    """Sets up (but does not train) random forest for classification.

    :param max_predictors_per_split: Max number of predictors to try at each
        split.
    :param num_trees: Number of trees.
    :param min_examples_at_split: Minimum number of examples at split node.
    :param min_examples_at_leaf: Minimum number of examples at leaf node.
    :return: model_object: Instance of
        `sklearn.ensemble.RandomForestClassifier`.
    """

    return sklearn.ensemble.RandomForestClassifier(
        n_estimators=num_trees, min_samples_split=min_examples_at_split,
        min_samples_leaf=min_examples_at_leaf,
        max_features=max_predictors_per_split, bootstrap=True,
        random_state=RANDOM_SEED, verbose=2)


def train_classification_forest(model_object, training_predictor_table,
                                training_target_table):
    """Trains random forest for classification.

    :param model_object: Untrained model created by
        `setup_classification_forest`.
    :param training_predictor_table: See doc for `read_feature_file`.
    :param training_target_table: Same.
    :return: model_object: Trained version of input.
    """

    model_object.fit(
        X=training_predictor_table.as_matrix(),
        y=training_target_table[BINARIZED_TARGET_NAME].values
    )

    return model_object


def setup_classification_gbt(
        max_predictors_per_split, num_trees=100, learning_rate=0.1,
        min_examples_at_split=30, min_examples_at_leaf=30):
    """Sets up (but does not train) gradient-boosted trees for classification.

    :param max_predictors_per_split: Max number of predictors to try at each
        split.
    :param num_trees: Number of trees.
    :param learning_rate: Learning rate.
    :param min_examples_at_split: Minimum number of examples at split node.
    :param min_examples_at_leaf: Minimum number of examples at leaf node.
    :return: model_object: Instance of
        `sklearn.ensemble.GradientBoostingClassifier`.
    """

    return sklearn.ensemble.GradientBoostingClassifier(
        loss='exponential', learning_rate=learning_rate, n_estimators=num_trees,
        min_samples_split=min_examples_at_split,
        min_samples_leaf=min_examples_at_leaf,
        max_features=max_predictors_per_split, random_state=RANDOM_SEED,
        verbose=2)


def train_classification_gbt(model_object, training_predictor_table,
                             training_target_table):
    """Trains gradient-boosted trees for classification.

    :param model_object: Untrained model created by
        `setup_classification_gbt`.
    :param training_predictor_table: See doc for `read_feature_file`.
    :param training_target_table: Same.
    :return: model_object: Trained version of input.
    """

    model_object.fit(
        X=training_predictor_table.as_matrix(),
        y=training_target_table[BINARIZED_TARGET_NAME].values
    )

    return model_object
