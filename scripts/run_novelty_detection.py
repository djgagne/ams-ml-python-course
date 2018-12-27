"""Runs novelty detection."""

import pickle
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from keras import backend as K
from module_4 import ML_Short_Course_Module_4_Interpretation as short_course

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

FIGURE_RESOLUTION_DPI = 300
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

UCN_FILE_ARG_NAME = 'input_ucn_file_name'
IMAGE_DIR_ARG_NAME = 'input_image_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
NUM_BASELINE_EX_ARG_NAME = 'num_baseline_examples'
NUM_TEST_EX_ARG_NAME = 'num_test_examples'
NUM_SVD_MODES_ARG_NAME = 'num_svd_modes_to_keep'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

UCN_FILE_HELP_STRING = (
    'Path to file with trained upconvnet (decoder).  Will be read by '
    '`short_course.read_keras_model`.')

IMAGE_DIR_HELP_STRING = (
    'Name of directory with image (NetCDF) files for input to novelty '
    'detection.  Images will be fed to the CNN to create scalar features '
    '(encode), and scalar features will be fed to the upconvnet to create '
    'images (decode).')

DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Novelty detection will be applied to `{0:s}` '
    'baseline examples and `{1:s}` test examples from the period '
    '`{2:s}`...`{3:s}`.'
).format(NUM_BASELINE_EX_ARG_NAME, NUM_TEST_EX_ARG_NAME, FIRST_DATE_ARG_NAME,
         LAST_DATE_ARG_NAME)

NUM_BASELINE_EX_HELP_STRING = (
    'Number of baseline examples.  `{0:s}` examples will be randomly selected '
    'from the period `{1:s}`...`{2:s}` and used to create the original SVD '
    '(singular-value decomposition) model.  The novelty of each test example '
    'will be computed relative to these baseline examples.'
).format(NUM_BASELINE_EX_ARG_NAME, FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

NUM_TEST_EX_HELP_STRING = (
    'Number of testing examples.  The `{0:s}` examples with the greatest future'
    ' vorticity will be selected from the period `{1:s}`...`{2:s}` and ranked '
    'by their novelty with respect to baseline examples.'
).format(NUM_TEST_EX_ARG_NAME, FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

NUM_SVD_MODES_HELP_STRING = (
    'Number of modes (top eigenvectors) to retain in the SVD (singular-value '
    'decomposition) model.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  The dictionary created by '
    '`short_course.do_novelty_detection`, as well as plots, will be saved here.'
)

DEFAULT_IMAGE_DIR_NAME = (
    '/condo/swatwork/ralager/ams2019_short_course/'
    'track_data_ncar_ams_3km_nc_small'
)
DEFAULT_NUM_BASELINE_EXAMPLES = 100
DEFAULT_NUM_TEST_EXAMPLES = 100

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + UCN_FILE_ARG_NAME, type=str, required=True,
    help=UCN_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + IMAGE_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_IMAGE_DIR_NAME, help=IMAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BASELINE_EX_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_BASELINE_EXAMPLES, help=NUM_BASELINE_EX_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TEST_EX_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_TEST_EXAMPLES, help=NUM_TEST_EX_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_SVD_MODES_ARG_NAME, type=int, required=True,
    help=NUM_SVD_MODES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _write_novelty_results(novelty_dict, pickle_file_name):
    """Writes novelty-detection results to Pickle file.

    :param novelty_dict: Dictionary created by
        `short_course.do_novelty_detection`.
    :param pickle_file_name: Path to output file.
    """

    short_course._create_directory(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(novelty_dict, pickle_file_handle)
    pickle_file_handle.close()


def _plot_novelty_detection(image_dict, novelty_dict, test_index,
                            top_output_dir_name):
    """Plots results of novelty detection.

    :param image_dict: Dictionary created by
        `short_course.read_many_image_files`, containing input data for novelty
        detection.
    :param novelty_dict: Dictionary created by
        `short_course.do_novelty_detection`, containing results.
    :param test_index: Array index.  The [i]th-most novel test example will be
        plotted, where i = `test_index`.
    :param top_output_dir_name: Name of top-level output directory.  Figures
        will be saved here.
    """

    predictor_names = image_dict[short_course.PREDICTOR_NAMES_KEY]
    temperature_index = predictor_names.index(short_course.TEMPERATURE_NAME)
    reflectivity_index = predictor_names.index(short_course.REFLECTIVITY_NAME)

    image_matrix_actual = novelty_dict[
        short_course.NOVEL_IMAGES_ACTUAL_KEY][test_index, ...]
    image_matrix_upconv = novelty_dict[
        short_course.NOVEL_IMAGES_UPCONV_KEY][test_index, ...]
    image_matrix_upconv_svd = novelty_dict[
        short_course.NOVEL_IMAGES_UPCONV_SVD_KEY][test_index, ...]

    combined_matrix_kelvins = numpy.concatenate(
        (image_matrix_actual[..., temperature_index],
         image_matrix_upconv[..., temperature_index]),
        axis=0)

    min_colour_temp_kelvins = numpy.percentile(combined_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_matrix_kelvins, 99)

    this_figure_object, _ = short_course.plot_many_predictors_with_barbs(
        predictor_matrix=image_matrix_actual, predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    base_title_string = '{0:d}th-most novel example'.format(test_index + 1)
    this_title_string = '{0:s}: actual'.format(base_title_string)
    this_figure_object.suptitle(this_title_string)

    this_file_name = '{0:s}/actual_images/actual_image{1:04d}.jpg'.format(
        top_output_dir_name, test_index)
    short_course._create_directory(file_name=this_file_name)

    print('Saving figure to file: "{0:s}"...'.format(this_file_name))
    pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    this_figure_object, _ = short_course.plot_many_predictors_with_barbs(
        predictor_matrix=image_matrix_upconv,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    this_title_string = r'{0:s}: upconvnet reconstruction'.format(
        base_title_string)
    this_title_string += r' ($\mathbf{X}_{up}$)'
    this_figure_object.suptitle(this_title_string)

    this_file_name = '{0:s}/upconv_images/upconv_image{1:04d}.jpg'.format(
        top_output_dir_name, test_index)
    short_course._create_directory(file_name=this_file_name)

    print('Saving figure to file: "{0:s}"...'.format(this_file_name))
    pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    novelty_matrix = image_matrix_upconv - image_matrix_upconv_svd
    max_absolute_temp_kelvins = numpy.percentile(
        numpy.absolute(novelty_matrix[..., temperature_index]), 99)
    max_absolute_refl_dbz = numpy.percentile(
        numpy.absolute(novelty_matrix[..., reflectivity_index]), 99)

    this_figure_object, _ = short_course._plot_novelty_for_many_predictors(
        novelty_matrix=novelty_matrix, predictor_names=predictor_names,
        max_absolute_temp_kelvins=max_absolute_temp_kelvins,
        max_absolute_refl_dbz=max_absolute_refl_dbz)

    this_title_string = r'{0:s}: novelty'.format(
        base_title_string)
    this_title_string += r' ($\mathbf{X}_{up} - \mathbf{X}_{up,svd}$)'
    this_figure_object.suptitle(this_title_string)

    this_file_name = (
        '{0:s}/upconv_svd_images/upconv_svd_image{1:04d}.jpg'
    ).format(top_output_dir_name, test_index)
    short_course._create_directory(file_name=this_file_name)

    print('Saving figure to file: "{0:s}"...'.format(this_file_name))
    pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


def _run(input_ucn_file_name, input_image_dir_name, first_date_string,
         last_date_string, num_baseline_examples, num_test_examples,
         num_svd_modes_to_keep, top_output_dir_name):
    """Runs novelty detection.

    This is effectively the main method.

    :param input_ucn_file_name: See documentation at top of file.
    :param input_image_dir_name: Same.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param num_baseline_examples: Same.
    :param num_test_examples: Same.
    :param num_svd_modes_to_keep: Same.
    :param top_output_dir_name: Same.
    """

    # Read upconvnet and CNN.
    ucn_metafile_name = short_course.find_model_metafile(
        model_file_name=input_ucn_file_name, raise_error_if_missing=True)

    print('Reading trained upconvnet from: "{0:s}"...'.format(
        input_ucn_file_name))
    ucn_model_object = short_course.read_keras_model(input_ucn_file_name)

    print('Reading upconvnet metadata from: "{0:s}"...'.format(
        ucn_metafile_name))
    ucn_metadata_dict = short_course.read_model_metadata(ucn_metafile_name)

    cnn_file_name = ucn_metadata_dict[short_course.CNN_FILE_KEY]
    cnn_metafile_name = short_course.find_model_metafile(
        model_file_name=cnn_file_name, raise_error_if_missing=True)

    print('Reading trained CNN from: "{0:s}"...'.format(cnn_file_name))
    cnn_model_object = short_course.read_keras_model(cnn_file_name)

    print('Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name))
    cnn_metadata_dict = short_course.read_model_metadata(cnn_metafile_name)
    print(SEPARATOR_STRING)

    # Read images.
    image_file_names = short_course.find_many_image_files(
        first_date_string=first_date_string, last_date_string=last_date_string,
        image_dir_name=input_image_dir_name)

    image_dict = short_course.read_many_image_files(image_file_names)
    print(SEPARATOR_STRING)

    # Extract test examples.
    target_matrix_s01 = image_dict[short_course.TARGET_MATRIX_KEY]
    num_examples = target_matrix_s01.shape[0]
    max_target_by_example_s01 = numpy.array(
        [numpy.max(target_matrix_s01[i, ...]) for i in range(num_examples)]
    )

    test_indices = numpy.argsort(-1 * max_target_by_example_s01)[:100]

    # test_storm_ids = numpy.round(
    #     image_dict[short_course.STORM_IDS_KEY][test_indices]
    # ).astype(int)
    # test_storm_steps = numpy.round(
    #     image_dict[short_course.STORM_STEPS_KEY][test_indices]
    # ).astype(int)
    test_image_matrix = image_dict[short_course.PREDICTOR_MATRIX_KEY][
        test_indices, ...]

    # Extract baseline examples.
    baseline_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int)

    baseline_indices = (
        set(baseline_indices.tolist()) - set(test_indices.tolist())
    )
    baseline_indices = numpy.array(list(baseline_indices), dtype=int)
    baseline_indices = numpy.random.choice(
        baseline_indices, size=num_baseline_examples, replace=False)

    baseline_image_matrix = image_dict[short_course.PREDICTOR_MATRIX_KEY][
        baseline_indices, ...]

    # Run novelty detection.
    novelty_dict = short_course.do_novelty_detection(
        baseline_image_matrix=baseline_image_matrix,
        test_image_matrix=test_image_matrix,
        image_normalization_dict=cnn_metadata_dict[
            short_course.NORMALIZATION_DICT_KEY],
        predictor_names=cnn_metadata_dict[short_course.PREDICTOR_NAMES_KEY],
        cnn_model_object=cnn_model_object,
        cnn_feature_layer_name=short_course.get_cnn_flatten_layer(
            cnn_model_object),
        ucn_model_object=ucn_model_object,
        num_novel_test_images=num_test_examples,
        num_svd_modes_to_keep=num_svd_modes_to_keep)
    print(SEPARATOR_STRING)

    novelty_file_name = '{0:s}/novelty_results.p'.format(top_output_dir_name)
    print('Writing novelty results to: "{0:s}"...\n'.format(novelty_file_name))
    _write_novelty_results(novelty_dict=novelty_dict,
                           pickle_file_name=novelty_file_name)

    for i in range(num_test_examples):
        _plot_novelty_detection(
            image_dict=image_dict, novelty_dict=novelty_dict, test_index=i,
            top_output_dir_name=top_output_dir_name)
        print('\n')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_ucn_file_name=getattr(INPUT_ARG_OBJECT, UCN_FILE_ARG_NAME),
        input_image_dir_name=getattr(INPUT_ARG_OBJECT, IMAGE_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        num_baseline_examples=getattr(
            INPUT_ARG_OBJECT, NUM_BASELINE_EX_ARG_NAME),
        num_test_examples=getattr(INPUT_ARG_OBJECT, NUM_TEST_EX_ARG_NAME),
        num_svd_modes_to_keep=getattr(INPUT_ARG_OBJECT, NUM_SVD_MODES_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
