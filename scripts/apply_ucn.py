"""Applies UCN (upconvnet) to one or more examples."""

import random
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from keras import backend as K
from module_4 import ML_Short_Course_Module_4_Interpretation as short_course

random.seed(6695)
numpy.random.seed(6695)

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

FIGURE_RESOLUTION_DPI = 300
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

UCN_FILE_ARG_NAME = 'input_ucn_file_name'
IMAGE_DIR_ARG_NAME = 'input_image_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

UCN_FILE_HELP_STRING = (
    'Path to file with trained upconvnet.  Will be read by '
    '`short_course.read_keras_model`.')

IMAGE_DIR_HELP_STRING = (
    'Name of directory with image (NetCDF) files for input to the CNN.  Each '
    'example will be fed to the CNN to create scalar features, and the scalar '
    'features will be fed to the UCN (upconvnet) to reconstruct the original '
    'example (image).')

DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  The upconvnet will be applied to `{0:s}` random'
    ' examples from the period `{1:s}`...`{2:s}`.'
).format(NUM_EXAMPLES_ARG_NAME, FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

NUM_EXAMPLES_HELP_STRING = (
    'The upconvnet will be applied to `{0:s}` random examples from the period '
    '`{1:s}`...`{2:s}`.'
).format(NUM_EXAMPLES_ARG_NAME, FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Reconstructed examples (images) will be plotted'
    ' and saved here.')

DEFAULT_NUM_EXAMPLES = 50
DEFAULT_IMAGE_DIR_NAME = (
    '/condo/swatwork/ralager/ams2019_short_course/'
    'track_data_ncar_ams_3km_nc_small')

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
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_EXAMPLES, help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(input_ucn_file_name, input_image_dir_name, first_date_string,
         last_date_string, num_examples_to_keep, output_dir_name):
    """Applies UCN (upconvnet) to one or more examples.

    This is effectively the main method.

    :param input_ucn_file_name: See documentation at top of file.
    :param input_image_dir_name: Same.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param num_examples_to_keep: Same.
    :param output_dir_name: Same.
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

    # Decide which images to keep.
    num_examples = len(image_dict[short_course.STORM_IDS_KEY])
    num_examples_to_keep = min([num_examples_to_keep, num_examples])

    example_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int)
    example_indices = numpy.random.choice(
        example_indices, size=num_examples_to_keep, replace=False)

    storm_ids = numpy.round(
        image_dict[short_course.STORM_IDS_KEY][example_indices]
    ).astype(int)
    storm_steps = numpy.round(
        image_dict[short_course.STORM_STEPS_KEY][example_indices]
    ).astype(int)
    image_matrix = image_dict[short_course.PREDICTOR_MATRIX_KEY][
        example_indices, ...]

    # Reconstruct images.
    predictor_names = image_dict[short_course.PREDICTOR_NAMES_KEY]

    print('Normalizing {0:d} images...'.format(num_examples_to_keep))
    image_matrix_norm, _ = short_course.normalize_images(
        predictor_matrix=image_matrix + 0., predictor_names=predictor_names,
        normalization_dict=cnn_metadata_dict[
            short_course.NORMALIZATION_DICT_KEY]
    )

    print('Applying CNN to create scalar features...')
    feature_matrix = short_course._apply_cnn(
        cnn_model_object=cnn_model_object, predictor_matrix=image_matrix_norm,
        output_layer_name=ucn_metadata_dict[short_course.CNN_FEATURE_LAYER_KEY],
        verbose=False)

    print('Applying upconvnet to reconstruct images from scalar features...')
    reconstructed_image_matrix_norm = ucn_model_object.predict(
        feature_matrix, batch_size=num_examples_to_keep)

    print('Denormalizing reconstructed images...\n')
    reconstructed_image_matrix = short_course.denormalize_images(
        predictor_matrix=reconstructed_image_matrix_norm,
        predictor_names=predictor_names,
        normalization_dict=cnn_metadata_dict[
            short_course.NORMALIZATION_DICT_KEY]
    )

    # Plot reconstructed images.
    actual_image_dir_name = '{0:s}/actual_images'.format(output_dir_name)
    reconstructed_image_dir_name = '{0:s}/reconstructed_images'.format(
        output_dir_name)

    short_course._create_directory(actual_image_dir_name)
    short_course._create_directory(reconstructed_image_dir_name)

    temperature_index = predictor_names.index(short_course.TEMPERATURE_NAME)

    for i in range(num_examples_to_keep):
        this_temp_matrix_kelvins = numpy.concatenate(
            (image_matrix[i, ..., temperature_index],
             reconstructed_image_matrix[i, ..., temperature_index]),
            axis=0)

        this_min_temp_kelvins = numpy.percentile(this_temp_matrix_kelvins, 1)
        this_max_temp_kelvins = numpy.percentile(this_temp_matrix_kelvins, 99)

        short_course.plot_many_predictors_with_barbs(
            predictor_matrix=image_matrix[i, ...],
            predictor_names=predictor_names,
            min_colour_temp_kelvins=this_min_temp_kelvins,
            max_colour_temp_kelvins=this_max_temp_kelvins)

        this_figure_file_name = (
            '{0:s}/storm={1:06d}_step={2:06d}_actual.jpg'
        ).format(actual_image_dir_name, storm_ids[i], storm_steps[i])

        print('Saving figure to file: "{0:s}"...'.format(this_figure_file_name))
        pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        short_course.plot_many_predictors_with_barbs(
            predictor_matrix=reconstructed_image_matrix[i, ...],
            predictor_names=predictor_names,
            min_colour_temp_kelvins=this_min_temp_kelvins,
            max_colour_temp_kelvins=this_max_temp_kelvins)

        this_figure_file_name = (
            '{0:s}/storm={1:06d}_step={2:06d}_reconstructed.jpg'
        ).format(reconstructed_image_dir_name, storm_ids[i], storm_steps[i])

        print('Saving figure to file: "{0:s}"...'.format(this_figure_file_name))
        pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_ucn_file_name=getattr(INPUT_ARG_OBJECT, UCN_FILE_ARG_NAME),
        input_image_dir_name=getattr(INPUT_ARG_OBJECT, IMAGE_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        num_examples_to_keep=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
