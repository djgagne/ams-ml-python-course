"""Trains UCN (upconvnet) for use in short course."""

import argparse
import numpy
from keras import backend as K
from module_4 import ML_Short_Course_Module_4_Interpretation as short_course

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=7, inter_op_parallelism_threads=7
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FIRST_TRAINING_DATE_STRING = '20100101'
LAST_TRAINING_DATE_STRING = '20141231'
FIRST_VALIDATION_DATE_STRING = '20150101'
LAST_VALIDATION_DATE_STRING = '20151231'

UPSAMPLING_FACTORS = numpy.array([2, 1, 1, 2, 1, 1], dtype=int)

CNN_FILE_ARG_NAME = 'input_cnn_file_name'
USE_BATCH_NORM_ARG_NAME = 'use_batch_norm_for_out_layer'
USE_TRANSPOSED_CONV_ARG_NAME = 'use_transposed_conv'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_px'
IMAGE_DIR_ARG_NAME = 'input_image_dir_name'
NUM_EXAMPLES_PER_BATCH_ARG_NAME = 'num_examples_per_batch'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_TRAINING_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDATION_BATCHES_ARG_NAME = 'num_validation_batches_per_epoch'
OUTPUT_FILE_ARG_NAME = 'output_model_file_name'

CNN_FILE_HELP_STRING = (
    'Path to file with trained CNN.  UCN predictors will be outputs from the '
    'CNN flattening layer, and UCN targets will be CNN predictors (input '
    'images).')

USE_BATCH_NORM_HELP_STRING = (
    'Boolean flag.  If 1, will use batch normalization after output layer.')

USE_TRANSPOSED_CONV_HELP_STRING = (
    'Boolean flag.  If 1, upsampling will be done with transposed-convolution '
    'layers.  If False, each upsampling will be done with an upsampling layer '
    'followed by a conv layer.')

SMOOTHING_RADIUS_HELP_STRING = (
    'Smoothing radius (pixels).  Gaussian smoothing with this e-folding radius '
    'will be done after each upsampling.  If you do not want smoothing, leave '
    'this alone.')

IMAGE_DIR_HELP_STRING = (
    'Name of directory with image (NetCDF) files for input to the CNN.  This '
    'directory will be used for both training and validation.')

NUM_EXAMPLES_PER_BATCH_HELP_STRING = (
    'Number of examples in each training or validation batch.')

NUM_EPOCHS_HELP_STRING = 'Number of training epochs.'

NUM_TRAINING_BATCHES_HELP_STRING = 'Number of training batches in each epoch.'

NUM_VALIDATION_BATCHES_HELP_STRING = (
    'Number of validation batches in each epoch.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (HDF5 format).  The trained UCN model will be saved '
    'here.')

DEFAULT_USE_BATCH_NORM_FLAG = 1
DEFAULT_TRANSPOSED_CONV_FLAG = 0
DEFAULT_SMOOTHING_RADIUS_PX = -1
DEFAULT_NUM_EXAMPLES_PER_BATCH = 1024
DEFAULT_NUM_EPOCHS = 100
DEFAULT_NUM_TRAINING_BATCHES_PER_EPOCH = 32
DEFAULT_NUM_VALIDATION_BATCHES_PER_EPOCH = 16
DEFAULT_IMAGE_DIR_NAME = (
    '/condo/swatwork/ralager/ams2019_short_course/'
    'track_data_ncar_ams_3km_nc_small')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + CNN_FILE_ARG_NAME, type=str, required=True,
    help=CNN_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + USE_BATCH_NORM_ARG_NAME, type=int, required=False,
    default=DEFAULT_USE_BATCH_NORM_FLAG, help=USE_BATCH_NORM_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + USE_TRANSPOSED_CONV_ARG_NAME, type=int, required=False,
    default=DEFAULT_TRANSPOSED_CONV_FLAG, help=USE_TRANSPOSED_CONV_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=int, required=False,
    default=DEFAULT_SMOOTHING_RADIUS_PX, help=SMOOTHING_RADIUS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + IMAGE_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_IMAGE_DIR_NAME, help=IMAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_BATCH_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_EXAMPLES_PER_BATCH,
    help=NUM_EXAMPLES_PER_BATCH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_EPOCHS, help=NUM_EPOCHS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TRAINING_BATCHES_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_TRAINING_BATCHES_PER_EPOCH,
    help=NUM_TRAINING_BATCHES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_VALIDATION_BATCHES_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_VALIDATION_BATCHES_PER_EPOCH,
    help=NUM_VALIDATION_BATCHES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(input_cnn_file_name, use_batch_norm_for_out_layer, use_transposed_conv,
         smoothing_radius_px, input_image_dir_name, num_examples_per_batch,
         num_epochs, num_training_batches_per_epoch,
         num_validation_batches_per_epoch, output_model_file_name):
    """Trains UCN (upconvnet) for use in short course.

    This is effectively the main method.

    :param input_cnn_file_name: See documentation at top of file.
    :param use_batch_norm_for_out_layer: Same.
    :param use_transposed_conv: Same.
    :param smoothing_radius_px: Same.
    :param input_image_dir_name: Same.
    :param num_examples_per_batch: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param output_model_file_name: Same.
    """

    if smoothing_radius_px <= 0:
        smoothing_radius_px = None

    print('Reading trained CNN from: "{0:s}"...'.format(input_cnn_file_name))
    cnn_model_object = short_course.read_keras_model(input_cnn_file_name)

    cnn_metafile_name = short_course.find_model_metafile(
        model_file_name=input_cnn_file_name, raise_error_if_missing=True)

    print('Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name))
    cnn_metadata_dict = short_course.read_model_metadata(cnn_metafile_name)

    cnn_feature_layer_name = short_course.get_cnn_flatten_layer(
        cnn_model_object)
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

    ucn_model_object = short_course.setup_ucn(
        num_input_features=num_input_features, first_num_rows=first_num_rows,
        first_num_columns=first_num_columns,
        upsampling_factors=UPSAMPLING_FACTORS,
        num_output_channels=num_output_channels,
        use_activation_for_out_layer=False,
        use_bn_for_out_layer=use_batch_norm_for_out_layer,
        use_transposed_conv=use_transposed_conv,
        smoothing_radius_px=smoothing_radius_px)
    print(SEPARATOR_STRING)

    training_file_names = short_course.find_many_image_files(
        first_date_string=FIRST_TRAINING_DATE_STRING,
        last_date_string=LAST_TRAINING_DATE_STRING,
        image_dir_name=input_image_dir_name)

    validation_file_names = short_course.find_many_image_files(
        first_date_string=FIRST_VALIDATION_DATE_STRING,
        last_date_string=LAST_VALIDATION_DATE_STRING,
        image_dir_name=input_image_dir_name)

    ucn_metadata_dict = short_course.train_ucn(
        ucn_model_object=ucn_model_object,
        training_file_names=training_file_names,
        normalization_dict=cnn_metadata_dict[
            short_course.NORMALIZATION_DICT_KEY],
        cnn_model_object=cnn_model_object, cnn_file_name=input_cnn_file_name,
        cnn_feature_layer_name=cnn_feature_layer_name,
        num_examples_per_batch=num_examples_per_batch, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        output_model_file_name=output_model_file_name,
        validation_file_names=validation_file_names,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch)
    print(SEPARATOR_STRING)

    ucn_metafile_name = short_course.find_model_metafile(
        model_file_name=output_model_file_name, raise_error_if_missing=False)

    print('Writing metadata to: "{0:s}"...'.format(ucn_metafile_name))
    short_course.write_model_metadata(model_metadata_dict=ucn_metadata_dict,
                                      json_file_name=ucn_metafile_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_cnn_file_name=getattr(INPUT_ARG_OBJECT, CNN_FILE_ARG_NAME),
        use_batch_norm_for_out_layer=bool(
            getattr(INPUT_ARG_OBJECT, USE_BATCH_NORM_ARG_NAME)),
        use_transposed_conv=bool(
            getattr(INPUT_ARG_OBJECT, USE_TRANSPOSED_CONV_ARG_NAME)),
        smoothing_radius_px=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME),
        input_image_dir_name=getattr(INPUT_ARG_OBJECT, IMAGE_DIR_ARG_NAME),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_BATCH_ARG_NAME),
        num_epochs=getattr(INPUT_ARG_OBJECT, NUM_EPOCHS_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, NUM_TRAINING_BATCHES_ARG_NAME),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, NUM_VALIDATION_BATCHES_ARG_NAME),
        output_model_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
