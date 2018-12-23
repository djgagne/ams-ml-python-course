"""Trains UCN (upconvnet) for use in short course."""

import pickle
import os.path
import argparse
import numpy
from keras import backend as K
from keras.models import load_model as read_keras_model
from module_4 import ML_Short_Course_Module_4_Interpretation as short_course

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=7, inter_op_parallelism_threads=7
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FIRST_TRAINING_DATE_STRING = '20100101'
LAST_TRAINING_DATE_STRING = '20141231'
FIRST_VALIDATION_DATE_STRING = '20150101'
LAST_VALIDATION_DATE_STRING = '20151231'

CNN_FILE_ARG_NAME = 'input_cnn_file_name'
OUT_LAYER_ACTIVATION_ARG_NAME = 'use_activation_for_out_layer'
OUT_LAYER_BN_ARG_NAME = 'use_bn_for_out_layer'
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

OUT_LAYER_ACTIVATION_HELP_STRING = (
    'Boolean flag.  If 1, will use activation after last UCN layer.')

OUT_LAYER_BN_HELP_STRING = (
    'Boolean flag.  If 1, will use batch normalization after last UCN layer.  '
    'Keep in mind that batch norm is always done after activation, so if both '
    '`{0:s}` and `{1:s}` are 1, the last deconv layer will be followed by '
    'activation, then batch norm.'
).format(OUT_LAYER_ACTIVATION_ARG_NAME, OUT_LAYER_BN_ARG_NAME)

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

DEFAULT_OUT_LAYER_ACTIVATION_FLAG = 0
DEFAULT_OUT_LAYER_BN_FLAG = 1
DEFAULT_IMAGE_DIR_NAME = ''
DEFAULT_NUM_EXAMPLES_PER_BATCH = 1024
DEFAULT_NUM_EPOCHS = 100
DEFAULT_NUM_TRAINING_BATCHES_PER_EPOCH = 32
DEFAULT_NUM_VALIDATION_BATCHES_PER_EPOCH = 16

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + CNN_FILE_ARG_NAME, type=str, required=True,
    help=CNN_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUT_LAYER_ACTIVATION_ARG_NAME, type=int, required=False,
    default=DEFAULT_OUT_LAYER_ACTIVATION_FLAG,
    help=OUT_LAYER_ACTIVATION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUT_LAYER_BN_ARG_NAME, type=int, required=False,
    default=DEFAULT_OUT_LAYER_BN_FLAG, help=OUT_LAYER_BN_HELP_STRING)

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


def get_flattening_layer(cnn_model_object):
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


def _run(input_cnn_file_name, use_activation_for_out_layer,
         use_bn_for_out_layer, input_image_dir_name, num_examples_per_batch,
         num_epochs, num_training_batches_per_epoch,
         num_validation_batches_per_epoch, output_model_file_name):
    """Trains UCN (upconvnet) for use in short course.

    This is effectively the main method.

    :param input_cnn_file_name: See documentation at top of file.
    :param use_activation_for_out_layer: Same.
    :param use_bn_for_out_layer: Same.
    :param input_image_dir_name: Same.
    :param num_examples_per_batch: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param output_model_file_name: Same.
    """

    print('Reading trained CNN from: "{0:s}"...'.format(input_cnn_file_name))
    cnn_model_object = read_keras_model(
        input_cnn_file_name,
        custom_objects=short_course.LIST_OF_METRIC_FUNCTIONS)

    # TODO(thunderhoser): Write method to find metafile.

    cnn_directory_name, pathless_cnn_file_name = os.path.split(
        input_cnn_file_name)
    cnn_metafile_name = '{0:s}/{1:s}_metadata.p'.format(
        cnn_directory_name, os.path.splitext(pathless_cnn_file_name)[0]
    )

    # TODO(thunderhoser): Write IO methods for metadata.

    print('Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name))
    metafile_handle = open(cnn_metafile_name, 'rb')
    cnn_metadata_dict = pickle.load(metafile_handle)
    metafile_handle.close()

    # TODO(thunderhoser): Need method to find flattening layer in CNN.
    flattening_layer_name = get_flattening_layer(cnn_model_object)

    short_course.setup_ucn_fancy(
        num_input_features, first_num_rows, first_num_columns,
        upsampling_factor_by_deconv_layer, num_output_channels,
        use_activation_for_out_layer, use_bn_for_out_layer)
    print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_cnn_file_name=getattr(INPUT_ARG_OBJECT, CNN_FILE_ARG_NAME),
        use_activation_for_out_layer=bool(
            getattr(INPUT_ARG_OBJECT, OUT_LAYER_ACTIVATION_ARG_NAME)),
        use_bn_for_out_layer=bool(
            getattr(INPUT_ARG_OBJECT, OUT_LAYER_BN_ARG_NAME)),
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
