"""Trains CNN for use in short course."""

import argparse
import faulthandler
from keras import backend as K
from module_4 import ML_Short_Course_Module_4_Interpretation as short_course

faulthandler.enable()

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=7, inter_op_parallelism_threads=7
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_GRID_ROWS = 32
NUM_GRID_COLUMNS = 32
FIRST_TRAINING_DATE_STRING = '20100101'
LAST_TRAINING_DATE_STRING = '20141231'
FIRST_VALIDATION_DATE_STRING = '20150101'
LAST_VALIDATION_DATE_STRING = '20151231'
PCT_LEVEL_FOR_BINARIZATION_THRESHOLD = 90.

IMAGE_DIR_ARG_NAME = 'input_image_dir_name'
NUM_EXAMPLES_PER_BATCH_ARG_NAME = 'num_examples_per_batch'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_TRAINING_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDATION_BATCHES_ARG_NAME = 'num_validation_batches_per_epoch'
OUTPUT_FILE_ARG_NAME = 'output_model_file_name'

IMAGE_DIR_HELP_STRING = (
    'Name of directory with image (NetCDF) files for training and validation.')
NUM_EXAMPLES_PER_BATCH_HELP_STRING = (
    'Number of examples in each training or validation batch.')
NUM_EPOCHS_HELP_STRING = 'Number of training epochs.'
NUM_TRAINING_BATCHES_HELP_STRING = 'Number of training batches in each epoch.'
NUM_VALIDATION_BATCHES_HELP_STRING = (
    'Number of validation batches in each epoch.')
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (HDF5 format).  The trained model will be saved here.')

DEFAULT_NUM_EXAMPLES_PER_BATCH = 1024
DEFAULT_NUM_EPOCHS = 100
DEFAULT_NUM_TRAINING_BATCHES_PER_EPOCH = 32
DEFAULT_NUM_VALIDATION_BATCHES_PER_EPOCH = 16
DEFAULT_IMAGE_DIR_NAME = (
    '/condo/swatwork/ralager/ams2019_short_course/'
    'track_data_ncar_ams_3km_nc_small')

INPUT_ARG_PARSER = argparse.ArgumentParser()
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


def _run(input_image_dir_name, num_examples_per_batch, num_epochs,
         num_training_batches_per_epoch, num_validation_batches_per_epoch,
         output_model_file_name):
    """Trains CNN for use in short course.

    This is effectively the main method.

    :param input_image_dir_name: See documentation at top of file.
    :param num_examples_per_batch: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param output_model_file_name: Same.
    """

    cnn_model_object = short_course.setup_cnn(
        num_grid_rows=NUM_GRID_ROWS, num_grid_columns=NUM_GRID_COLUMNS)
    print(SEPARATOR_STRING)

    training_file_names = short_course.find_many_image_files(
        first_date_string=FIRST_TRAINING_DATE_STRING,
        last_date_string=LAST_TRAINING_DATE_STRING,
        image_dir_name=input_image_dir_name)

    normalization_dict = short_course.get_image_normalization_params(
        training_file_names)
    print(SEPARATOR_STRING)

    binarization_threshold = short_course.get_binarization_threshold(
        netcdf_file_names=training_file_names,
        percentile_level=PCT_LEVEL_FOR_BINARIZATION_THRESHOLD)
    print(SEPARATOR_STRING)

    validation_file_names = short_course.find_many_image_files(
        first_date_string=FIRST_VALIDATION_DATE_STRING,
        last_date_string=LAST_VALIDATION_DATE_STRING,
        image_dir_name=input_image_dir_name)

    model_metadata_dict = short_course.train_cnn(
        cnn_model_object=cnn_model_object,
        training_file_names=training_file_names,
        normalization_dict=normalization_dict,
        binarization_threshold=binarization_threshold,
        num_examples_per_batch=num_examples_per_batch, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        output_model_file_name=output_model_file_name,
        validation_file_names=validation_file_names,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch)
    print(SEPARATOR_STRING)

    model_metafile_name = short_course.find_model_metafile(
        model_file_name=output_model_file_name, raise_error_if_missing=False)

    print('Writing metadata to: "{0:s}"...'.format(model_metafile_name))
    short_course.write_model_metadata(model_metadata_dict=model_metadata_dict,
                                      json_file_name=model_metafile_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
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
