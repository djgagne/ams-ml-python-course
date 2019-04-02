"""Non-notebook version of Module 2 for AMS 2019 short course."""

import numpy
import matplotlib.pyplot as pyplot
from module_2 import utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'


def find_training_and_validation():
    """Finds training and validation data."""

    training_file_names = utils.find_many_feature_files(
        first_date_string='20100101', last_date_string='20141231')

    validation_file_names = utils.find_many_feature_files(
        first_date_string='20150101', last_date_string='20151231')


def read_validation(validation_file_names):
    """Reads validation data.

    :param validation_file_names: 1-D list of paths to input files.
    """

    (validation_metadata_table, validation_predictor_table,
     validation_target_table
    ) = utils.read_many_feature_files(validation_file_names)

    print(MINOR_SEPARATOR_STRING)
    print('Variables in metadata are as follows:\n{0:s}'.format(
        str(list(validation_metadata_table))
    ))

    print('\nPredictor variables are as follows:\n{0:s}'.format(
        str(list(validation_predictor_table))
    ))

    print('\nTarget variable is as follows:\n{0:s}'.format(
        str(list(validation_target_table))
    ))

    first_predictor_name = list(validation_predictor_table)[0]
    these_predictor_values = (
        validation_predictor_table[first_predictor_name].values[:10]
    )

    message_string = (
        '\nValues of predictor variable "{0:s}" for the first few storm '
        'objects:\n{1:s}'
    ).format(first_predictor_name, str(these_predictor_values))
    print(message_string)

    target_name = list(validation_target_table)[0]
    these_target_values = validation_target_table[target_name].values[:10]

    message_string = (
        '\nValues of target variable ("{0:s}") for the first few storm '
        'objects:\n{1:s}'
    ).format(target_name, str(these_target_values))
    print(message_string)


def norm_and_denorm(training_file_names):
    """Finds and applies normalization parameters.

    :param training_file_names: 1-D list of paths to input files.
    """

    normalization_dict = utils.get_normalization_params(training_file_names)
    print(MINOR_SEPARATOR_STRING)

    first_training_predictor_table = utils.read_feature_file(
        training_file_names[0]
    )[1]

    predictor_names = list(first_training_predictor_table)
    these_predictor_values = (
        first_training_predictor_table[predictor_names[0]].values[:10]
    )

    message_string = (
        '\nOriginal values of "{0:s}" for the first few storm objects:\n{1:s}'
    ).format(predictor_names[0], str(these_predictor_values))
    print(message_string)

    first_training_predictor_table, _ = utils.normalize_predictors(
        predictor_table=first_training_predictor_table,
        normalization_dict=normalization_dict)

    these_predictor_values = (
        first_training_predictor_table[predictor_names[0]].values[:10]
    )

    message_string = (
        '\nNormalized values of "{0:s}" for the first few storm objects:\n{1:s}'
    ).format(predictor_names[0], str(these_predictor_values))
    print(message_string)

    first_training_predictor_table = utils.denormalize_predictors(
        predictor_table=first_training_predictor_table,
        normalization_dict=normalization_dict)

    these_predictor_values = (
        first_training_predictor_table[predictor_names[0]].values[:10]
    )

    message_string = (
        '\nDenormalized values of "{0:s}" for the first few storm objects:'
        '\n{1:s}'
    ).format(predictor_names[0], str(these_predictor_values))
    print(message_string)


def binarization_example(training_file_names):
    """Finds and applies binarization threshold.

    :param training_file_names: 1-D list of paths to input files.
    """

    binarization_threshold = utils.get_binarization_threshold(
        csv_file_names=training_file_names, percentile_level=90.)
    print(MINOR_SEPARATOR_STRING)

    first_training_target_table = utils.read_feature_file(
        training_file_names[0]
    )[-1]

    these_target_values = (
        first_training_target_table[utils.TARGET_NAME].values[:10]
    )

    message_string = (
        '\nReal-numbered target values ("{0:s}") the first few storm objects:'
        '\n{1:s}'
    ).format(utils.TARGET_NAME, str(these_target_values))
    print(message_string)

    these_target_values = utils.binarize_target_values(
        target_values=these_target_values,
        binarization_threshold=binarization_threshold)

    message_string = (
        '\nBinarized target values the first few storm objects:\n{0:s}'
    ).format(str(these_target_values))
    print(message_string)
