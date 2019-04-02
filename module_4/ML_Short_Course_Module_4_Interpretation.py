"""Code for AMS 2019 short course."""

import numpy
import matplotlib.pyplot as pyplot
from module_4 import utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'


def find_training_and_validation():
    """Finds training and validation data."""

    training_file_names = utils.find_many_image_files(
        first_date_string='20100101', last_date_string='20141231')

    validation_file_names = utils.find_many_image_files(
        first_date_string='20150101', last_date_string='20151231')


def read_validation(validation_file_names):
    """Reads validation data.

    :param validation_file_names: 1-D list of paths to input files.
    """

    validation_image_dict = utils.read_many_image_files(validation_file_names)

    print(MINOR_SEPARATOR_STRING)
    print('Variables in dictionary are as follows:')
    for this_key in validation_image_dict.keys():
        print(this_key)

    print('\nPredictor variables are as follows:')
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]
    for this_name in predictor_names:
        print(this_name)

    these_predictor_values = (
        validation_image_dict[utils.PREDICTOR_MATRIX_KEY][0, :5, :5, 0]
    )

    message_string = (
        '\nSome values of predictor variable "{0:s}" for first storm object:'
        '\n{1:s}'
    ).format(predictor_names[0], str(these_predictor_values))
    print(message_string)

    these_target_values = (
        validation_image_dict[utils.TARGET_MATRIX_KEY][0, :5, :5]
    )

    message_string = (
        '\nSome values of target variable "{0:s}" for first storm object:'
        '\n{1:s}'
    ).format(
        validation_image_dict[utils.TARGET_NAME_KEY], str(these_target_values)
    )

    print(message_string)


def plot_random_example(validation_image_dict):
    """Plots random example.

    :param validation_image_dict: Dictionary created by
        `utils.read_many_image_files`.
    """

    predictor_matrix = validation_image_dict[utils.PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]
    temperature_matrix_kelvins = predictor_matrix[
        ..., predictor_names.index(utils.TEMPERATURE_NAME)
    ]

    utils.plot_many_predictors_with_barbs(
        predictor_matrix=predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=numpy.percentile(temperature_matrix_kelvins, 1),
        max_colour_temp_kelvins=numpy.percentile(temperature_matrix_kelvins, 99)
    )

    pyplot.show()


def plot_strong_example(validation_image_dict):
    """Plots example with strong future rotation.

    :param validation_image_dict: Dictionary created by
        `utils.read_many_image_files`.
    """

    target_matrix_s01 = validation_image_dict[utils.TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    predictor_matrix = validation_image_dict[utils.PREDICTOR_MATRIX_KEY][
        example_index, ...]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]
    temperature_matrix_kelvins = predictor_matrix[
        ..., predictor_names.index(utils.TEMPERATURE_NAME)
    ]

    utils.plot_many_predictors_with_barbs(
        predictor_matrix=predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=numpy.percentile(temperature_matrix_kelvins, 1),
        max_colour_temp_kelvins=numpy.percentile(temperature_matrix_kelvins, 99)
    )

    pyplot.show()


def norm_and_denorm(training_file_names):
    """Finds and applies normalization parameters.

    :param training_file_names: 1-D list of paths to input files.
    """

    normalization_dict = utils.get_image_normalization_params(
        training_file_names)
    print(MINOR_SEPARATOR_STRING)

    first_training_image_dict = utils.read_image_file(training_file_names[0])

    predictor_names = first_training_image_dict[utils.PREDICTOR_NAMES_KEY]
    these_predictor_values = (
        first_training_image_dict[utils.PREDICTOR_MATRIX_KEY][0, :5, :5, 0]
    )

    print('\nOriginal values of "{0:s}" for first storm object:\n{1:s}'.format(
        predictor_names[0], str(these_predictor_values)
    ))

    first_training_image_dict[utils.PREDICTOR_MATRIX_KEY], _ = (
        utils.normalize_images(
            predictor_matrix=first_training_image_dict[
                utils.PREDICTOR_MATRIX_KEY],
            predictor_names=predictor_names,
            normalization_dict=normalization_dict)
    )

    these_predictor_values = (
        first_training_image_dict[utils.PREDICTOR_MATRIX_KEY][0, :5, :5, 0]
    )

    message_string = (
        '\nNormalized values of "{0:s}" for first storm object:\n{1:s}'
    ).format(predictor_names[0], str(these_predictor_values))
    print(message_string)

    first_training_image_dict[utils.PREDICTOR_MATRIX_KEY] = (
        utils.denormalize_images(
            predictor_matrix=first_training_image_dict[
                utils.PREDICTOR_MATRIX_KEY],
            predictor_names=predictor_names,
            normalization_dict=normalization_dict)
    )

    these_predictor_values = (
        first_training_image_dict[utils.PREDICTOR_MATRIX_KEY][0, :5, :5, 0]
    )

    message_string = (
        '\nDenormalized values of "{0:s}" for first storm object:\n{1:s}'
    ).format(predictor_names[0], str(these_predictor_values))
    print(message_string)


def binarization_example(training_file_names):
    """Finds and applies binarization threshold.

    :param training_file_names: 1-D list of paths to input files.
    """

    binarization_threshold = utils.get_binarization_threshold(
        netcdf_file_names=training_file_names, percentile_level=90.)
    print(MINOR_SEPARATOR_STRING)

    first_training_image_dict = utils.read_image_file(training_file_names[0])
    these_max_target_values = numpy.array([
        numpy.max(first_training_image_dict[utils.TARGET_MATRIX_KEY][i, ...])
        for i in range(10)
    ])

    message_string = (
        '\nSpatial maxima of "{0:s}" for the first few storm objects:\n{1:s}'
    ).format(
        first_training_image_dict[utils.TARGET_NAME_KEY],
        str(these_max_target_values)
    )

    print(message_string)

    target_values = utils.binarize_target_images(
        target_matrix=first_training_image_dict[utils.TARGET_MATRIX_KEY],
        binarization_threshold=binarization_threshold)

    message_string = (
        '\nBinarized target values for the first few storm objects:\n{0:s}'
    ).format(str(target_values[:10]))
    print(message_string)


def read_pretrained_cnn():
    """Reads pre-trained CNN from file."""

    cnn_file_name = '{0:s}/pretrained_cnn/pretrained_cnn.h5'.format(
        utils.MODULE4_DIR_NAME)

    cnn_model_object = utils.read_keras_model(cnn_file_name)
    cnn_model_object.summary()


def evaluate_cnn(cnn_file_name, cnn_model_object, validation_image_dict):
    """Evaluates CNN on validation data.

    :param cnn_file_name: Path to HDF5 file with trained CNN.
    :param cnn_model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param validation_image_dict: Dictionary created by
        `utils.read_many_image_files`.
    """

    cnn_metafile_name = utils.find_model_metafile(model_file_name=cnn_file_name)
    cnn_metadata_dict = utils.read_model_metadata(cnn_metafile_name)
    validation_dir_name = '{0:s}/validation'.format(utils.MODULE4_DIR_NAME)

    utils.evaluate_cnn(
        cnn_model_object=cnn_model_object, image_dict=validation_image_dict,
        cnn_metadata_dict=cnn_metadata_dict,
        output_dir_name=validation_dir_name)

    print(SEPARATOR_STRING)


def run_permutation_test(cnn_model_object, validation_image_dict,
                         cnn_metadata_dict):
    """Runs the permutation test on validation data.

    :param cnn_model_object: See doc for `permutation_test_for_cnn`.
    :param validation_image_dict: Same.
    :param cnn_metadata_dict: Same.
    """

    permutation_dir_name = '{0:s}/permutation_test'.format(
        utils.MODULE4_DIR_NAME)
    main_permutation_file_name = '{0:s}/permutation_results.p'.format(
        permutation_dir_name)

    permutation_dict = utils.permutation_test_for_cnn(
        cnn_model_object=cnn_model_object, image_dict=validation_image_dict,
        cnn_metadata_dict=cnn_metadata_dict,
        output_pickle_file_name=main_permutation_file_name)

    breiman_file_name = '{0:s}/breiman_results.jpg'.format(permutation_dir_name)
    utils.plot_breiman_results(
        result_dict=permutation_dict, output_file_name=breiman_file_name,
        plot_percent_increase=False)

    lakshmanan_file_name = '{0:s}/lakshmanan_results.jpg'.format(
        permutation_dir_name)
    utils.plot_lakshmanan_results(
        result_dict=permutation_dict, output_file_name=lakshmanan_file_name,
        plot_percent_increase=False)


def saliency_example1(validation_image_dict, normalization_dict,
                      cnn_model_object):
    """Computes saliency map for random example wrt positive-class probability.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    predictor_matrix = validation_image_dict[utils.PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = utils.normalize_images(
        predictor_matrix=predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    saliency_matrix = utils.get_saliency_for_class(
        cnn_model_object=cnn_model_object, target_class=1,
        list_of_input_matrices=[predictor_matrix_norm]
    )[0][0, ...]

    temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
    min_colour_temp_kelvins = numpy.percentile(
        predictor_matrix[..., temperature_index], 1)
    max_colour_temp_kelvins = numpy.percentile(
        predictor_matrix[..., temperature_index], 99)

    wind_indices = numpy.array([
        predictor_names.index(utils.U_WIND_NAME),
        predictor_names.index(utils.V_WIND_NAME)
    ], dtype=int)

    max_colour_wind_speed_m_s01 = numpy.percentile(
        numpy.absolute(predictor_matrix[..., wind_indices]), 99
    )

    _, axes_objects_2d_list = utils.plot_many_predictors_sans_barbs(
        predictor_matrix=predictor_matrix, predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins,
        max_colour_wind_speed_m_s01=max_colour_wind_speed_m_s01)

    max_absolute_contour_level = numpy.percentile(
        numpy.absolute(saliency_matrix), 99
    )

    utils.plot_many_saliency_maps(
        saliency_matrix=saliency_matrix,
        axes_objects_2d_list=axes_objects_2d_list,
        colour_map_object=utils.SALIENCY_COLOUR_MAP_OBJECT,
        max_absolute_contour_level=max_absolute_contour_level,
        contour_interval=max_absolute_contour_level / 10)

    pyplot.show()


def saliency_example2(validation_image_dict, normalization_dict,
                      cnn_model_object):
    """Computes saliency map for random example wrt negative-class probability.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    predictor_matrix = validation_image_dict[utils.PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = utils.normalize_images(
        predictor_matrix=predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    saliency_matrix = utils.get_saliency_for_class(
        cnn_model_object=cnn_model_object, target_class=0,
        list_of_input_matrices=[predictor_matrix_norm]
    )[0][0, ...]

    temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
    min_colour_temp_kelvins = numpy.percentile(
        predictor_matrix[..., temperature_index], 1)
    max_colour_temp_kelvins = numpy.percentile(
        predictor_matrix[..., temperature_index], 99)

    wind_indices = numpy.array([
        predictor_names.index(utils.U_WIND_NAME),
        predictor_names.index(utils.V_WIND_NAME)
    ], dtype=int)

    max_colour_wind_speed_m_s01 = numpy.percentile(
        numpy.absolute(predictor_matrix[..., wind_indices]), 99
    )

    _, axes_objects_2d_list = utils.plot_many_predictors_sans_barbs(
        predictor_matrix=predictor_matrix, predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins,
        max_colour_wind_speed_m_s01=max_colour_wind_speed_m_s01)

    max_absolute_contour_level = numpy.percentile(
        numpy.absolute(saliency_matrix), 99
    )

    utils.plot_many_saliency_maps(
        saliency_matrix=saliency_matrix,
        axes_objects_2d_list=axes_objects_2d_list,
        colour_map_object=utils.SALIENCY_COLOUR_MAP_OBJECT,
        max_absolute_contour_level=max_absolute_contour_level,
        contour_interval=max_absolute_contour_level / 10)

    pyplot.show()


def saliency_example3(validation_image_dict, normalization_dict,
                      cnn_model_object):
    """Computes saliency map for strong example wrt positive-class probability.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    target_matrix_s01 = validation_image_dict[utils.TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    predictor_matrix = validation_image_dict[utils.PREDICTOR_MATRIX_KEY][
        example_index, ...]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = utils.normalize_images(
        predictor_matrix=predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    saliency_matrix = utils.get_saliency_for_class(
        cnn_model_object=cnn_model_object, target_class=1,
        list_of_input_matrices=[predictor_matrix_norm]
    )[0][0, ...]

    temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
    min_colour_temp_kelvins = numpy.percentile(
        predictor_matrix[..., temperature_index], 1)
    max_colour_temp_kelvins = numpy.percentile(
        predictor_matrix[..., temperature_index], 99)

    wind_indices = numpy.array([
        predictor_names.index(utils.U_WIND_NAME),
        predictor_names.index(utils.V_WIND_NAME)
    ], dtype=int)

    max_colour_wind_speed_m_s01 = numpy.percentile(
        numpy.absolute(predictor_matrix[..., wind_indices]), 99
    )

    _, axes_objects_2d_list = utils.plot_many_predictors_sans_barbs(
        predictor_matrix=predictor_matrix, predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins,
        max_colour_wind_speed_m_s01=max_colour_wind_speed_m_s01)

    max_absolute_contour_level = numpy.percentile(
        numpy.absolute(saliency_matrix), 99
    )

    utils.plot_many_saliency_maps(
        saliency_matrix=saliency_matrix,
        axes_objects_2d_list=axes_objects_2d_list,
        colour_map_object=utils.SALIENCY_COLOUR_MAP_OBJECT,
        max_absolute_contour_level=max_absolute_contour_level,
        contour_interval=max_absolute_contour_level / 10)

    pyplot.show()


def saliency_example4(validation_image_dict, normalization_dict,
                      cnn_model_object):
    """Computes saliency map for weak example wrt positive-class probability.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    target_matrix_s01 = validation_image_dict[utils.TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmin(target_matrix_s01), target_matrix_s01.shape
    )[0]

    predictor_matrix = validation_image_dict[utils.PREDICTOR_MATRIX_KEY][
        example_index, ...]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = utils.normalize_images(
        predictor_matrix=predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    saliency_matrix = utils.get_saliency_for_class(
        cnn_model_object=cnn_model_object, target_class=1,
        list_of_input_matrices=[predictor_matrix_norm]
    )[0][0, ...]

    temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
    min_colour_temp_kelvins = numpy.percentile(
        predictor_matrix[..., temperature_index], 1)
    max_colour_temp_kelvins = numpy.percentile(
        predictor_matrix[..., temperature_index], 99)

    wind_indices = numpy.array([
        predictor_names.index(utils.U_WIND_NAME),
        predictor_names.index(utils.V_WIND_NAME)
    ], dtype=int)

    max_colour_wind_speed_m_s01 = numpy.percentile(
        numpy.absolute(predictor_matrix[..., wind_indices]), 99
    )

    _, axes_objects_2d_list = utils.plot_many_predictors_sans_barbs(
        predictor_matrix=predictor_matrix, predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins,
        max_colour_wind_speed_m_s01=max_colour_wind_speed_m_s01)

    max_absolute_contour_level = numpy.percentile(
        numpy.absolute(saliency_matrix), 99
    )

    utils.plot_many_saliency_maps(
        saliency_matrix=saliency_matrix,
        axes_objects_2d_list=axes_objects_2d_list,
        colour_map_object=utils.SALIENCY_COLOUR_MAP_OBJECT,
        max_absolute_contour_level=max_absolute_contour_level,
        contour_interval=max_absolute_contour_level / 10)

    pyplot.show()


def gradcam_example1(validation_image_dict, normalization_dict,
                     cnn_model_object):
    """Runs Grad-CAM for strong example wrt positive-class probability.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    target_matrix_s01 = validation_image_dict[utils.TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    predictor_matrix = validation_image_dict[utils.PREDICTOR_MATRIX_KEY][
        example_index, ...]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = utils.normalize_images(
        predictor_matrix=predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    target_layer_names = [
        'batch_normalization_1', 'batch_normalization_2',
        'batch_normalization_3', 'batch_normalization_4'
    ]

    for this_layer_name in target_layer_names:
        class_activation_matrix = utils.run_gradcam(
            model_object=cnn_model_object,
            list_of_input_matrices=[predictor_matrix_norm],
            target_class=1, target_layer_name=this_layer_name)

        temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
        min_colour_temp_kelvins = numpy.percentile(
            predictor_matrix[..., temperature_index], 1)
        max_colour_temp_kelvins = numpy.percentile(
            predictor_matrix[..., temperature_index], 99)

        wind_indices = numpy.array([
            predictor_names.index(utils.U_WIND_NAME),
            predictor_names.index(utils.V_WIND_NAME)
        ], dtype=int)

        max_colour_wind_speed_m_s01 = numpy.percentile(
            numpy.absolute(predictor_matrix[..., wind_indices]), 99
        )

        figure_object, axes_objects_2d_list = (
            utils.plot_many_predictors_sans_barbs(
                predictor_matrix=predictor_matrix,
                predictor_names=predictor_names,
                min_colour_temp_kelvins=min_colour_temp_kelvins,
                max_colour_temp_kelvins=max_colour_temp_kelvins,
                max_colour_wind_speed_m_s01=max_colour_wind_speed_m_s01)
        )

        dummy_saliency_matrix = numpy.expand_dims(
            class_activation_matrix, axis=-1)
        dummy_saliency_matrix = numpy.repeat(
            dummy_saliency_matrix, repeats=4, axis=-1)

        max_absolute_contour_level = numpy.percentile(
            numpy.absolute(dummy_saliency_matrix), 99
        )

        if max_absolute_contour_level == 0:
            max_absolute_contour_level = 10.

        utils.plot_many_saliency_maps(
            saliency_matrix=dummy_saliency_matrix,
            axes_objects_2d_list=axes_objects_2d_list,
            colour_map_object=utils.SALIENCY_COLOUR_MAP_OBJECT,
            max_absolute_contour_level=max_absolute_contour_level,
            contour_interval=max_absolute_contour_level / 10)

        figure_object.suptitle(
            'Class-activation map for layer "{0:s}"'.format(this_layer_name)
        )
        pyplot.show()


def gradcam_example2(validation_image_dict, normalization_dict,
                     cnn_model_object):
    """Runs Grad-CAM for random example wrt positive-class probability.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    predictor_matrix = validation_image_dict[utils.PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = utils.normalize_images(
        predictor_matrix=predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    target_layer_names = [
        'batch_normalization_1', 'batch_normalization_2',
        'batch_normalization_3', 'batch_normalization_4'
    ]

    for this_layer_name in target_layer_names:
        class_activation_matrix = utils.run_gradcam(
            model_object=cnn_model_object,
            list_of_input_matrices=[predictor_matrix_norm],
            target_class=1, target_layer_name=this_layer_name)

        temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
        min_colour_temp_kelvins = numpy.percentile(
            predictor_matrix[..., temperature_index], 1)
        max_colour_temp_kelvins = numpy.percentile(
            predictor_matrix[..., temperature_index], 99)

        wind_indices = numpy.array([
            predictor_names.index(utils.U_WIND_NAME),
            predictor_names.index(utils.V_WIND_NAME)
        ], dtype=int)

        max_colour_wind_speed_m_s01 = numpy.percentile(
            numpy.absolute(predictor_matrix[..., wind_indices]), 99
        )

        figure_object, axes_objects_2d_list = (
            utils.plot_many_predictors_sans_barbs(
                predictor_matrix=predictor_matrix,
                predictor_names=predictor_names,
                min_colour_temp_kelvins=min_colour_temp_kelvins,
                max_colour_temp_kelvins=max_colour_temp_kelvins,
                max_colour_wind_speed_m_s01=max_colour_wind_speed_m_s01)
        )

        dummy_saliency_matrix = numpy.expand_dims(
            class_activation_matrix, axis=-1)
        dummy_saliency_matrix = numpy.repeat(
            dummy_saliency_matrix, repeats=4, axis=-1)

        max_absolute_contour_level = numpy.percentile(
            numpy.absolute(dummy_saliency_matrix), 99
        )

        if max_absolute_contour_level == 0:
            max_absolute_contour_level = 10.

        utils.plot_many_saliency_maps(
            saliency_matrix=dummy_saliency_matrix,
            axes_objects_2d_list=axes_objects_2d_list,
            colour_map_object=utils.SALIENCY_COLOUR_MAP_OBJECT,
            max_absolute_contour_level=max_absolute_contour_level,
            contour_interval=max_absolute_contour_level / 10)

        figure_object.suptitle(
            'Class-activation map for layer "{0:s}"'.format(this_layer_name)
        )
        pyplot.show()


def gradcam_example3(validation_image_dict, normalization_dict,
                     cnn_model_object):
    """Runs Grad-CAM for random example wrt negative-class probability.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    predictor_matrix = validation_image_dict[utils.PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = utils.normalize_images(
        predictor_matrix=predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    target_layer_names = [
        'batch_normalization_1', 'batch_normalization_2',
        'batch_normalization_3', 'batch_normalization_4'
    ]

    for this_layer_name in target_layer_names:
        class_activation_matrix = utils.run_gradcam(
            model_object=cnn_model_object,
            list_of_input_matrices=[predictor_matrix_norm],
            target_class=0, target_layer_name=this_layer_name)

        temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
        min_colour_temp_kelvins = numpy.percentile(
            predictor_matrix[..., temperature_index], 1)
        max_colour_temp_kelvins = numpy.percentile(
            predictor_matrix[..., temperature_index], 99)

        wind_indices = numpy.array([
            predictor_names.index(utils.U_WIND_NAME),
            predictor_names.index(utils.V_WIND_NAME)
        ], dtype=int)

        max_colour_wind_speed_m_s01 = numpy.percentile(
            numpy.absolute(predictor_matrix[..., wind_indices]), 99
        )

        figure_object, axes_objects_2d_list = (
            utils.plot_many_predictors_sans_barbs(
                predictor_matrix=predictor_matrix,
                predictor_names=predictor_names,
                min_colour_temp_kelvins=min_colour_temp_kelvins,
                max_colour_temp_kelvins=max_colour_temp_kelvins,
                max_colour_wind_speed_m_s01=max_colour_wind_speed_m_s01)
        )

        dummy_saliency_matrix = numpy.expand_dims(
            class_activation_matrix, axis=-1)
        dummy_saliency_matrix = numpy.repeat(
            dummy_saliency_matrix, repeats=4, axis=-1)

        max_absolute_contour_level = numpy.percentile(
            numpy.absolute(dummy_saliency_matrix), 99
        )

        if max_absolute_contour_level == 0:
            max_absolute_contour_level = 10.

        utils.plot_many_saliency_maps(
            saliency_matrix=dummy_saliency_matrix,
            axes_objects_2d_list=axes_objects_2d_list,
            colour_map_object=utils.SALIENCY_COLOUR_MAP_OBJECT,
            max_absolute_contour_level=max_absolute_contour_level,
            contour_interval=max_absolute_contour_level / 10)

        figure_object.suptitle(
            'Class-activation map for layer "{0:s}"'.format(this_layer_name)
        )
        pyplot.show()


def bwo_example1(validation_image_dict, normalization_dict, cnn_model_object):
    """Optimizes random example (storm object) for positive class.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    orig_predictor_matrix = validation_image_dict[
        utils.PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    orig_predictor_matrix_norm, _ = utils.normalize_images(
        predictor_matrix=orig_predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    orig_predictor_matrix_norm = numpy.expand_dims(
        orig_predictor_matrix_norm, axis=0)

    optimized_predictor_matrix_norm = utils.bwo_for_class(
        cnn_model_object=cnn_model_object, target_class=1,
        init_function_or_matrices=[orig_predictor_matrix_norm]
    )[0][0, ...]

    optimized_predictor_matrix = utils.denormalize_images(
        predictor_matrix=optimized_predictor_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (orig_predictor_matrix[..., temperature_index],
         optimized_predictor_matrix[..., temperature_index]),
        axis=0
    )

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = utils.plot_many_predictors_with_barbs(
        predictor_matrix=orig_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Real example (before optimization)')
    pyplot.show()

    figure_object, _ = utils.plot_many_predictors_with_barbs(
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

    orig_predictor_matrix = validation_image_dict[
        utils.PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    orig_predictor_matrix_norm, _ = utils.normalize_images(
        predictor_matrix=orig_predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    orig_predictor_matrix_norm = numpy.expand_dims(
        orig_predictor_matrix_norm, axis=0)

    optimized_predictor_matrix_norm = utils.bwo_for_class(
        cnn_model_object=cnn_model_object, target_class=0,
        init_function_or_matrices=[orig_predictor_matrix_norm]
    )[0][0, ...]

    optimized_predictor_matrix = utils.denormalize_images(
        predictor_matrix=optimized_predictor_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (orig_predictor_matrix[..., temperature_index],
         optimized_predictor_matrix[..., temperature_index]),
        axis=0
    )

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = utils.plot_many_predictors_with_barbs(
        predictor_matrix=orig_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Real example (before optimization)')
    pyplot.show()

    figure_object, _ = utils.plot_many_predictors_with_barbs(
        predictor_matrix=optimized_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Synthetic example (after optimization)')
    pyplot.show()


def bwo_example3(validation_image_dict, normalization_dict, cnn_model_object):
    """Optimizes strong example (storm object) for positive class.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    target_matrix_s01 = validation_image_dict[utils.TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    orig_predictor_matrix = validation_image_dict[
        utils.PREDICTOR_MATRIX_KEY][example_index, ...]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    orig_predictor_matrix_norm, _ = utils.normalize_images(
        predictor_matrix=orig_predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    orig_predictor_matrix_norm = numpy.expand_dims(
        orig_predictor_matrix_norm, axis=0)

    optimized_predictor_matrix_norm = utils.bwo_for_class(
        cnn_model_object=cnn_model_object, target_class=1,
        init_function_or_matrices=[orig_predictor_matrix_norm]
    )[0][0, ...]

    optimized_predictor_matrix = utils.denormalize_images(
        predictor_matrix=optimized_predictor_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (orig_predictor_matrix[..., temperature_index],
         optimized_predictor_matrix[..., temperature_index]),
        axis=0
    )

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = utils.plot_many_predictors_with_barbs(
        predictor_matrix=orig_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Real example (before optimization)')
    pyplot.show()

    figure_object, _ = utils.plot_many_predictors_with_barbs(
        predictor_matrix=optimized_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Synthetic example (after optimization)')
    pyplot.show()


def bwo_example4(validation_image_dict, normalization_dict, cnn_model_object):
    """Optimizes strong example (storm object) for negative class.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    target_matrix_s01 = validation_image_dict[utils.TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    orig_predictor_matrix = validation_image_dict[
        utils.PREDICTOR_MATRIX_KEY][example_index, ...]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    orig_predictor_matrix_norm, _ = utils.normalize_images(
        predictor_matrix=orig_predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    orig_predictor_matrix_norm = numpy.expand_dims(
        orig_predictor_matrix_norm, axis=0)

    optimized_predictor_matrix_norm = utils.bwo_for_class(
        cnn_model_object=cnn_model_object, target_class=0,
        init_function_or_matrices=[orig_predictor_matrix_norm]
    )[0][0, ...]

    optimized_predictor_matrix = utils.denormalize_images(
        predictor_matrix=optimized_predictor_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (orig_predictor_matrix[..., temperature_index],
         optimized_predictor_matrix[..., temperature_index]),
        axis=0
    )

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = utils.plot_many_predictors_with_barbs(
        predictor_matrix=orig_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Real example (before optimization)')
    pyplot.show()

    figure_object, _ = utils.plot_many_predictors_with_barbs(
        predictor_matrix=optimized_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Synthetic example (after optimization)')
    pyplot.show()


def read_pretrained_ucn():
    """Reads pre-trained upconvnet from file."""

    ucn_file_name = '{0:s}/pretrained_cnn/pretrained_ucn.h5'.format(
        utils.MODULE4_DIR_NAME)
    ucn_metafile_name = utils.find_model_metafile(model_file_name=ucn_file_name)

    ucn_model_object = utils.read_keras_model(ucn_file_name)
    ucn_metadata_dict = utils.read_model_metadata(ucn_metafile_name)

    ucn_model_object.summary()


def apply_ucn_example1(
        validation_image_dict, normalization_dict, ucn_model_object,
        cnn_model_object):
    """Uses upconvnet to reconstruct random validation example.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param ucn_model_object: Trained upconvnet (instance of `keras.models.Model`
        or `keras.models.Sequential`).
    :param cnn_model_object: Trained instance of `keras.models.Model`,
        representing the CNN that goes with the upconvnet.
    """

    image_matrix = validation_image_dict[utils.PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    image_matrix_norm, _ = utils.normalize_images(
        predictor_matrix=image_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    image_matrix_norm = numpy.expand_dims(image_matrix_norm, axis=0)

    feature_matrix = utils.apply_cnn(
        cnn_model_object=cnn_model_object, predictor_matrix=image_matrix_norm,
        output_layer_name=utils.get_cnn_flatten_layer(cnn_model_object),
        verbose=False)

    reconstructed_image_matrix_norm = ucn_model_object.predict(
        feature_matrix, batch_size=1)

    reconstructed_image_matrix = utils.denormalize_images(
        predictor_matrix=reconstructed_image_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict
    )[0, ...]

    temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (image_matrix[..., temperature_index],
         reconstructed_image_matrix[..., temperature_index]),
        axis=0
    )

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = utils.plot_many_predictors_with_barbs(
        predictor_matrix=image_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Original image (CNN input)')
    pyplot.show()

    figure_object, _ = utils.plot_many_predictors_with_barbs(
        predictor_matrix=reconstructed_image_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Reconstructed image (upconvnet output)')
    pyplot.show()


def apply_ucn_example2(
        validation_image_dict, normalization_dict, ucn_model_object,
        cnn_model_object):
    """Uses upconvnet to reconstruct strongest validation example.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param ucn_model_object: Trained upconvnet (instance of `keras.models.Model`
        or `keras.models.Sequential`).
    :param cnn_model_object: Trained instance of `keras.models.Model`,
        representing the CNN that goes with the upconvnet.
    """

    target_matrix_s01 = validation_image_dict[utils.TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    image_matrix = validation_image_dict[utils.PREDICTOR_MATRIX_KEY][
        example_index, ...]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    image_matrix_norm, _ = utils.normalize_images(
        predictor_matrix=image_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    image_matrix_norm = numpy.expand_dims(image_matrix_norm, axis=0)

    feature_matrix = utils.apply_cnn(
        cnn_model_object=cnn_model_object, predictor_matrix=image_matrix_norm,
        output_layer_name=utils.get_cnn_flatten_layer(cnn_model_object),
        verbose=False)

    reconstructed_image_matrix_norm = ucn_model_object.predict(
        feature_matrix, batch_size=1)

    reconstructed_image_matrix = utils.denormalize_images(
        predictor_matrix=reconstructed_image_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict
    )[0, ...]

    temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (image_matrix[..., temperature_index],
         reconstructed_image_matrix[..., temperature_index]),
        axis=0
    )

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = utils.plot_many_predictors_with_barbs(
        predictor_matrix=image_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Original image (CNN input)')
    pyplot.show()

    figure_object, _ = utils.plot_many_predictors_with_barbs(
        predictor_matrix=reconstructed_image_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Reconstructed image (upconvnet output)')
    pyplot.show()


def novelty_detection_example(
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

    target_matrix_s01 = validation_image_dict[utils.TARGET_MATRIX_KEY]
    num_examples = target_matrix_s01.shape[0]

    max_target_by_example_s01 = numpy.array([
        numpy.max(target_matrix_s01[i, ...]) for i in range(num_examples)
    ])

    test_indices = numpy.argsort(-1 * max_target_by_example_s01)[:100]
    baseline_indices = numpy.argsort(-1 * max_target_by_example_s01)[100:]

    numpy.random.seed(6695)
    numpy.random.shuffle(baseline_indices)
    baseline_indices = baseline_indices[:100]

    novelty_dict = utils.do_novelty_detection(
        baseline_image_matrix=validation_image_dict[
            utils.PREDICTOR_MATRIX_KEY][baseline_indices, ...],
        test_image_matrix=validation_image_dict[
            utils.PREDICTOR_MATRIX_KEY][test_indices, ...],
        image_normalization_dict=normalization_dict,
        predictor_names=validation_image_dict[utils.PREDICTOR_NAMES_KEY],
        cnn_model_object=cnn_model_object,
        cnn_feature_layer_name=utils.get_cnn_flatten_layer(cnn_model_object),
        ucn_model_object=ucn_model_object,
        num_novel_test_images=4)


def plot_novelty_detection_example1(validation_image_dict, novelty_dict):
    """Plots first-most novel example, selon novelty detection.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param novelty_dict: Dictionary created by `do_novelty_detection`.
    """

    utils.plot_novelty_detection(image_dict=validation_image_dict,
                                 novelty_dict=novelty_dict, test_index=0)


def plot_novelty_detection_example2(validation_image_dict, novelty_dict):
    """Plots second-most novel example, selon novelty detection.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param novelty_dict: Dictionary created by `do_novelty_detection`.
    """

    utils.plot_novelty_detection(image_dict=validation_image_dict,
                                 novelty_dict=novelty_dict, test_index=1)


def plot_novelty_detection_example3(validation_image_dict, novelty_dict):
    """Plots third-most novel example, selon novelty detection.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param novelty_dict: Dictionary created by `do_novelty_detection`.
    """

    utils.plot_novelty_detection(image_dict=validation_image_dict,
                                 novelty_dict=novelty_dict, test_index=2)


def plot_novelty_detection_example4(validation_image_dict, novelty_dict):
    """Plots fourth-most novel example, selon novelty detection.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param novelty_dict: Dictionary created by `do_novelty_detection`.
    """

    utils.plot_novelty_detection(image_dict=validation_image_dict,
                                 novelty_dict=novelty_dict, test_index=3)
