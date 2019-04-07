"""Non-notebook version of Module 2 for AMS 2019 short course."""

import copy
import warnings
import numpy
import matplotlib.pyplot as pyplot
from module_2 import utils

warnings.filterwarnings('ignore')

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

MODULE2_DIR_NAME = '.'
SHORT_COURSE_DIR_NAME = '..'


def find_tvt_data():
    """Finds training, validation, and testing files."""

    training_file_names = utils.find_many_feature_files(
        first_date_string='20100101', last_date_string='20141231')

    validation_file_names = utils.find_many_feature_files(
        first_date_string='20150101', last_date_string='20151231')

    testing_file_names = utils.find_many_feature_files(
        first_date_string='20160101', last_date_string='20171231')


def read_tvt_data(training_file_names, validation_file_names,
                  testing_file_names):
    """Reads training, validation, and testing data.

    :param training_file_names: 1-D list of paths to training files.
    :param validation_file_names: 1-D list of paths to validation files.
    :param testing_file_names: 1-D list of paths to testing files.
    """

    (training_metadata_table, training_predictor_table_denorm,
     training_target_table
    ) = utils.read_many_feature_files(training_file_names)
    print(MINOR_SEPARATOR_STRING)

    (validation_metadata_table, validation_predictor_table_denorm,
     validation_target_table
    ) = utils.read_many_feature_files(validation_file_names)
    print(MINOR_SEPARATOR_STRING)

    (testing_metadata_table, testing_predictor_table_denorm,
     testing_target_table
    ) = utils.read_many_feature_files(testing_file_names)
    print(MINOR_SEPARATOR_STRING)

    print('Variables in metadata are as follows:\n{0:s}'.format(
        str(list(training_metadata_table))
    ))

    print('\nPredictor variables are as follows:\n{0:s}'.format(
        str(list(training_predictor_table_denorm))
    ))

    print('\nTarget variable is as follows:\n{0:s}'.format(
        str(list(training_target_table))
    ))

    first_predictor_name = list(training_predictor_table_denorm)[0]
    these_predictor_values = (
        training_predictor_table_denorm[first_predictor_name].values[:10]
    )

    message_string = (
        '\nValues of predictor variable "{0:s}" for the first training '
        'examples:\n{1:s}'
    ).format(first_predictor_name, str(these_predictor_values))
    print(message_string)

    target_name = list(training_target_table)[0]
    these_target_values = training_target_table[target_name].values[:10]

    message_string = (
        '\nValues of target variable for the first training examples:\n{0:s}'
    ).format(str(these_target_values))
    print(message_string)


def normalize_tvt_data(
        training_predictor_table_denorm, validation_predictor_table_denorm,
        testing_predictor_table_denorm):
    """Normalizes training, validation, and testing data.

    :param training_predictor_table_denorm: See doc for
        `utils.read_feature_file`.
    :param validation_predictor_table_denorm: Same.
    :param testing_predictor_table_denorm: Same.
    """

    predictor_names = list(training_predictor_table_denorm)
    these_predictor_values = (
        training_predictor_table_denorm[predictor_names[0]].values[:10]
    )

    message_string = (
        'Original values of "{0:s}" for the first training examples:\n{1:s}'
    ).format(predictor_names[0], str(these_predictor_values))
    print(message_string)

    training_predictor_table, normalization_dict = utils.normalize_predictors(
        predictor_table=copy.deepcopy(training_predictor_table_denorm)
    )

    these_predictor_values = (
        training_predictor_table[predictor_names[0]].values[:10]
    )

    message_string = (
        '\nNormalized values of "{0:s}" for the first training examples:\n{1:s}'
    ).format(predictor_names[0], str(these_predictor_values))
    print(message_string)

    training_predictor_table_denorm = utils.denormalize_predictors(
        predictor_table=copy.deepcopy(training_predictor_table),
        normalization_dict=normalization_dict
    )

    these_predictor_values = (
        training_predictor_table_denorm[predictor_names[0]].values[:10]
    )

    message_string = (
        '\n*De*normalized values (should equal original values) of "{0:s}" for '
        'the first training examples:\n{1:s}'
    ).format(predictor_names[0], str(these_predictor_values))
    print(message_string)

    validation_predictor_table, _ = utils.normalize_predictors(
        predictor_table=copy.deepcopy(validation_predictor_table_denorm),
        normalization_dict=normalization_dict)

    testing_predictor_table, _ = utils.normalize_predictors(
        predictor_table=copy.deepcopy(testing_predictor_table_denorm),
        normalization_dict=normalization_dict)


def train_linear_regression(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains plain linear regression.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    linreg_model_object = utils.setup_linear_regression(
        lambda1=0., lambda2=0.)

    _ = utils.train_linear_regression(
        model_object=linreg_model_object,
        training_predictor_table=training_predictor_table,
        training_target_table=training_target_table)

    training_predictions = linreg_model_object.predict(
        training_predictor_table.as_matrix()
    )
    mean_training_target_value = numpy.mean(
        training_target_table[utils.TARGET_NAME].values
    )

    _ = utils.evaluate_regression(
        target_values=training_target_table[utils.TARGET_NAME].values,
        predicted_target_values=training_predictions,
        mean_training_target_value=mean_training_target_value,
        dataset_name='training')
    print(MINOR_SEPARATOR_STRING)

    validation_predictions = linreg_model_object.predict(
        validation_predictor_table.as_matrix()
    )

    _ = utils.evaluate_regression(
        target_values=validation_target_table[utils.TARGET_NAME].values,
        predicted_target_values=validation_predictions,
        mean_training_target_value=mean_training_target_value,
        dataset_name='validation')


def plot_linear_regression_coeffs(linreg_model_object,
                                  training_predictor_table):
    """Plots coefficients for plain linear regression.

    :param linreg_model_object: Trained instance of `sklearn.linear_model`.
    :param training_predictor_table: See doc for `utils.read_feature_file`.
    """

    utils.plot_model_coefficients(
        model_object=linreg_model_object,
        predictor_names=list(training_predictor_table)
    )

    pyplot.show()


def train_linear_ridge(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains linear regression with ridge penalty.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    linear_ridge_model_object = utils.setup_linear_regression(
        lambda1=0., lambda2=1e5)

    _ = utils.train_linear_regression(
        model_object=linear_ridge_model_object,
        training_predictor_table=training_predictor_table,
        training_target_table=training_target_table)

    training_predictions = linear_ridge_model_object.predict(
        training_predictor_table.as_matrix()
    )
    mean_training_target_value = numpy.mean(
        training_target_table[utils.TARGET_NAME].values
    )

    _ = utils.evaluate_regression(
        target_values=training_target_table[utils.TARGET_NAME].values,
        predicted_target_values=training_predictions,
        mean_training_target_value=mean_training_target_value,
        dataset_name='training')
    print(MINOR_SEPARATOR_STRING)

    validation_predictions = linear_ridge_model_object.predict(
        validation_predictor_table.as_matrix()
    )

    _ = utils.evaluate_regression(
        target_values=validation_target_table[utils.TARGET_NAME].values,
        predicted_target_values=validation_predictions,
        mean_training_target_value=mean_training_target_value,
        dataset_name='validation')


def plot_linear_ridge_coeffs(linear_ridge_model_object,
                             training_predictor_table):
    """Plots coefficients for linear regression with ridge penalty.

    :param linear_ridge_model_object: Trained instance of
        `sklearn.linear_model`.
    :param training_predictor_table: See doc for `utils.read_feature_file`.
    """

    utils.plot_model_coefficients(
        model_object=linear_ridge_model_object,
        predictor_names=list(training_predictor_table)
    )

    pyplot.show()


def train_linear_lasso(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains linear regression with lasso penalty.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    linear_lasso_model_object = utils.setup_linear_regression(
        lambda1=1e-5, lambda2=0.)

    _ = utils.train_linear_regression(
        model_object=linear_lasso_model_object,
        training_predictor_table=training_predictor_table,
        training_target_table=training_target_table)

    training_predictions = linear_lasso_model_object.predict(
        training_predictor_table.as_matrix()
    )
    mean_training_target_value = numpy.mean(
        training_target_table[utils.TARGET_NAME].values
    )

    _ = utils.evaluate_regression(
        target_values=training_target_table[utils.TARGET_NAME].values,
        predicted_target_values=training_predictions,
        mean_training_target_value=mean_training_target_value,
        dataset_name='training')
    print(MINOR_SEPARATOR_STRING)

    validation_predictions = linear_lasso_model_object.predict(
        validation_predictor_table.as_matrix()
    )

    _ = utils.evaluate_regression(
        target_values=validation_target_table[utils.TARGET_NAME].values,
        predicted_target_values=validation_predictions,
        mean_training_target_value=mean_training_target_value,
        dataset_name='validation')


def plot_linear_lasso_coeffs(linear_lasso_model_object,
                             training_predictor_table):
    """Plots coefficients for linear regression with lasso penalty.

    :param linear_lasso_model_object: Trained instance of `sklearn.linear_model`.
    :param training_predictor_table: See doc for `utils.read_feature_file`.
    """

    utils.plot_model_coefficients(
        model_object=linear_lasso_model_object,
        predictor_names=list(training_predictor_table)
    )

    pyplot.show()


def train_linear_elastic_net(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains linear regression with elastic-net penalty.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    linear_en_model_object = utils.setup_linear_regression(
        lambda1=1e-5, lambda2=5.)

    _ = utils.train_linear_regression(
        model_object=linear_en_model_object,
        training_predictor_table=training_predictor_table,
        training_target_table=training_target_table)

    training_predictions = linear_en_model_object.predict(
        training_predictor_table.as_matrix()
    )
    mean_training_target_value = numpy.mean(
        training_target_table[utils.TARGET_NAME].values
    )

    _ = utils.evaluate_regression(
        target_values=training_target_table[utils.TARGET_NAME].values,
        predicted_target_values=training_predictions,
        mean_training_target_value=mean_training_target_value,
        dataset_name='training')
    print(MINOR_SEPARATOR_STRING)

    validation_predictions = linear_en_model_object.predict(
        validation_predictor_table.as_matrix()
    )

    _ = utils.evaluate_regression(
        target_values=validation_target_table[utils.TARGET_NAME].values,
        predicted_target_values=validation_predictions,
        mean_training_target_value=mean_training_target_value,
        dataset_name='validation')


def plot_linear_en_coeffs(linear_en_model_object, training_predictor_table):
    """Plots coefficients for linear regression with elastic-net penalty.

    :param linear_en_model_object: Trained instance of `sklearn.linear_model`.
    :param training_predictor_table: See doc for `utils.read_feature_file`.
    """

    utils.plot_model_coefficients(
        model_object=linear_en_model_object,
        predictor_names=list(training_predictor_table)
    )

    pyplot.show()


def l1l2_experiment_training(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains models for hyperparameter experiment with L1/L2 regularization.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    lambda1_values = numpy.logspace(-8, -4, num=9)
    lambda2_values = numpy.logspace(-4, 1, num=11)

    num_lambda1 = len(lambda1_values)
    num_lambda2 = len(lambda2_values)

    validation_mae_matrix_s01 = numpy.full(
        (num_lambda1, num_lambda2), numpy.nan
    )
    validation_mse_matrix_s02 = numpy.full(
        (num_lambda1, num_lambda2), numpy.nan
    )
    validation_mae_skill_matrix = numpy.full(
        (num_lambda1, num_lambda2), numpy.nan
    )
    validation_mse_skill_matrix = numpy.full(
        (num_lambda1, num_lambda2), numpy.nan
    )

    mean_training_target_value = numpy.mean(
        training_target_table[utils.TARGET_NAME].values
    )

    for i in range(num_lambda1):
        for j in range(num_lambda2):
            this_message_string = (
                'Training model with lasso coeff = 10^{0:.1f}, ridge coeff = '
                '10^{1:.1f}...'
            ).format(
                numpy.log10(lambda1_values[i]), numpy.log10(lambda2_values[j])
            )

            print(this_message_string)

            this_model_object = utils.setup_linear_regression(
                lambda1=lambda1_values[i], lambda2=lambda2_values[j]
            )

            _ = utils.train_linear_regression(
                model_object=this_model_object,
                training_predictor_table=training_predictor_table,
                training_target_table=training_target_table)

            these_validation_predictions = this_model_object.predict(
                validation_predictor_table.as_matrix()
            )

            this_evaluation_dict = utils.evaluate_regression(
                target_values=validation_target_table[utils.TARGET_NAME].values,
                predicted_target_values=these_validation_predictions,
                mean_training_target_value=mean_training_target_value,
                verbose=False, create_plots=False)

            validation_mae_matrix_s01[i, j] = this_evaluation_dict[
                utils.MAE_KEY]
            validation_mse_matrix_s02[i, j] = this_evaluation_dict[
                utils.MSE_KEY]
            validation_mae_skill_matrix[i, j] = this_evaluation_dict[
                utils.MAE_SKILL_SCORE_KEY]
            validation_mse_skill_matrix[i, j] = this_evaluation_dict[
                utils.MSE_SKILL_SCORE_KEY]


def l1l2_experiment_validation(
        lambda1_values, lambda2_values, validation_mae_matrix_s01,
        validation_mse_matrix_s02, validation_mae_skill_matrix,
        validation_mse_skill_matrix):
    """Validates models for hyperparameter experiment with L1/L2 regularization.

    M = number of lambda_1 values
    N = number of lambda_2 values

    :param lambda1_values: length-M numpy array of lambda_1 values.
    :param lambda2_values: length-N numpy array of lambda_2 values.
    :param validation_mae_matrix_s01: M-by-N numpy array of mean absolute errors
        (s^-1) on validation data.
    :param validation_mse_matrix_s02: M-by-N numpy array of mean squared errors
        (s^-2) on validation data.
    :param validation_mae_skill_matrix: M-by-N numpy array of MAE skill scores
        on validation data.
    :param validation_mse_skill_matrix: M-by-N numpy array of MSE skill scores
        on validation data.
    """

    utils.plot_scores_2d(
        score_matrix=validation_mae_matrix_s01,
        min_colour_value=numpy.percentile(validation_mae_matrix_s01, 1.),
        max_colour_value=numpy.percentile(validation_mae_matrix_s01, 99.),
        x_tick_labels=numpy.log10(lambda2_values),
        y_tick_labels=numpy.log10(lambda1_values)
    )

    pyplot.xlabel(r'log$_{10}$ of ridge coefficient ($\lambda_2$)')
    pyplot.ylabel(r'log$_{10}$ of lasso coefficient ($\lambda_1$)')
    pyplot.title(r'Mean absolute error (s$^{-1}$) on validation data')

    utils.plot_scores_2d(
        score_matrix=validation_mse_matrix_s02,
        min_colour_value=numpy.percentile(validation_mse_matrix_s02, 1.),
        max_colour_value=numpy.percentile(validation_mse_matrix_s02, 99.),
        x_tick_labels=numpy.log10(lambda2_values),
        y_tick_labels=numpy.log10(lambda1_values)
    )

    pyplot.xlabel(r'log$_{10}$ of ridge coefficient ($\lambda_2$)')
    pyplot.ylabel(r'log$_{10}$ of lasso coefficient ($\lambda_1$)')
    pyplot.title(r'Mean squared error (s$^{-2}$) on validation data')

    utils.plot_scores_2d(
        score_matrix=validation_mae_skill_matrix,
        min_colour_value=numpy.percentile(validation_mae_skill_matrix, 1.),
        max_colour_value=numpy.percentile(validation_mae_skill_matrix, 99.),
        x_tick_labels=numpy.log10(lambda2_values),
        y_tick_labels=numpy.log10(lambda1_values)
    )

    pyplot.xlabel(r'log$_{10}$ of ridge coefficient ($\lambda_2$)')
    pyplot.ylabel(r'log$_{10}$ of lasso coefficient ($\lambda_1$)')
    pyplot.title(r'MAE skill score on validation data')

    utils.plot_scores_2d(
        score_matrix=validation_mse_skill_matrix,
        min_colour_value=numpy.percentile(validation_mse_skill_matrix, 1.),
        max_colour_value=numpy.percentile(validation_mse_skill_matrix, 99.),
        x_tick_labels=numpy.log10(lambda2_values),
        y_tick_labels=numpy.log10(lambda1_values)
    )

    pyplot.xlabel(r'log$_{10}$ of ridge coefficient ($\lambda_2$)')
    pyplot.ylabel(r'log$_{10}$ of lasso coefficient ($\lambda_1$)')
    pyplot.title(r'MSE skill score on validation data')


def l1l2_experiment_testing(
        lambda1_values, lambda2_values, validation_mae_skill_matrix,
        training_predictor_table, training_target_table,
        testing_predictor_table, testing_target_table):
    """Selects and tests model for experiment with L1/L2 regularization.

    :param lambda1_values: See doc for `l1l2_experiment_validation`.
    :param lambda2_values: Same.
    :param validation_mae_skill_matrix: Same.
    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param testing_predictor_table: Same.
    :param testing_target_table: Same.
    """

    best_linear_index = numpy.argmax(numpy.ravel(validation_mae_skill_matrix))

    best_lambda1_index, best_lambda2_index = numpy.unravel_index(
        best_linear_index, (len(lambda1_values), len(lambda2_values))
    )

    best_lambda1 = lambda1_values[best_lambda1_index]
    best_lambda2 = lambda2_values[best_lambda2_index]
    best_validation_maess = numpy.max(validation_mae_skill_matrix)

    message_string = (
        'Best MAE skill score on validation data = {0:.3f} ... corresponding '
        'lasso coeff = 10^{1:.1f}, ridge coeff = 10^{2:.1f}'
    ).format(
        best_validation_maess, numpy.log10(best_lambda1),
        numpy.log10(best_lambda2)
    )

    print(message_string)

    final_model_object = utils.setup_linear_regression(
        lambda1=best_lambda1, lambda2=best_lambda2)

    _ = utils.train_linear_regression(
        model_object=final_model_object,
        training_predictor_table=training_predictor_table,
        training_target_table=training_target_table)

    testing_predictions = final_model_object.predict(
        testing_predictor_table.as_matrix()
    )
    mean_training_target_value = numpy.mean(
        training_target_table[utils.TARGET_NAME].values
    )

    this_evaluation_dict = utils.evaluate_regression(
        target_values=testing_target_table[utils.TARGET_NAME].values,
        predicted_target_values=testing_predictions,
        mean_training_target_value=mean_training_target_value,
        dataset_name='testing')


def binarize_tvt_data(training_file_names, training_target_table,
                      validation_target_table, testing_target_table):
    """Binarizes target variable in training, validation, and testing data.

    :param training_file_names: 1-D list of paths to training files.
    :param training_target_table: See doc for `utils.read_feature_file`.
    :param validation_target_table: Same.
    :param testing_target_table: Same.
    """

    binarization_threshold = utils.get_binarization_threshold(
        csv_file_names=training_file_names, percentile_level=90.)
    print(MINOR_SEPARATOR_STRING)

    these_target_values = (
        training_target_table[utils.TARGET_NAME].values[:10]
    )

    message_string = (
        'Real-numbered target values for the first training examples:\n{0:s}'
    ).format(str(these_target_values))
    print(message_string)

    training_target_values = utils.binarize_target_values(
        target_values=training_target_table[utils.TARGET_NAME].values,
        binarization_threshold=binarization_threshold)

    training_target_table = training_target_table.assign(
        **{utils.BINARIZED_TARGET_NAME: training_target_values}
    )

    print('\nBinarization threshold = {0:.3e} s^-1'.format(
        binarization_threshold
    ))

    these_target_values = (
        training_target_table[utils.TARGET_NAME].values[:10]
    )

    message_string = (
        '\nBinarized target values for the first training examples:\n{0:s}'
    ).format(str(these_target_values))
    print(message_string)

    validation_target_values = utils.binarize_target_values(
        target_values=validation_target_table[utils.TARGET_NAME].values,
        binarization_threshold=binarization_threshold)

    validation_target_table = validation_target_table.assign(
        **{utils.BINARIZED_TARGET_NAME: validation_target_values}
    )

    testing_target_values = utils.binarize_target_values(
        target_values=testing_target_table[utils.TARGET_NAME].values,
        binarization_threshold=binarization_threshold)

    testing_target_table = testing_target_table.assign(
        **{utils.BINARIZED_TARGET_NAME: testing_target_values}
    )


def train_logistic_model(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains plain logistic-regression model.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    plain_log_model_object = utils.setup_logistic_regression(
        lambda1=0., lambda2=0.)

    _ = utils.train_logistic_regression(
        model_object=plain_log_model_object,
        training_predictor_table=training_predictor_table,
        training_target_table=training_target_table)

    training_probabilities = plain_log_model_object.predict_proba(
        training_predictor_table.as_matrix()
    )[:, 1]
    training_event_frequency = numpy.mean(
        training_target_table[utils.BINARIZED_TARGET_NAME].values
    )

    utils.eval_binary_classifn(
        observed_labels=training_target_table[
            utils.BINARIZED_TARGET_NAME].values,
        forecast_probabilities=training_probabilities,
        training_event_frequency=training_event_frequency,
        dataset_name='training')

    validation_probabilities = plain_log_model_object.predict_proba(
        validation_predictor_table.as_matrix()
    )[:, 1]

    utils.eval_binary_classifn(
        observed_labels=validation_target_table[
            utils.BINARIZED_TARGET_NAME].values,
        forecast_probabilities=validation_probabilities,
        training_event_frequency=training_event_frequency,
        dataset_name='validation')


def plot_logistic_regression_coeffs(plain_log_model_object,
                                    training_predictor_table):
    """Plots coefficients for plain logistic regression.

    :param plain_log_model_object: Trained instance of
        `sklearn.linear_model.SGDClassifier`.
    :param training_predictor_table: See doc for `utils.read_feature_file`.
    """

    utils.plot_model_coefficients(
        model_object=plain_log_model_object,
        predictor_names=list(training_predictor_table)
    )

    pyplot.show()


def train_logistic_elastic_net(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains logistic regression with elastic-net penalty.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    logistic_en_model_object = utils.setup_logistic_regression(
        lambda1=1e-3, lambda2=1e-3)

    _ = utils.train_logistic_regression(
        model_object=logistic_en_model_object,
        training_predictor_table=training_predictor_table,
        training_target_table=training_target_table)

    validation_probabilities = logistic_en_model_object.predict_proba(
        validation_predictor_table.as_matrix()
    )[:, 1]
    training_event_frequency = numpy.mean(
        training_target_table[utils.BINARIZED_TARGET_NAME].values
    )

    utils.eval_binary_classifn(
        observed_labels=validation_target_table[
            utils.BINARIZED_TARGET_NAME].values,
        forecast_probabilities=validation_probabilities,
        training_event_frequency=training_event_frequency,
        dataset_name='validation')


def plot_logistic_en_coeffs(logistic_en_model_object, training_predictor_table):
    """Plots coefficients for logistic regression with elastic-net penalty.

    :param logistic_en_model_object: Trained instance of
        `sklearn.linear_model.SGDClassifier`.
    :param training_predictor_table: See doc for `utils.read_feature_file`.
    """

    utils.plot_model_coefficients(
        model_object=logistic_en_model_object,
        predictor_names=list(training_predictor_table)
    )

    pyplot.show()


def train_tree_default(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains decision tree with default params.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    default_tree_model_object = utils.setup_classification_tree(
        min_examples_at_split=30, min_examples_at_leaf=30)

    _ = utils.train_classification_tree(
        model_object=default_tree_model_object,
        training_predictor_table=training_predictor_table,
        training_target_table=training_target_table)

    training_probabilities = default_tree_model_object.predict_proba(
        training_predictor_table.as_matrix()
    )[:, 1]
    training_event_frequency = numpy.mean(
        training_target_table[utils.BINARIZED_TARGET_NAME].values
    )

    utils.eval_binary_classifn(
        observed_labels=training_target_table[
            utils.BINARIZED_TARGET_NAME].values,
        forecast_probabilities=training_probabilities,
        training_event_frequency=training_event_frequency,
        dataset_name='training')

    validation_probabilities = default_tree_model_object.predict_proba(
        validation_predictor_table.as_matrix()
    )[:, 1]

    utils.eval_binary_classifn(
        observed_labels=validation_target_table[
            utils.BINARIZED_TARGET_NAME].values,
        forecast_probabilities=validation_probabilities,
        training_event_frequency=training_event_frequency,
        dataset_name='validation')


def tree_experiment_training(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains decision trees for experiment with min examples per split/leaf.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    min_per_split_values = numpy.array(
        [2, 5, 10, 20, 30, 40, 50, 100, 200, 500], dtype=int)
    min_per_leaf_values = numpy.array(
        [1, 5, 10, 20, 30, 40, 50, 100, 200, 500], dtype=int)

    num_min_per_split_values = len(min_per_split_values)
    num_min_per_leaf_values = len(min_per_leaf_values)

    validation_auc_matrix = numpy.full(
        (num_min_per_split_values, num_min_per_leaf_values), numpy.nan
    )

    validation_max_csi_matrix = validation_auc_matrix + 0.
    validation_bs_matrix = validation_auc_matrix + 0.
    validation_bss_matrix = validation_auc_matrix + 0.

    training_event_frequency = numpy.mean(
        training_target_table[utils.BINARIZED_TARGET_NAME].values
    )

    for i in range(num_min_per_split_values):
        for j in range(num_min_per_leaf_values):
            if min_per_leaf_values[j] >= min_per_split_values[i]:
                continue
            
            this_message_string = (
                'Training model with minima of {0:d} examples per split node, '
                '{1:d} per leaf node...'
            ).format(min_per_split_values[i], min_per_leaf_values[j])

            print(this_message_string)

            this_model_object = utils.setup_classification_tree(
                min_examples_at_split=min_per_split_values[i],
                min_examples_at_leaf=min_per_leaf_values[j]
            )

            _ = utils.train_classification_tree(
                model_object=this_model_object,
                training_predictor_table=training_predictor_table,
                training_target_table=training_target_table)

            these_validation_predictions = this_model_object.predict_proba(
                validation_predictor_table.as_matrix()
            )[:, 1]

            this_evaluation_dict = utils.eval_binary_classifn(
                observed_labels=validation_target_table[
                    utils.BINARIZED_TARGET_NAME].values,
                forecast_probabilities=these_validation_predictions,
                training_event_frequency=training_event_frequency,
                create_plots=False, verbose=False)

            validation_auc_matrix[i, j] = this_evaluation_dict[utils.AUC_KEY]
            validation_max_csi_matrix[i, j] = this_evaluation_dict[
                utils.MAX_CSI_KEY]
            validation_bs_matrix[i, j] = this_evaluation_dict[
                utils.BRIER_SCORE_KEY]
            validation_bss_matrix[i, j] = this_evaluation_dict[
                utils.BRIER_SKILL_SCORE_KEY]


def tree_experiment_validation(
        min_per_split_values, min_per_leaf_values, validation_auc_matrix,
        validation_max_csi_matrix, validation_bs_matrix,
        validation_bss_matrix):
    """Validates decision trees for experiment with min examples per split/leaf.

    M = number of min-per-split values
    N = number of min-per-leaf values

    :param min_per_split_values: length-M numpy array of minimum example counts
        per split node.
    :param min_per_leaf_values: length-N numpy array of minimum example counts
        per leaf node.
    :param validation_auc_matrix: M-by-N numpy array of areas under ROC curve on
        validation data.
    :param validation_max_csi_matrix: M-by-N numpy array of maximum CSIs on
        validation data.
    :param validation_bs_matrix: M-by-N numpy array of Brier scores on
        validation data.
    :param validation_bss_matrix: M-by-N numpy array of Brier skill scores on
        validation data.
    """

    utils.plot_scores_2d(
        score_matrix=validation_auc_matrix,
        min_colour_value=numpy.nanpercentile(validation_auc_matrix, 1.),
        max_colour_value=numpy.nanpercentile(validation_auc_matrix, 99.),
        x_tick_labels=min_per_leaf_values,
        y_tick_labels=min_per_split_values
    )

    pyplot.xlabel('Min num examples at leaf node')
    pyplot.ylabel('Min num examples at split node')
    pyplot.title('AUC (area under ROC curve) on validation data')

    utils.plot_scores_2d(
        score_matrix=validation_max_csi_matrix,
        min_colour_value=numpy.nanpercentile(validation_max_csi_matrix, 1.),
        max_colour_value=numpy.nanpercentile(validation_max_csi_matrix, 99.),
        x_tick_labels=min_per_leaf_values,
        y_tick_labels=min_per_split_values
    )

    pyplot.xlabel('Min num examples at leaf node')
    pyplot.ylabel('Min num examples at split node')
    pyplot.title('Max CSI (critical success index) on validation data')

    utils.plot_scores_2d(
        score_matrix=validation_bs_matrix,
        min_colour_value=numpy.nanpercentile(validation_bs_matrix, 1.),
        max_colour_value=numpy.nanpercentile(validation_bs_matrix, 99.),
        x_tick_labels=min_per_leaf_values,
        y_tick_labels=min_per_split_values
    )

    pyplot.xlabel('Min num examples at leaf node')
    pyplot.ylabel('Min num examples at split node')
    pyplot.title('Brier score on validation data')

    utils.plot_scores_2d(
        score_matrix=validation_bss_matrix,
        min_colour_value=numpy.nanpercentile(validation_bss_matrix, 1.),
        max_colour_value=numpy.nanpercentile(validation_bss_matrix, 99.),
        x_tick_labels=min_per_leaf_values,
        y_tick_labels=min_per_split_values
    )

    pyplot.xlabel('Min num examples at leaf node')
    pyplot.ylabel('Min num examples at split node')
    pyplot.title('Brier skill score on validation data')


def tree_experiment_testing(
        min_per_split_values, min_per_leaf_values, validation_bss_matrix,
        training_predictor_table, training_target_table,
        testing_predictor_table, testing_target_table):
    """Selects and tests tree for experiment with min examples per split/leaf.

    :param min_per_split_values: See doc for `tree_experiment_validation`.
    :param min_per_leaf_values: Same.
    :param validation_bss_matrix: Same.
    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param testing_predictor_table: Same.
    :param testing_target_table: Same.
    """

    best_linear_index = numpy.nanargmax(numpy.ravel(validation_bss_matrix))

    best_split_index, best_leaf_index = numpy.unravel_index(
        best_linear_index, validation_bss_matrix.shape)

    best_min_examples_per_split = min_per_split_values[best_split_index]
    best_min_examples_per_leaf = min_per_leaf_values[best_leaf_index]
    best_validation_bss = numpy.nanmax(validation_bss_matrix)

    message_string = (
        'Best validation BSS = {0:.3f} ... corresponding min examples per split'
        ' node = {1:d} ... min examples per leaf node = {2:d}'
    ).format(
        best_validation_bss, best_min_examples_per_split,
        best_min_examples_per_leaf
    )

    print(message_string)

    final_model_object = utils.setup_classification_tree(
        min_examples_at_split=best_min_examples_per_split,
        min_examples_at_leaf=best_min_examples_per_leaf
    )

    _ = utils.train_classification_tree(
        model_object=final_model_object,
        training_predictor_table=training_predictor_table,
        training_target_table=training_target_table)

    testing_predictions = final_model_object.predict_proba(
        testing_predictor_table.as_matrix()
    )[:, 1]
    training_event_frequency = numpy.mean(
        training_target_table[utils.BINARIZED_TARGET_NAME].values
    )

    _ = utils.eval_binary_classifn(
        observed_labels=testing_target_table[
            utils.BINARIZED_TARGET_NAME].values,
        forecast_probabilities=testing_predictions,
        training_event_frequency=training_event_frequency,
        create_plots=True, verbose=True, dataset_name='testing')


def train_random_forest(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains random forest.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    num_predictors = len(list(training_predictor_table))
    max_predictors_per_split = int(numpy.round(
        numpy.sqrt(num_predictors)
    ))

    random_forest_model_object = utils.setup_classification_forest(
        max_predictors_per_split=max_predictors_per_split,
        num_trees=100, min_examples_at_split=500, min_examples_at_leaf=200)

    _ = utils.train_classification_forest(
        model_object=random_forest_model_object,
        training_predictor_table=training_predictor_table,
        training_target_table=training_target_table)

    training_probabilities = random_forest_model_object.predict_proba(
        training_predictor_table.as_matrix()
    )[:, 1]
    training_event_frequency = numpy.mean(
        training_target_table[utils.BINARIZED_TARGET_NAME].values
    )

    utils.eval_binary_classifn(
        observed_labels=training_target_table[
            utils.BINARIZED_TARGET_NAME].values,
        forecast_probabilities=training_probabilities,
        training_event_frequency=training_event_frequency,
        dataset_name='training')

    validation_probabilities = random_forest_model_object.predict_proba(
        validation_predictor_table.as_matrix()
    )[:, 1]

    utils.eval_binary_classifn(
        observed_labels=validation_target_table[
            utils.BINARIZED_TARGET_NAME].values,
        forecast_probabilities=validation_probabilities,
        training_event_frequency=training_event_frequency,
        dataset_name='validation')


def train_gradient_boosted_trees(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains gradient-boosted trees.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    num_predictors = len(list(training_predictor_table))
    # max_predictors_per_split = int(numpy.round(
    #     numpy.sqrt(num_predictors)
    # ))

    gbt_model_object = utils.setup_classification_forest(
        max_predictors_per_split=num_predictors, num_trees=100,
        min_examples_at_split=500, min_examples_at_leaf=200)

    _ = utils.train_classification_gbt(
        model_object=gbt_model_object,
        training_predictor_table=training_predictor_table,
        training_target_table=training_target_table)

    training_probabilities = gbt_model_object.predict_proba(
        training_predictor_table.as_matrix()
    )[:, 1]
    training_event_frequency = numpy.mean(
        training_target_table[utils.BINARIZED_TARGET_NAME].values
    )

    utils.eval_binary_classifn(
        observed_labels=training_target_table[
            utils.BINARIZED_TARGET_NAME].values,
        forecast_probabilities=training_probabilities,
        training_event_frequency=training_event_frequency,
        dataset_name='training')

    validation_probabilities = gbt_model_object.predict_proba(
        validation_predictor_table.as_matrix()
    )[:, 1]

    utils.eval_binary_classifn(
        observed_labels=validation_target_table[
            utils.BINARIZED_TARGET_NAME].values,
        forecast_probabilities=validation_probabilities,
        training_event_frequency=training_event_frequency,
        dataset_name='validation')
