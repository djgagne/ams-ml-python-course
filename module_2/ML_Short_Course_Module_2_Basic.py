"""Non-notebook version of Module 2 for AMS 2019 short course."""

import copy
import numpy
import matplotlib.pyplot as pyplot
import sklearn.tree
from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image
from module_2 import utils

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


def linear_regression_example(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains linear-regression model.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    plain_linear_model_object = utils.setup_linear_regression(
        lambda1=0., lambda2=0.)

    _ = utils.train_linear_regression(
        model_object=plain_linear_model_object,
        training_predictor_table=training_predictor_table,
        training_target_table=training_target_table)

    training_predictions = plain_linear_model_object.predict(
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

    validation_predictions = plain_linear_model_object.predict(
        validation_predictor_table.as_matrix()
    )

    _ = utils.evaluate_regression(
        target_values=validation_target_table[utils.TARGET_NAME].values,
        predicted_target_values=validation_predictions,
        mean_training_target_value=mean_training_target_value,
        dataset_name='validation')


def plot_linear_regression_coeffs(plain_linear_model_object,
                                  training_predictor_table):
    """Plots linear-regression coefficients.

    :param plain_linear_model_object: Trained instance of `sklearn.linear_model`.
    :param training_predictor_table: See doc for `utils.read_feature_file`.
    """

    utils.plot_model_coefficients(
        model_object=plain_linear_model_object,
        predictor_names=list(training_predictor_table)
    )

    pyplot.show()


def ridge_regression_example(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains ridge-regression model.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    ridge_model_object = utils.setup_linear_regression(
        lambda1=0., lambda2=1.)

    _ = utils.train_linear_regression(
        model_object=ridge_model_object,
        training_predictor_table=training_predictor_table,
        training_target_table=training_target_table)

    training_predictions = ridge_model_object.predict(
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

    validation_predictions = ridge_model_object.predict(
        validation_predictor_table.as_matrix()
    )

    _ = utils.evaluate_regression(
        target_values=validation_target_table[utils.TARGET_NAME].values,
        predicted_target_values=validation_predictions,
        mean_training_target_value=mean_training_target_value,
        dataset_name='validation')


def plot_ridge_regression_coeffs(ridge_model_object, training_predictor_table):
    """Plots ridge-regression coefficients.

    :param ridge_model_object: Trained instance of `sklearn.linear_model`.
    :param training_predictor_table: See doc for `utils.read_feature_file`.
    """

    utils.plot_model_coefficients(
        model_object=ridge_model_object,
        predictor_names=list(training_predictor_table)
    )

    pyplot.show()


def lasso_regression_example(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains lasso-regression model.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    lasso_model_object = utils.setup_linear_regression(
        lambda1=1e-6, lambda2=0.)

    _ = utils.train_linear_regression(
        model_object=lasso_model_object,
        training_predictor_table=training_predictor_table,
        training_target_table=training_target_table)

    training_predictions = lasso_model_object.predict(
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

    validation_predictions = lasso_model_object.predict(
        validation_predictor_table.as_matrix()
    )

    _ = utils.evaluate_regression(
        target_values=validation_target_table[utils.TARGET_NAME].values,
        predicted_target_values=validation_predictions,
        mean_training_target_value=mean_training_target_value,
        dataset_name='validation')


def plot_lasso_regression_coeffs(lasso_model_object, training_predictor_table):
    """Plots lasso-regression coefficients.

    :param lasso_model_object: Trained instance of `sklearn.linear_model`.
    :param training_predictor_table: See doc for `utils.read_feature_file`.
    """

    utils.plot_model_coefficients(
        model_object=lasso_model_object,
        predictor_names=list(training_predictor_table)
    )

    pyplot.show()


def elastic_net_example(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains elastic-net model.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    elastic_net_model_object = utils.setup_linear_regression(
        lambda1=1e-6, lambda2=1.)

    _ = utils.train_linear_regression(
        model_object=elastic_net_model_object,
        training_predictor_table=training_predictor_table,
        training_target_table=training_target_table)

    training_predictions = elastic_net_model_object.predict(
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

    validation_predictions = elastic_net_model_object.predict(
        validation_predictor_table.as_matrix()
    )

    _ = utils.evaluate_regression(
        target_values=validation_target_table[utils.TARGET_NAME].values,
        predicted_target_values=validation_predictions,
        mean_training_target_value=mean_training_target_value,
        dataset_name='validation')


def plot_elastic_net_coeffs(elastic_net_model_object, training_predictor_table):
    """Plots elastic-net coefficients.

    :param elastic_net_model_object: Trained instance of `sklearn.linear_model`.
    :param training_predictor_table: See doc for `utils.read_feature_file`.
    """

    utils.plot_model_coefficients(
        model_object=elastic_net_model_object,
        predictor_names=list(training_predictor_table)
    )

    pyplot.show()


def train_for_linear_l1l2_experiment(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains model for linear-model experiment with L1 and L2 regularization.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    lambda1_values = numpy.logspace(-8, -4, num=9)
    lambda2_values = numpy.logspace(-8, -4, num=9)

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
                verbose=False)

            validation_mae_matrix_s01[i, j] = this_evaluation_dict[
                utils.MAE_KEY]
            validation_mse_matrix_s02[i, j] = this_evaluation_dict[
                utils.MSE_KEY]
            validation_mae_skill_matrix[i, j] = this_evaluation_dict[
                utils.MAE_SKILL_SCORE_KEY]
            validation_mse_skill_matrix[i, j] = this_evaluation_dict[
                utils.MSE_SKILL_SCORE_KEY]


def plot_linear_l1l2_experiment(
        lambda1_values, lambda2_values, validation_mae_matrix_s01,
        validation_mse_matrix_s02, validation_mae_skill_matrix,
        validation_mse_skill_matrix):
    """Plots results of linear-model experiment with L1 and L2 regularization.

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
    pyplot.title(r'MAE (mean absolute error) skill score on validation data')

    utils.plot_scores_2d(
        score_matrix=validation_mse_skill_matrix,
        min_colour_value=numpy.percentile(validation_mse_skill_matrix, 1.),
        max_colour_value=numpy.percentile(validation_mse_skill_matrix, 99.),
        x_tick_labels=numpy.log10(lambda2_values),
        y_tick_labels=numpy.log10(lambda1_values)
    )

    pyplot.xlabel(r'log$_{10}$ of ridge coefficient ($\lambda_2$)')
    pyplot.ylabel(r'log$_{10}$ of lasso coefficient ($\lambda_1$)')
    pyplot.title(r'MSE (mean squared error) skill score on validation data')


def finish_linear_l1l2_experiment(
        lambda1_values, lambda2_values, validation_mae_skill_matrix,
        training_predictor_table, training_target_table,
        testing_predictor_table, testing_target_table):
    """Finishes linear-model experiment with L1 and L2 regularization.

    :param lambda1_values: See doc for `plot_linear_l1l2_experiment`.
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


def logistic_regression_example(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains logistic-regression model.

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
    """Plots logistic-regression coefficients.

    :param plain_log_model_object: Trained instance of
        `sklearn.linear_model.SGDClassifier`.
    :param training_predictor_table: See doc for `utils.read_feature_file`.
    """

    utils.plot_model_coefficients(
        model_object=plain_log_model_object,
        predictor_names=list(training_predictor_table)
    )

    pyplot.show()


def elastic_net_log_example(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains logistic regression with elastic-net penalty.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    en_logistic_model_object = utils.setup_logistic_regression(
        lambda1=1e-3, lambda2=1e-3)

    _ = utils.train_logistic_regression(
        model_object=en_logistic_model_object,
        training_predictor_table=training_predictor_table,
        training_target_table=training_target_table)

    validation_probabilities = en_logistic_model_object.predict_proba(
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


def plot_en_log_coeffs(en_logistic_model_object, training_predictor_table):
    """Plots logistic-regression coefficients.

    :param en_logistic_model_object: Trained instance of
        `sklearn.linear_model.SGDClassifier`.
    :param training_predictor_table: See doc for `utils.read_feature_file`.
    """

    utils.plot_model_coefficients(
        model_object=en_logistic_model_object,
        predictor_names=list(training_predictor_table)
    )

    pyplot.show()


def train_tree_example1(
        training_predictor_table, training_target_table,
        validation_predictor_table, validation_target_table):
    """Trains decision tree with default params.

    :param training_predictor_table: See doc for `utils.read_feature_file`.
    :param training_target_table: Same.
    :param validation_predictor_table: Same.
    :param validation_target_table: Same.
    """

    model_object = utils.setup_classification_tree()

    _ = utils.train_classification_tree(
        model_object=model_object,
        training_predictor_table=training_predictor_table,
        training_target_table=training_target_table)

    validation_probabilities = model_object.predict_proba(
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


def plot_decision_tree(model_object):
    """Plots decision tree.

    :param model_object: Trained decision tree (instance of
        `sklearn.tree.DecisionTreeClassifier`).
    """

    io_handle = StringIO()

    sklearn.tree.export_graphviz(
        model_object, out_file=io_handle,
        filled=True, rounded=True, special_characters=True)

    graph_object = pydotplus.graph_from_dot_data(io_handle.getvalue())
    Image(graph_object.create_png())


