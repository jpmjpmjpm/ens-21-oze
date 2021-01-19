"""
As described in the challenge webpage, the metric considered for this problem combines the Prediction Interval Coverage Probability (PICP)
with the MPIW metric.

The metric can also be found here https://arxiv.org/pdf/1802.07167.pdf in equation (15) with lambda = 1 and alpha = 0.05
"""

import numpy as np

def prediction_interval_coverage_probability(dataframe_y_true, dataframe_y_pred):
    """
        NOTA: the order (dataframe_y_true, dataframe_y_pred) matters if the metric is
        non symmetric.

    Args
        dataframe_y_true: Pandas Dataframe
            Dataframe containing the true values of y.
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_y_true = pd.read_csv(CSV_1_FILE_PATH, index_col=0, sep=',')

        dataframe_y_pred: Pandas Dataframe
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_y_pred = pd.read_csv(CSV_2_FILE_PATH, index_col=0, sep=',')

    Returns
        score: Float
            The metric evaluated with the two dataframes. This must not be NaN.
    """
    # number of data to predict in each sample
    dfc = 2  # drop first columns
    n_data_to_pred = (dataframe_y_true.shape[1] - dfc) // 2
    n_row = dataframe_y_true.shape[0]
    alpha = 0.05

    # test to ensure that result is the same if arguments are switched
    infs = dataframe_y_true.iloc[:, dfc:n_data_to_pred+dfc]
    sups = dataframe_y_true.iloc[:, n_data_to_pred+dfc:]
    size_intervals = np.mean(
        np.abs(sups.values - infs.values)
    )
    reverse = size_intervals > 0

    score = 0
    for row in range(n_row):
        # get the time series (without the id and the initial date of the
        # sample, only the numerical values)
        if reverse:
            y_pred = dataframe_y_true.iloc[row].values[dfc::]
            y_true = dataframe_y_pred.iloc[row].values[dfc::]
        else:
            y_pred = dataframe_y_pred.iloc[row].values[dfc::]
            y_true = dataframe_y_true.iloc[row].values[dfc::]
        nb_pred_in_the_interval = 0
        length_interval = 0
        for idx in range(n_data_to_pred):
            # the first n_data_to_pred predictions are the lower bounds
            if y_true[idx] >= y_pred[idx]:
                # the last n_data_to_pred predictions are the upper bounds
                if y_true[idx] <= y_pred[idx+n_data_to_pred]:
                    nb_pred_in_the_interval = nb_pred_in_the_interval + 1
                    l_interval = y_pred[idx + n_data_to_pred] - y_pred[idx]
                    length_interval = length_interval + l_interval
        picp = nb_pred_in_the_interval / n_data_to_pred
        mpiw = length_interval / n_data_to_pred

        penalization = np.maximum(0, (1 - alpha) - picp) ** 2
        weight = n_data_to_pred / (alpha * (1-alpha))
        score = score + mpiw + weight*penalization

    return score / n_row


if __name__ == '__main__':
    import pandas as pd

    CSV_FILE_Y_TRUE = 'y_train_full_BhBAcvH.csv'
    df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',')

    # make fake prediction: y_inf = y+w-1, y_max = y+w+1 with w random
    dfc = 2  # drop first columns
    n = (df_y_true.shape[1] - dfc) // 2
    w = np.random.randn(df_y_true.shape[0], n)
    df_y_pred = df_y_true.copy()
    df_y_pred.iloc[:, dfc:n+dfc] = df_y_true.iloc[:, dfc:n+dfc] + w - 1
    df_y_pred.iloc[:, n+dfc:] = df_y_true.iloc[:, n+dfc:] + w + 1

    print(custom_metric_function(df_y_true, df_y_pred), '=', custom_metric_function(df_y_pred, df_y_true))
    print(custom_metric_function(df_y_true, df_y_true))
