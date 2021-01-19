import matplotlib.pyplot as plt
import pandas as pd


class OzeData:
    df = None  # DF representing the data

    def __init__(self, fn):
        """
        Load one of the data sets
        :param fn: file name of the data set to be loaded
        """
        self.df = pd.read_csv(fn)

    def get_ts(self, week_id, stream):
        """
        Build a time series based on a week id and a stream (represented as a stream)
        :param week_id: week id
        :param stream: name of the input data stream to represent
        :return: pandas time series
        """
        dfw = self.df.filter(like=stream, axis=1)
        dti = pd.date_range(self.df.at[week_id, 'FirstDayOfWeek'], periods=168, freq="H")
        return pd.Series(dfw.iloc[week_id].values, index=dti)


if __name__ == '__main__':
    oze_xt = OzeData('x_train_QwN5vrl.csv')
    ts = oze_xt.get_ts(10, 'AHU_3_AIRFLOWTEMP')

    plt.figure()
    ts.plot()
    plt.show()
