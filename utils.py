import pandas
import numpy as np

def get_data(X_start, X_end, y_year, datafile):
    data = pandas.read_csv(datafile)
    X_mask = ((data['YEAR'] == X_start) & (data['WEEK'] >= 40)) | ((data['YEAR'] == X_end) & (data["WEEK"] <= 39)) | \
             ((data['YEAR'] > X_start) & (data["YEAR"] < X_end))
    X = data.loc[X_mask]["TOTAL CASES"]
    y_mask = ((data['YEAR'] == y_year) & (data['WEEK'] >= 40)) | ((data['YEAR'] == (y_year + 1)) & (data['WEEK'] <= 39))
    y = data.loc[y_mask]
    y = y.groupby(['WEEK']).sum()["TOTAL CASES"]
    return np.array(X), np.array(y)