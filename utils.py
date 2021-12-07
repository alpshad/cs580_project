import pandas
import numpy as np

def get_data(start, end, datafile, window_size=2, step_size=1):
    '''
    start: year to start
    end: year to end 
    datafile: csv file with the data to use
    window_size: # years, or time period, for X features
    step_size: # years to shift 

    returns X, the features, as a 2-d array, with one array per time period
    of length equal to # locations * 52 (# weeks in a year) * window_size
    returns y, the labels, as a 2-d array, with one array per time period
    of length equal to # locations * 52, for the year after the last X year
    returns mean and standard deviation for the entire period from start to end 
    (used for training)

    ** note that "years" start at week 40 and go to the next year week 39 **
    e.g., if start = 2010 and end = 2018
    returns X [[2010-2015], [2011-2016], [2012-2017], [2013-2018]]
    returns y [[2016],      [2017],      [2018],      [2019]]

    where 2010 = week 40 of year 2010 through week 39 of year 2011
    '''
    data = pandas.read_csv(datafile)
    X_list = []
    y_list = []
    for year in range(start, end+1, step_size):
        X_start = year
        X_end = year + window_size
        X_mask = ((data['YEAR'] == year) & (data['WEEK'] >= 40)) | ((data['YEAR'] == X_end) & (data["WEEK"] <= 39)) | \
                ((data['YEAR'] > X_start) & (data["YEAR"] < X_end))
        X = data.loc[X_mask]["TOTAL CASES"]
        y_mask = ((data['YEAR'] == X_end) & (data['WEEK'] >= 40)) | ((data['YEAR'] == (X_end + 1)) & (data['WEEK'] <= 39))
        y = data.loc[y_mask]
        y = y.groupby(['WEEK'], sort=False).sum()["TOTAL CASES"]
        X_list.append(np.array(X, dtype=np.float32))
        y_list.append(np.array(y, dtype=np.float32))
    X_list = np.array(X_list, dtype=np.float32)
    y_list = np.array(y_list, dtype=np.float32)

    X_mean = X_list.mean()
    X_std = X_list.std()
    y_mean = y_list.mean()
    y_std = y_list.std()
    
    loc_list = list(data['REGION'].unique())
    return X_list, y_list, X_mean, X_std, y_mean, y_std, loc_list

def get_edges(X, loc_list):
    # for now, add an edge between all nodes
    rows = []
    cols = []
    for i in range(len(loc_list)):
        for j in range(len(loc_list)):
            rows.append(i)
            cols.append(j)
    # rows, cols makes up the edge list where node ID is index into loc_list
    return (rows, cols)
