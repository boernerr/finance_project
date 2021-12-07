from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

from finance_project.src import plottingUtil
from finance_project.src import DATA_PATH, MODULE_PATH
# DATA_PATH = path.join(MODULE_PATH,'data')

df = pd.DataFrame(pd.read_csv(path.join(MODULE_PATH,'model','entire_data_all_label_2021_09_17.csv')))
# dtreeNum = pd.DataFrame(pd.read_csv(path.join(DATA_PATH,'raw','dtree_num.txt'), sep='\t'))
df['year'] = pd.to_datetime(df.master_dt).dt.year

all_age_all_time = df[(df.TIME_ON_JOB.notna()) & (df.num.notna())].copy()
bad_ages = all_age_all_time[(all_age_all_time.num < all_age_all_time.TIME_ON_JOB) | (all_age_all_time.num -all_age_all_time.TIME_ON_JOB < 17)]
only_age = df[df.num.notna()].copy()

# all_age_all_time[(all_age_all_time.num >17) & (all_age_all_time.num < 80)].num.plot.hist(bins=15, ec='black')
# all_age_all_time[(all_age_all_time.num >17) & (all_age_all_time.num < 80)].TIME_ON_JOB.plot.hist(bins=15, ec='black')
# high_earners = df[(df.num <25) & (df.salaryCpiAdj>150000)].copy()
eng = all_age_all_time[all_age_all_time.category=='engineer'.upper()]
snrMn = all_age_all_time[all_age_all_time.category=='senior_management'.upper()]
snrMn.TIME_ON_JOB.plot.hist(bins=20,ec='b')
eng.TIME_ON_JOB.plot.hist(bins=20,ec='b')

def linear_reg_data(df):
    X = df.TIME_ON_JOB.to_numpy()
    y = df.salaryCpiAdj.to_numpy()

    linr = LinearRegression()
    linr.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    cf = linr.coef_
    intc = linr.intercept_
    xlin = np.linspace(0, 80, 20)
    predicted = linr.predict(X.reshape(-1, 1))
    yln = xlin * cf + intc
    mse = mean_squared_error(X, predicted)
    r2 = r2_score(X, predicted)

    plt.scatter(X, y)
    plt.plot(xlin, yln.flatten(), ls='-', c='r')
    plt.title(f'MSE: {mse} R-sqrd: {r2}')


linear_reg_data(snrMn)



df.year.value_counts().keys()
xYear, yYear = df.groupby('year').year.count().keys(), df.groupby('year').year.count().values
date_plot(xYear, yYear)
plt.plot(xYear, yYear)