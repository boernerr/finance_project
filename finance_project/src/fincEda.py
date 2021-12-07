import numpy as np
import pandas as pd
import os
from os import path
import matplotlib.pyplot as plt
from importlib import reload
# reload(preprocess_util)

from finance_project.src import DATA_PATH,MODULE_PATH
from finance_project.src import plottingUtil
from finance_project.src import preprocess_util


df = pd.read_csv(path.join(DATA_PATH,'raw','2020_creditCardExpense.CSV'), parse_dates=[1])
df['month'] = df.Date.dt.month
df['day'] = df.Date.dt.day
df['weekday'] = df.Date.dt.day_name()

df.day.value_counts().to_dict()
df.Date.min()

plottingUtil.bar_plot(df.day.value_counts().to_dict(), ptype=plt.bar, title='Expenditures by day')

plottingUtil.bar_plot(df.Description.value_counts().to_dict(), ptype=plt.barh, title='Vendor total counts')

vendor_dict = df.Description.value_counts().to_dict()
top10vendor = preprocess_util.topNdict(vendor_dict, 10)

plottingUtil.bar_plot(top10vendor, ptype=plt.barh, title='Top 10 Vendor total counts')

df.columns
df.Debit.plot.hist()
less_than100 = df[df.Debit<=100]
less_than100.Debit.plot.hist()

less_than500.Debit.plot.hist()
grtr_100 = df[df.Debit>=100]

class ManuallyClf():

    '''This class will read in a pd.DataFrame and allow user input to be used to manually classify rows.'''

    CATEGORIES = ['Automotive', 'food_supplies', 'food_takeout', 'gas', 'entertainment', 'tools', 'ebay/online', 'misc', 'parking']

    def __init__(self, df, txt_col=None):
        self.df = df.copy()
        self.text_col = txt_col
        self.xprime = None

    def sample_generator(self, frac, seed=None):
        '''Take a sample from self.df of size frac that is representative of the distribution of purchases in self.df by
        dollar amount.'''

        self.xprime = self.df.sample(frac=frac)
        return self.xprime

    def provide_usr_input(self):
        '''Take the sample generated from self.sample_generator() and provide manual classification.'''

        # maybe turn this into a Try/Except block instead of if/else
        if self.xprime is not None:
            # int_ = np.random.randint(self.xprime.index.min(), self.xprime.index.max())
            classification_dict = dict()
            int_ = self.xprime.index.min()
            for int_ in [int_]:#self.xprime.index
                # print('Input manual classification for the following:')
                print(f'Idx: {int_},  Desc:({self.xprime.Description.loc[int_]}), $: {self.xprime.Debit.loc[int_]}')

                class_ = input('Input manual classification for the following:')
                classification_dict[int_] = class_
                # self.xprime.loc[int_]['manual_class'] = class_

            return classification_dict

        else:
            print('Run self.sample_generator() to create a sample for classification!')

    def integrate_manual_classifications(self, dictLike):
        self.xprime['manual_class'] = self.xprime.index.map(dictLike)



manualClf = ManuallyClf(less_than100)

manualClf.sample_generator(frac=.1)

dictLike = manualClf.provide_usr_input()
manualClf.integrate_manual_classifications(dictLike)
xprime = manualClf.xprime







