import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def date_plot(x, y, xmin=None, xmax=None, figsize=(15, 7), **kwargs):
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(figsize=figsize)
        # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        # ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        plt.plot_date(x, y, ls='-', **kwargs)
        plt.xlim(xmin, xmax)
        plt.title(f'Plot for {self.value}')


def bar_plot(data, ptype=plt.barh, figsize=(15, 7), r=90, title=None, **kwargs):
    """Accepts data as a dict()."""
    X = list(data.keys())
    y = list(data.values())
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(figsize=figsize)
        ptype(X, y,**kwargs)
        plt.xticks(rotation=r) # adjust these accordingly
        plt.yticks(fontsize=6) # adjust these accordingly
        plt.title(f'Plot for {title}')
