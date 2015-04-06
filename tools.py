__author__ = 'jkelleher'
import re
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

from pandas import DataFrame, Series
from itertools import groupby

# TODO: instead of regenerating stat values, look them up in the stat table
# perhaps this can be done by forcing my own label names onto the stats in the stat table
# then changing them around once everything has been put in? this will guarantee that
# i can always access

# also change gen_stat_dataframe to concat a list of series instead of calling concat many times

# strip nan from array
def denan(a):
    return a[~np.isnan(a)]


# strip any pair which contains at least one nan
def double_denan(a, b):
    if not len(a) == len(b):
        raise ValueError('Arrays must be the same length')
    mask1 = ~np.isnan(np.array(a))
    mask2 = ~np.isnan(np.array(b))
    mask = mask1[mask2]
    return a[mask], b[mask]


# get num items in array
def count(a):
    if isinstance(a, type(DataFrame())):
        return a.apply(count)
    return len(denan(a))


# get mean of array
def mean(a):
    if isinstance(a, type(DataFrame())):
        return a.apply(mean)
    return denan(a).mean()


# variance
def var(a):
    if isinstance(a, type(DataFrame())):
        return a.apply(var)
    a1 = denan(a)
    a2 = (a1 - a1.mean()) ** 2
    return a2.sum() / (len(a2) - 1)


# standard dev
def std(a):
    if isinstance(a, type(DataFrame())):
        return a.apply(std)
    return np.sqrt(var(a))


# standard error
def sde(a):
    if isinstance(a, type(DataFrame())):
        return a.apply(sde)
    a = denan(a)
    return std(a) / np.sqrt(count(a))


def corr_with(to_corr_with, kind='pearson', name='', pvalue=False):
    corr_with_storage = [0]
    # if isinstance(to_corr_with, str):
    #     to_corr_with = [to_corr_with]
    def temp(a):
        if isinstance(a, type(DataFrame())):
            corr_with_storage[0] = a[to_corr_with]
            return a.apply(temp)
        cw = corr_with_storage[0]
        x, y = double_denan(a, cw)
        if re.match(kind, 'pearson'):
            r, p = stats.pearsonr(x, y)
        elif re.match(kind, 'spearman'):
            r, p = stats.spearmanr(x, y)
        else:
            raise ValueError('No valid type of correlation coefficient given')
        r = 0 if np.isnan(r) else r
        p = 1 if np.isnan(p) else p
        if pvalue:
            return [r, p]
        else:
            return r
    temp.name = 'corr with'
    if not name == '':
        temp.name += ' ' + name
    if pvalue:
        temp.name += ',p-value with ' + name
    return temp


# mean dev
def md(a):
    if isinstance(a, type(DataFrame())):
        return a.apply(md)
    a = denan(a)
    u = a.mean()
    dists = [abs(x - u) for x in a]
    return sum(dists) / len(dists)


# check if missing values in DataFrame
def has_nan(df, verbose=False):
    column_count = 0
    found = False
    for a in df.T.values:
        if True in np.isnan(a):
            found = True
            if verbose:
                message = []
                found_pos = [i for i, is_true in enumerate(np.isnan(a)) if is_true]
                message.append('NaN found in column #' + str(column_count) + ' : ' + df.columns[column_count])
                for rowcount in found_pos:
                    message.append('\n\trow #' + str(rowcount) + ' : ' + str(df.index[rowcount]))
                print ''.join(message)
            else:
                break
        column_count += 1
    return found


# check if DataFrame has values we expect, given the data (ie no strings)
def has_only_good_values(df, allowed=range(11)):
    vals_1d = df.values.ravel()
    vals_sans_nan = vals_1d[~np.isnan(vals_1d)]
    vals_set = set(vals_sans_nan)
    for a in allowed:
        if a in vals_set:
            vals_set.remove(a)
    if len(vals_set) > 0:
        print vals_set
        return False
    return True

# generate a statistical dataframe
def gen_stats_dataframe(df, funclist):
    stat_dataframe = DataFrame()
    for func in funclist:
        name = func.__name__.replace('_', ' ')
        if name == 'temp':
            name = func.name
        new_series = apply_stat_func(df, func, name)
        stat_dataframe = pd.concat([stat_dataframe, new_series], axis=1)
    return stat_dataframe


# create a series based off of a statistical analysis of each column in a dataframe
def apply_stat_func(df, func, func_name):
    result_series = func(df)
    return DataFrame(list(result_series.values), index=df.columns, columns=func_name.split(','))

# create a histogram in one step for DataFrames, Series, 1-2D arrays, and lists of the above
def simplehist(plotdata, title='', shape=(1, 1), normed=False):
    superlist = []
    if isinstance(plotdata, type([])):
        for member in plotdata:
            superlist.append(np.array(member))
    else:
        superlist.append(np.array(plotdata))
    fig, axes = plt.subplots(shape[0], shape[1])
    axes = np.array(axes).ravel()
    fig.suptitle(title)
    count_members = len(superlist) if len(superlist) < len(axes) else len(axes)
    for i in range(count_members):
        list1d = superlist[i].ravel()
        list_sans_nan = denan(list1d)
        x_all = sorted(list_sans_nan)
        y = [len(list(group)) for key, group in groupby(x_all)]
        if normed:
            y = np.array(y) / float(sum(y))
        x_set = list(set(x_all))
        axes[i].bar(x_set, y, align='center')
        # ax.set_title(title)

def rough_assessment(data):
    means = [mean(row) for row in data.values]
    return Series(means, index=data.index)

def equal_sample_frame(df, indices, n=40):
    if isinstance(indices, str):
        indices = [indices]
    sample_frames = []
    new_ix = []
    for ix in indices:
        frame = df.ix[ix]
        sample_frame = frame.ix[np.random.choice(frame.index.values, n)]
        sample_frames.append(sample_frame)
        new_ix.append([ix]*n)
        # new_ix.append(zip([ix]*n, sample_frame.index.values))
    new_ix = np.concatenate(new_ix)
    new_frame = pd.concat(sample_frames)
    new_frame = new_frame.set_index([new_ix, new_frame.index])
    return new_frame

def average_corr(df, cw, name, indices, n_samps=10, n_trials=100):
    if isinstance(indices, str):
        indices = [indices]
    sample_frame = equal_sample_frame(df, indices, n_samps)
    sample_stats = gen_stats_dataframe(sample_frame, [count, corr_with(cw, name=name, kind='spearman')])
    for i in range(n_trials-1):
        new_sample_frame = equal_sample_frame(df, indices, n_samps)
        new_sample_stats = gen_stats_dataframe(new_sample_frame, [count, corr_with(cw, name=name, kind='spearman')])
        sample_stats = sample_stats.add(new_sample_stats)
    sample_stats /= n_trials
    r = sample_stats['corr with ' + name]
    n = sample_stats['count']
    t = r * np.sqrt((n-2)/(1-r**2))
    p = stats.t.sf(abs(t), df=n) * 2
    return pd.concat([sample_stats, Series(p, index=sample_stats.index, name=('p-value with '+name))], axis=1)

def average_equal_sample_frame(df, funclist, indices, n_samps=10, n_trials=100, pvalues=True):
    #TODO: make count optional
    if count not in funclist:
        funclist = [count] + funclist
    if isinstance(indices, str):
        indices = [indices]
    sample_frame = equal_sample_frame(df, indices, n_samps)
    sample_stats = gen_stats_dataframe(sample_frame, funclist)
    for i in range(n_trials-1):
        new_sample_frame = equal_sample_frame(df, indices, n_samps)
        new_sample_stats = gen_stats_dataframe(new_sample_frame, funclist)
        sample_stats = sample_stats.add(new_sample_stats)
    cols = sample_stats.columns.values
    pval_cols = []
    for col in cols:
            if re.match('p-value', col):
                pval_cols.append(col)
    if pval_cols:
        sample_stats = sample_stats.drop(pval_cols, axis=1)
    sample_stats /= n_trials
    if pvalues:
        cols = sample_stats.columns.values
        n = sample_stats['count']
        for i in range(len(cols)-1, -1, -1):
            if re.match('corr with', cols[i]):
                r = sample_stats[cols[i]]
                t = r * np.sqrt((n-2)/(1-r**2))
                p = stats.t.sf(abs(t), df=n) * 2
                name = 'p-value with '+cols[i][10:]
                s = Series(p, sample_stats.index)
                sample_stats.insert(i+1, name, s)
    return sample_stats

def plot_variance_analysis(indices, stat_frames, legend_labels, shape):
    x = np.linspace(1, 5, 500)
    fig, axes = plt.subplots(shape[0], shape[1], sharex=True, sharey=True)
    questions_and_axes = zip(indices, axes.ravel())
    frames_and_labels = zip(stat_frames, legend_labels)
    for qa in questions_and_axes:
        q = qa[0]
        ax = qa[1]
        for fl in frames_and_labels:
            frame = fl[0]
            label = fl[1]
            ax.plot(x, stats.norm.pdf(x, frame['mean'][q], frame['std'][q]), label=label)
            ax.set_xlabel(q)
            ax.legend(loc='best')
    plt.xticks([1,2,3,4,5])
    return fig, axes

def z_frame(df, sumstats=None):
    if sumstats is None:
        sumstats = gen_stats_dataframe(df, [mean, std])
    df_z = df.apply(lambda x: (x - sumstats['mean']) / sumstats['std'], axis=1)
    return df_z

def map_data(from_frame, to_frame=None, to_stats=None):
    sumstats_from = gen_stats_dataframe(from_frame, [mean, std])
    if to_frame is not None:
        sumstats_to = gen_stats_dataframe(to_frame, [mean, std])
    else:
        sumstats_to = to_stats[['mean', 'std']]
    from_z = from_frame.apply(lambda x: (x - sumstats_from['mean']) / sumstats_from['std'], axis=1)
    mapped = from_z.apply(lambda x: x * sumstats_to['std'] + sumstats_to['mean'], axis=1)
    return mapped

def table_from_col_vals(df, tcol, col_map=None, incl_sum=False, reduce_frame=False, na_fill=None):
    r = df.set_index([tcol, range(len(df))]).unstack(level=0).apply(lambda x: x.value_counts())
    if reduce_frame:
        r = r.stack(level=0).groupby(level=0).sum()
    elif incl_sum and isinstance(r.columns, pd.MultiIndex):
        r_sum = r.stack(level=0).groupby(level=0).sum()
        frames = [r_sum] + [r[col] for col in r.columns.levels[0]]
        keys=['TOTAL'] + list(r.columns.levels[0])
        r = pd.concat(frames, keys=keys, axis=1)
    if col_map:
        if isinstance(r.columns, pd.MultiIndex):
            col_names = [col_map[c] for c in r.columns.levels[-1]]
            level_list = list(r.columns.levels[:-1])
            level_list.append(col_names)
            r.columns.set_levels(level_list, inplace=True)
        else:
            col_names = [col_map[c] for c in r.columns]
            r.columns = col_names
    if na_fill is not None:
        r.fillna(na_fill, inplace=True)
    return r
