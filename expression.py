import pandas as pd
import numpy as np
from dfply import *
from scipy import stats
from statsmodels.formula.api import ols
import seaborn as sns
pd.set_option('display.max_columns', 500)

expression_data = pd.read_csv('data/Expression.txt', delimiter='\t')
expression_data.head()

df = expression_data
df_t = df >> gather('key', 'value')

# NOTE for the CI check this:
# https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

df_summary = (df_t  >>
        separate('key', ['treatment', 'gene', 'time'], remove=True) >>
        mask(~df_t.value.isnull()) >>
        group_by('gene', 'treatment', 'time') >>
        summarize_each([np.mean, np.std, len, stats.sem, mean_confidence_interval], 'value'))
df_summary

# TODO for one gene
# (df_t  >>
#     separate('key', ['treatment', 'gene', 'time'], remove=True) >>
#     mask(X.gene == 'RIPK2') >>
#     group_by(X.gene) >>
#     ols("value ~ treatment", X))

# TODO for all genes

# make a plot
df_temp = df_t  >> separate('key', ['treatment', 'gene', 'time'], remove=True)
df_temp = df_temp.dropna()
g = sns.FacetGrid(df_temp, col="treatment",  row="gene")
g = g.map(sns.stripplot, "time", "value")
