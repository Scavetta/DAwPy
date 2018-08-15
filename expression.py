import pandas as pd
import numpy as np
from dfply import *
from scipy import stats
import statsmodels.stats.api as sms

sms.DescrStatsW(a).tconfint_mean()

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
