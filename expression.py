import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)

expression_data = pd.read_csv('data/Expression.txt', delimiter='\t')
expression_data.head()

expression_data_melted = expression_data.melt()

# split
expression_data_melted['treatment'] = expression_data_melted['variable'].str.split('_').str[0]
expression_data_melted['gene'] = expression_data_melted['variable'].str.split('_').str[1]
expression_data_melted['time'] = expression_data_melted['variable'].str.split('_').str[2]

# compute
# TODO first group by gene, treatment, time



expression_data_melted['avg'] = expression_data_melted['value'].apply(np.mean)

expression_data_melted.groupby(['gene', 'treatment', 'time']).mean()
