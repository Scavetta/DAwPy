######################################
# DATA ANALYSIS WITH PYTHON WORKSHOP #
# Rick Scavetta and Boyan Angelov    #
######################################

import math
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy import stats
from scipy.special import factorial

######################################### DAY 1 #########################################

###########################
# Plant growth case study #
###########################

# load dataset
plant_growth = pd.read_csv('data/plant_growth.csv')

# examine the data summaries
plant_growth.describe()
plant_growth['group'].value_counts()

# explore first dataset rows
plant_growth.head()

# count group members
plant_growth['group'].value_counts()

# get average weight
np.mean(plant_growth['weight'])

# get summary statistics
# NOTE quite complicated to get the exact same output in R, have to create a custom function because of the index
plant_growth.groupby(['group']).describe()
df1 = plant_growth.groupby(['group']).mean()
df2 = plant_growth.groupby(['group']).std()
final_df = pd.concat([df1, df2], axis=1)

# visualize weight per group
sns.stripplot(x='group', y='weight', data=plant_growth, jitter=0.2)
sns.boxplot(x='group', y='weight', data=plant_growth)

# fit a linear model
# specify model
model = ols("weight ~ group", plant_growth)

# fit model
results = model.fit()

# explore model results
results.summary()

# extract coefficients
results.params.Intercept
results.params["group[T.trt1]"]
results.params["group[T.trt2]"]

# ANOVA
# compute anova
aov_table = sm.stats.anova_lm(results, typ=2)

# explore anova results
aov_table

# t-test
from statsmodels.stats.multicomp import pairwise_tukeyhsd
print(pairwise_tukeyhsd(plant_growth['weight'], plant_growth['group']))


#############
# Functions #
#############

# generic name of functions
# function(args)

# An example function
np.log2(8)
# an alternative example function
math.log(8, 2)

# task: build a function that sums a list
heights = [167, 188, 178, 194, 171, 169]

# sum function
def count_tall_people(heights_list):
    tall_people_number = 0
    for height in heights:
        if height >= 170:
            tall_people_number += 1

    return tall_people_number

# use function
count_tall_people(heights)

# several bult-in functions
sum(heights)
len(heights)
name = "Berlin"
name.lower()
name.upper()
len(name)

###########
# Objects #
###########

# types

a = 1
type(a)

b = 0.25
type(b)

c = "Berlin"
type(c)

d = True
type(d)

# coercion
str(a)
type(str(a))

# lists
xx = [3, 8, 9, 23]
print(xx)

# but what yo ureally want is an np array
xx = np.array([3, 8, 9, 23])
xx

m = 1.12
b = -0.4

myNames = ["healthy", "tissue", "quantity"]
myNames

# Exercise:
m * np.array(xx) + b


#########
# NUMPY #
#########

n = 34
p = 6

# sequential numbers
foo1 = np.arange(1, 100, 7)
foo1

# use objects in functions
foo2 = np.arange(1, n, p)
foo2

np.log(foo1)
np.log10(foo1)
np.log2(8)


np.sqrt(foo1)
stats.zscore(foo1)
foo1 * 3

sum(foo1)
factorial(6)
len(foo1)

foo1 + 100
foo1 * 3
foo2 + foo2

heights = [167, 188, 178, 194, 171, 169]
type(heights)
heights[0]
heights[0:2]
heights[-1]

heights_persons = {"Mary": 176, "John" : 169, "Jeremy": 194, "Elena": 170}
type(heights_persons)

heights_persons.values()
heights_persons.keys()
type(heights)
# heights + 2 # so this doesn't work, only numpy works that wayself.


heights_np = np.array(heights)
heights_np

heights_np + 2

##########
# PANDAS #
##########

foo2

foo3 = np.array(["Liver", "Brain", "Testes", "Muscle", "Intestine", "Heart"])
foo4 = np.array([True, False, False, True, True, False])

# TODO add numeric column
type(foo4)


foo_df = pd.DataFrame([foo4, foo3, foo2]).T
foo_df.columns = ['bool_col', 'character_col', 'numeric_col']
foo_df

foo_df.describe()

foo_df.shape

foo_df.head()

foo_df.tail()

foo_df.sample()


protein_data = pd.read_csv('data/Protein.txt', delimiter='\t')
protein_data.head()

######################################### DAY 2 #########################################

# protein df exercises
# pandas rearrange data

# Construct your own dataframe
foo_df['numeric_col'] > 20

foo_df[foo_df['numeric_col'] > 20]

foo_df[(foo_df['character_col'] == "Heart") | (foo_df['character_col'] == "Liver")]

# %>% within operator in R:
# foo_df[chr_col %in% c("H", "L")]

foo_df[(foo_df['character_col'].isin(["Heart", "Liver"]))]

mySearch = np.array(["Heart", "Liver"])
foo_df[(foo_df['character_col'].isin(mySearch))]

foo_df['character_col'][0:4]

# ## Initial views

protein_data.head()
protein_data.tail()

# ## Summary statistics
protein_data.columns

protein_data.describe()

# ## Subsetting

# select columns
protein_data['Sequence.Length']

type(protein_data['Sequence.Length'])
# [] returns a series (kind of like a vector, a nympy array under the hood)
# [[]] returns a data frame. Just stick to oneself.
# use dtype for the data dtypes


# select several columns
protein_data[['Peptides', 'MW.kDa']]

# command chaining
protein_data[['Peptides', 'MW.kDa']].sample(10).head(3)

# frequency table
protein_data['Peptides'].value_counts()

######################################################
######################################################
############### protein exercises in depth ###########

# Examine the data See above
# print the first six rows
protein_data.head(6)
# structure
protein_data.describe()
protein_data.dtypes

# Transformations
# log10 of intensities
# First, just print to screen:
np.log10(protein_data[['Intensity.H', 'Intensity.M', 'Intensity.L']])

# Save as new columns:
protein_data[['Intensity.H.log10', 'Intensity.M.log10', 'Intensity.L.log10']] = np.log10(protein_data[['Intensity.H', 'Intensity.M', 'Intensity.L']])


# Save in situ, i.e. replace the columns
protein_data[['Intensity.H', 'Intensity.M', 'Intensity.L']] = np.log10(protein_data[['Intensity.H', 'Intensity.M', 'Intensity.L']])

# Add intensities:
protein_data['Intensity.H'] + protein_data['Intensity.M']


# log2 of ratios:
# xx = np.log2(protein_data[["Ratio.H.M"]])
# type(xx)
# xx.apply(zscore)
#
#
#
from scipy.stats import zscore

xx = protein_data[['Ratio.H.M']].apply(np.log2)

(xx - xx.mean())/xx.std()

yy = np.array([1, 10, 23, 99])
(yy - np.mean(yy))/np.std(yy, ddof = 1)

def calcZscore(input_list):
    output_list = (input_list - np.mean(input_list))/np.std(input_list, ddof = 1)
    return output_list

calcZscore(yy)

# Exercises 1 - 4, p58-59:
# First off, how many contaminants are there?
protein_data['Contaminant'].value_counts()

# Ex1: Remove contaminants
# only not NA, i.e. contaminants:
protein_data[~protein_data['Contaminant'].isna()]

# only NA, i.e. real numbers:
protein_data[protein_data['Contaminant'].isna()]

# now remove:
# protein_data['Contaminant' != "+"]
protein_data[protein_data['Contaminant'] != "+"]
protein_data['Contaminant'].dropna()


# Ex2: Extract specific uniprot IDs:
# "GOGA7", "PSA6", "S10AB" found in the Uniprot column
# all have _MOUSE appendend to them
mySearch = np.array(["GOGA7_MOUSE", "PSA6_MOUSE", "S10AB_MOUSE"])
protein_data[(protein_data['Uniprot'].isin(mySearch))]

# Ex3: Get low HM ratio p-value proteins
# e.g. Ratio.H.M.Sig < 0.05
# 104 proteins

# Ex4: Get extreme log2 ratio proteins for HM
# e.g. Ratio.H.M > 2.0 | Ratio.H.M < -2.0

# for completness, plot the data:
# log10-Intensity vs log2-Ratio
# sns.regplot(x='Intensity.H', y='Ratio.H.M', data=protein_data)
sns.regplot(x = 'Ratio.H.M', y='Intensity.H', data=protein_data, scatter_kws={'alpha':0.15}, fit_reg=False)

protein_data.plot.scatter(x = 'Ratio.H.M',
                          y = 'Intensity.H')

protein_data.plot.scatter(x = 'Ratio.H.M',
                          y = 'Intensity.H',
                          c = 'Ratio.H.M.Sig')

protein_data.plot.scatter(x = 'Ratio.H.M',
                          y = 'Intensity.H',
                          c = 'Ratio.H.M.Sig',
                          colormap = 'viridis_r',
                          alpha = 0.35,
                          title = "hello",
                          sharex = False)


myPlot = protein_data.plot.scatter(x = 'Ratio.H.M',
                          y = 'Intensity.H',
                          c = 'Ratio.H.M.Sig',
                          colormap = 'viridis_r',
                          alpha = 0.35,
                          title = "hello",
                          sharex = False)

myPlot_fig = myPlot.get_figure()

myPlot_fig.savefig("test.png", dpi = 300)

# Exercises 4


# top 20 highest HM and ML ratios
# use arrange() in the dplyr package
topHM = protein_data.sort_values("Ratio.H.M", ascending = False).head(20)[["Uniprot"]]
topML = protein_data.sort_values("Ratio.M.L", ascending = False).head(20)[["Uniprot"]]

# Exercise 5:
# What is the intersection between these lists? see page 65

np.intersect1d(topHM, topML)
np.intersect1d(topHM["Uniprot"], topML["Uniprot"])
np.setdiff1d(topHM["Uniprot"], topML["Uniprot"])
np.setdiff1d(topML["Uniprot"], topHM["Uniprot"])
np.union1d(topML["Uniprot"], topHM["Uniprot"])

######################################################
######################################################

##################
# MISC MATERIALS #
##################

# ## General Operations

# arithmetic operators
34 + 6

# order of operations (careful if python 2)
2 - 3 / 4

(2 - 3) / 4

n = 34
p = 6
n + p

# exercise 1
1.12 * 3 - 0.4

m = 1.12
b = -0.4

m * 3 + b

m * 8 + b

m * 9 + b

m * 23 + b

# ## Logical expressions

1 == 1

1 < 2

1 > 2

1 != 1

# # Control Flow and Loops

# ## Comparison operators
if (1 + 1 == 2):
    print("This is true")
elif (10 * 2 != 20):
    print("This is true")

heights = [167, 188, 178, 194, 171, 169]

for height in heights:
    print(height)

tall_or_short = []
for height in heights:
    if height >= 170:
        print("Tall")
    else:
        print("Short")

heights_squared = []
for height in heights:
    heights_squared.append(height ** 2)
heights_squared
