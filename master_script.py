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

# explore first dataset rows
plant_growth.head()

# count group members
plant_growth['group'].value_counts()

# get average weight
np.mean(plant_growth['weight'])

# get summary statistics
plant_growth.groupby(['group']).describe()
plant_growth.groupby(['group']).mean()

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

# ANOVA
# compute anova
aov_table = sm.stats.anova_lm(results, typ=2)

# explore anova results
aov_table

# TODO add t-test

#############
# Functions #
#############

# generic name of functions
# function(args)

# example function
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

myNames = ["healthy", "tissue", "quantity"]
myNames

#########
# NUMPY #
#########

# sequential numbers
foo1 = np.arange(1, 100, 7)
foo1

# use objects in functions
foo2 = np.arange(1, 100, 5)
foo2

np.log(foo1)
np.log10(foo1)
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

heights + 2

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


foo_df = pd.DataFrame([foo2, foo3, foo4]).T
foo_df.columns = ['numeric_col', 'character_col', 'bool_col']
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
foo_df['numeric_col'] == 7

foo_df[foo_df['numeric_col'] > 20]

foo_df[(foo_df['character_col'] == "Heart") | (foo_df['character_col'] == "Liver")]

foo_df['character_col'][0]

# ## Initial views

protein_data.head()
protein_data.tail()

# ## Summary statistics

protein_data.describe()

# ## Subsetting

# select columns
protein_data['Sequence.Length']

type(protein_data['Sequence.Length'])


# select several columns
protein_data[['Peptides', 'MW.kDa']]

# command chaining
protein_data[['Peptides', 'MW.kDa']].sample(10).head(3)

# frequency table
protein_data['Peptides'].value_counts()


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
