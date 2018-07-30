# coding: utf-8

# # Basic Operations

import math
import pandas as pd
import numpy as np
import seaborn as sns


# Load data

plant_growth = pd.read_csv('data/plant_growth.csv')
plant_growth.head()


plant_growth['group'].value_counts()


np.mean(plant_growth['weight'])


plant_growth.groupby(['group']).describe()


plant_growth.groupby(['group']).mean()


sns.stripplot(x='group', y='weight', data=plant_growth, jitter=0.2)


sns.boxplot(x='group', y='weight', data=plant_growth)


# ## Linear model

from statsmodels.formula.api import ols


model = ols("weight ~ group", plant_growth)
results = model.fit()
results.summary()


# ## ANOVA

import statsmodels.api as sm
aov_table = sm.stats.anova_lm(results, typ=2)
aov_table


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


# generic name of functions
# function(args)

math.log(8, 2)


# create a list
xx = [3, 8, 9, 23]
xx


myNames = ["healthy", "tissue", "quantity"]
myNames


# sequential numbers
foo1 = np.arange(1, 100, 7)
foo1


# use objects in functions
foo2 = np.arange(1, n, p)
foo2


# Type of math functions:
# 1 - transformation: Every value is transformed in the SAME way
# +, -, *,
# Normalisation, Z-scores
# log, log10, log2, sqrt
# Multiply every element by 3
from scipy import stats
np.log(foo1)
np.log10(foo1)
np.sqrt(foo1)
stats.zscore(foo1)
foo1 * 3


sum(foo1)


from scipy.special import factorial
factorial(6)
len(foo1)


foo1 + 100


foo1 * 3


foo2 + foo2


# ## Types

type(1)


type(0.25)


type("Berlin")


type(True)


# coercion
str(1)


type(str(1))


# ## Dataframes
# Note: use a separate notebook for this

# TODO numeric column fix


foo2


foo3 = np.array(["Liver", "Brain", "Testes", "Muscle", "Intestine", "Heart"])
foo4 = np.array([True, False, False, True, True, False])
type(foo4)


foo_df = pd.DataFrame([foo2, foo3, foo4]).T
foo_df.columns = ['numeric_col', 'character_col', 'bool_col']
foo_df


foo_df.describe()


foo_df.shape


foo_df.head()


foo_df.tail()


foo_df.sample()


# ## Logical expressions

1 == 1


1 < 2


1 > 2


1 != 1


foo_df['numeric_col'] == 7


foo_df[foo_df['numeric_col'] > 20]


foo_df[(foo_df['character_col'] == "Heart") | (foo_df['character_col'] == "Liver")]


foo_df['character_col'][0]


# coding: utf-8

# # Basic Operations



1 + 1




3 / 4




7 * 10




14 - 2 * 2




(88  * 23 - 2) / (45 + 7)


# coding: utf-8

# # Data Structures

# ## Variables



x = 1
y = 1.6
z = "Berlin"
j = True




x




type(x)




type(y)




type(z)




type(j)




heights = [167, 188, 178, 194, 171, 169]




type(heights)




heights[0]




heights[0:2]




heights[-1]




heights_persons = {"Mary": 176, "John" : 169, "Jeremy": 194, "Elena": 170}




type(heights_persons)




heights_persons.values()




heights_persons.keys()


# ## Numpy



heights + 2




import numpy as np




heights_np = np.array(heights)
heights_np




heights_np + 2


# coding: utf-8

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


# coding: utf-8

# # Functions



heights = [167, 188, 178, 194, 171, 169]




# function that returns number of tall people from a list
def count_tall_people(heights_list):
    tall_people_number = 0
    for height in heights:
        if height >= 170:
            tall_people_number += 1

    return tall_people_number




count_tall_people(heights)




# bult in functions
sum(heights)




len(heights)




name = "Berlin"
name.lower()




name.upper()




len(name)


# coding: utf-8

# # Dataframe Basics and Visualisation



import pandas as pd


# ## Get data



mtcars = pd.read_csv("data/mtcars.csv")


# ## Initial views



mtcars.head()




mtcars.tail()


# ## Summary statistics



mtcars.describe()


# ## Subsetting



# select columns
mtcars['mpg']
# or
mtcars.mpg




type(mtcars['mpg'])




# select several columns
mtcars[['mpg', 'cyl']]




# command chaining
mtcars[['mpg', 'cyl']].sample(10).head(3)




mtcars['gear'].value_counts()


# ## Visualize
