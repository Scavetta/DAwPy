# To create a virtual environment, only do once
# python3 -m venv .venv

# Activate the environment, each time
# source .venv/bin/activate

# To install a package, use this in the terminal, only once
# pip install matplotlib

# To import an installed package, each time
import matplotlib as plt

# Plant Growth case study

# Import the necessary packages

# import an entire library
import math # Functions beyond the basic maths

# Import an entire library and give it an alias
import pandas as pd # For DataFrame and handling
import numpy as np # Array and numerical processing
import matplotlib.pyplot as plt # Low level plotting
import seaborn as sns # High level Plotting
import statsmodels.api as sm # Modeling, e.g. ANOVA

# Import only specific modules from a library
# we'll use this for the t-test function
from scipy import stats
# from scipy.stats import zscore

# Import only specific functions from a library 
# ols is for ordinary least squares
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load our data set:
plant_growth = pd.read_csv('data/plant_growth.csv')

# Explore
plant_growth.info() # overview
plant_growth.shape # dimensions (rows, cols)
plant_growth # print entire data set
plant_growth.columns # names of columns

# Descriptive stats (mean, sd)
plant_growth.describe() # numerical columns
plant_growth['group'].value_counts() # describe a categorical variable
plant_growth.head() # first 5 rows
np.mean(plant_growth['weight']) # specific function for weights
plant_growth.groupby(['group']).describe()  # Group-wise statistics
plant_growth.groupby(['group'])['weight'].mean() # call mean as a builtin method
plant_growth.groupby(['group']).mean() # call mean as a builtin method
plant_growth['weight'].mean() # called as a builtin method

# Data visualizations
# Box plot
sns.boxplot(x = 'group', y = 'weight', data = plant_growth)

# Dot plot
sns.catplot(x = 'group', y = 'weight', data = plant_growth)

# Mean and sd plot
sns.pointplot(x = 'group', y = 'weight', data = plant_growth, join = False)

# Inferential stats (t-test, ANOVA)

# Two-smaple t-test
# First step: fit a model
# Continuous response (y, weight, dependent) to a
# categorical predictor (x, group, independent)
model = ols("weight ~ group", plant_growth)
# ~ "described by"
# model
print(model)
# call the fit method
results = model.fit()
# results
print(results)
# call specific attributes
results.params # all
results.params['group[T.trt2]'] # only one

# Directly to t-tests:
results.summary()

# One-way ANOVA
aov_table = sm.stats.anova_lm(results)
aov_table

help(sm.stats.anova_lm)

# Exercise 2.1 - chick weights analysis

# Import and examine the data
chick = pd.read_csv('./data/Chick Weights.txt', delimiter = "\t")
chick.info()

# Mean and sd for each type of feed
chick.groupby(['feed']).describe()

# Number of chicks for each type of feed
chick['feed'].value_counts()

# Within-group z-scores
chick.groupby(['feed'])['weight'].transform(lambda x: stats.zscore(x, ddof = 1))
chick.groupby(['feed'])['weight'].agg('mean')
# Transform = same number of output as input (z-score, log)
# Aggregrations = 1 output for any number of input (mean, sd)

stats.zscore([3,5,6,8,4,3,], ddof = 1)
# chick.groupby('feed').stats.zscore()
# stats.zscore(chick.groupby(['feed'])['weight'], ddof = 1)
# help(zscore)

# Plot
# sns.stripplot(x='feed', y='weight', data=chick)
sns.catplot(x='feed', y='weight', data=chick)

# One-way ANOVA
model = ols('weight ~ feed', chick)
results = model.fit()
anova_table = sm.stats.anova_lm(results)
anova_table

# Tukey’s post-hoc test
print(pairwise_tukeyhsd(chick['weight'], chick['feed']))

## Chapter 3: Python Fundamentals

# Four key types
a = 1 # int (whole numbers)
type(a)
b = 0.25 # float (numeric, real numbers)
type(b)
c = 'Berlin' # str (object)
type(c)
d = True # bool (binary, logical, True/False)
type(d)

# Arithmetic operators, table section 3.2
32 % 6 # remainder
32 // 6 # int division
2 ** 3 # exponents

1 + 1 # basic addition
'1' + '1' # add str
'1' * 5 # multiply str
'1' '1' # bad syntax :/

# Functions
# e.g. 
numb = 8
np.log2(numb) # (2 ** x = 8)
math.log2(8)
math.log(8, 2)

# Lists - Data containers, pt 1
# A builtin type
# 1-dimensional
# Defined with []
# heterogenous, i.e. many different data types

# [1, 6, 9, 36] + 100 # TypeError
[1, 6, 9, 36] + [100] # Add two lists together
[1, 6, 9, 36] + [10, 100]
[1, 6, 9, 36] + [100, 200, 300, 400]
[1, 6, 9, 36] + [100, 'dogs', 300, 400]

heights = [167, 188, 178, 194, 171, 169]
# Table 3.1 for typical functions
sum(heights)
len(heights)
np.mean(heights)

# Exercise 3.3:
lower95 = np.mean(heights) - 1.96 * np.std(heights)/(len(heights)**0.5)
upper95 = np.mean(heights) + 1.96 * np.std(heights)/(len(heights)**0.5)
# np.sqrt(100)
# 100**0.5
# x ** 2 = 100, solve for x
# x = 100**0.5
lower95
upper95
# Exercise 3.4:
cities = ['Munich', 'Paris', 'Amsterdam', 'Madrid', 'Istanbul']
dist = [584, 1054, 653, 2301, 2191]

len(dist)
max(dist)
min(dist)
np.mean(dist)

# Exercise 3.5:
# np.histogram(dist)
sns.stripplot(dist)
sns.stripplot(cities, dist)
# sns.distplot(dist) 

# Defining functions
# Basic form:

# A multi line comment: use """..."""
"""This is 
a long description
about my work...
"""
# Use them to also add "docstrings" to a function
def addNumbs(x, y):
    """Add two numbers together"""
    z = x + y
    return z

# Call the function
addNumbs(5,8)

# doc strings can be called as a method:
addNumbs.__doc__

# Exercise 3.6: Write a function to return lower95 for any list
def confInt(x):
    z = np.mean(x) - 1.96 * np.std(x)/np.sqrt(len(x))
    return z

confInt(heights)
confInt([3,4,3,2,2,1,3,4,3,4,5])

# Returning two values - output is a tuple
# like a list, but use () instead of []
# tuples are immutable - they can't be modified
def mathFun(x, y):
    """Add and subtract two numbers together"""
    result = (x + y, x - y)
    
    return result
    
myOutput = mathFun(4, 6)
type(myOutput)

# Exercise 3.7 Return a tuple of two values (lower and upper limit)
def confInt(x):
    lower = np.mean(x) - 1.96 * np.std(x)/np.sqrt(len(x))
    upper = np.mean(x) + 1.96 * np.std(x)/np.sqrt(len(x))
    return (lower, upper)

confInt(heights)

# inner functions
# Define three_shouts
def three_shouts(word1, word2, word3):
    """Returns a tuple of strings
    concatenated with '!!!'."""

    # Define inner
    def inner(word):
        """Returns a string concatenated with '!!!'."""
        return word + '!!!'

    # Return a tuple of strings
    return (inner(word1), inner(word2), inner(word3))

three_shouts('A', 'B', 'C')
# inner('hello') # no access to inner()

# Exercise 3.8 Inner functions
def confInt(x):
    """calculate the conf int"""

    def error(x):
        return 1.96 * np.std(x)/np.sqrt(len(x))

    lower = np.mean(x) - error(x)
    upper = np.mean(x) + error(x)

    return (lower, upper)

confInt(heights)

# Function generators - Functions that make new functions
def raise_value(n):
    """return a function"""

    def inner(x):
        """raise x to the power of n"""  
        raised = x ** n
        return raised
    
    return inner

raise_2 = raise_value(2)
raise_2(3)

raise_4 = raise_value(4)
raise_4(3)
3**4

# For conf ints, using the N dist we have 1.96
# For t-dist, this is dependent on n (smaple size)
# e.g. n=10, use 2.26

# Exercise 3.9: Function generators
def setCorrection(c):
    def confInt(x):
        """calculate the conf int"""

        def error(x):
            return c * np.std(x)/np.sqrt(len(x))

        lower = np.mean(x) - error(x)
        upper = np.mean(x) + error(x)

        return (lower, upper)

    return confInt

CI_Norm = setCorrection(1.96)
CI_Norm(heights)

CI_n10 = setCorrection(2.26)
CI_n10(heights)

# Setting default values
def confInt(x, correct = 1.96):
    """calculate the conf int"""

    lower = np.mean(x) - correct * np.std(x)/np.sqrt(len(x))
    upper = np.mean(x) + correct * np.std(x)/np.sqrt(len(x))

    return (lower, upper)

confInt(heights, 1.96)
confInt(heights)
confInt(heights, 2.26)

# Determine the correction factor dependent on 
# Sample size on a t-distbutions
# i.e. df (degrees of freedom) = n-1
# First, get the t distribution
# from scipy.stats import t

def confInt(x):
    """calculate the conf int"""

    correct = stats.t.ppf(0.975,  len(x)-1 )

    lower = np.mean(x) - correct * np.std(x)/np.sqrt(len(x))
    upper = np.mean(x) + correct * np.std(x)/np.sqrt(len(x))

    return (lower, upper)

confInt(heights)

# 3.7.8 lambda functions
def raise_to_power(x, y):
    """exponents"""
    return x ** y

raise_to_power(2,3)

# anonymous, unnamed, lambda
lambda x, y: x ** y

# e.g. we saw it already in the context of a DataFrame
# the \ breaks up a long command onto the next line
chick.groupby(['feed'])['weight']. \
   transform(lambda x: stats.zscore(x, ddof = 1))

# or use the map()
# heights ** 3 # lists are not iterable
cubed = map(lambda x: x ** 3, heights)
# This results in a map object
# Which we can convert to a list 
list(cubed)

# cubed_2 = map(lambda x, y: x ** y, {heights, 3})
# This results in a map object
# Which we can convert to a list 
# list(cubed_2)

# map without lambda
list(map(np.mean, [[1,2,3], [4,5,6]]))

# e.g. filter()
list(filter(lambda x: x > 175, heights))

# Exercise 3.14 Returning meaningful messages:
# i.e. a function that returns:
# There is a 95% chance that the true population 
# parameter is covered by the region XX, YY
def confInt(x):
    """calculate the conf int"""

    correct = stats.t.ppf(0.975,  len(x)-1)

    lower = np.mean(x) - correct * np.std(x)/np.sqrt(len(x))
    upper = np.mean(x) + correct * np.std(x)/np.sqrt(len(x))

    print(f"There is a 95% chance that the true population parameter is covered by the region [{round(lower, 2)}, {round(upper, 2)}].")

confInt(heights)

# See section 3.8
# use f-string formatting
name = 'Rick'
print(f"Hi my name is {name}")
print(f"{100 * 566} is a big number,\nbut {100 / 566} is not")
print(f"{np.mean([1,2,3])} is the mean of 1,2,3")

# Methods:
# access using the . notation

# like, but not the same as...
np.mean(heights) # a function in the np module
len(heights) # a function

# len is not a method
# heights.len()
# 'list' object (heights) has no attribute 'len'

# Function takes an object as an argument
# Objects call methods
# Methods are just functions specific to a given class of object

l = [1, '2', True]
l
l.append('appended value')
l

name = "Berlin"
name.upper()
name.lower()

dist
dist.append(560)
dist
dist.pop(3) # extract a value
dist # one less value

cities = ['Munich', 'Paris', 'Amsterdam', 'Madrid', 'Istanbul']
dist = [584, 1054, 653, 2301, 2191]
dist
dist.sort()
dist
dist.sort(reverse = True)
dist
dist.remove(653)
dist # one less value

# Exercise 3.15
# try these methods that we haven't explored yet:
# copy()
heights
heights_2 = heights.copy()
heights_2
heights_2.append(455)
heights_2
heights

# clear()
heights_2.clear()
heights_2

# A list of lists
heights.append([45, 74, 23])
heights

# Reset list
heights = [167, 188, 178, 194, 171, 169]
heights

# reverse()
heights.reverse()
heights

# count()
heights.count(194)

# These are methods we already saw,
# but the are NOT methods for lists!
# heights.value_counts()
# heights.mean() # called as a builtin method

# index()
heights
heights.index(194)

# Revisit for iterables
range(3)
list(range(3)) # 0,1,2

heights.extend(range(3))
heights
heights.extend([4,6,8])
heights

# insert()
heights.insert(8, 653)
heights

help(round)

# Attributes
l.__len__

len(l)

addNumbs.__doc__
setattr(addNumbs, '__doc__', "hello")
addNumbs.__doc__


# __ are reserved names from python
# but you can use them like all other attributes!
# pronunce them as "dunder"
# don't define you own variables, etc, using __xx__

# Attributes on a data frame
plant_growth.shape
plant_growth.columns

# A method
plant_growth.info()

# Methods are class functions, i.e. functions specific to a type of object
# Attributes are class parameters, i.e. values specific to a type of object

getattr(plant_growth, 'columns')
plant_growth.columns
dir(plant_growth)
setattr(plant_growth, 'Rick', 'Scavetta')
plant_growth.Rick

getattr(plant_growth, 'shape')
plant_growth.shape

# Dictionaries - Data containers, pt 2
# aka dict

# key/value pairs
# list, use []
# tuples, use ()
# dict, use {}
d = {'int_value': 3,
     'bool_value': False,
     'str_value': 'hello'}

d

# Access values using their key
d['bool_value']

# values can be more than length one
# i.e. they can contain a list
organizations = {'name': ['Volkswagen', 'Daimler', 'Allianz', 'Charite'],
                 'structure': ['company', 'company', 'company', 'research']}
organizations['name']

# Make a dictionary (key/value pair) from lists:
heights = [167, 188, 178, 194, 171, 169]
persons = ["Mary", "John", "Kevin", "Elena", "Doug", "Galin"]

# key is persons, 
# value is corresponding value in heights
heights_persons = dict(zip(persons, heights))
heights_persons


heights_persons.values()
heights_persons.keys()

# in a list use index position:
heights[3]
heights.index(194)
heights_persons['Mary']

# NumPy Array - Data containers, pt 3
# n-dimensional (1, 2, ...)
# what we think as a matrix
# They can only have ONE type

xx = [3, 8, 9, 23]
type(xx)

yy = np.array([3, 8, 9, 23])
type(yy)

# A list [] of lists [], [], range()
zz_list = [[5, 7, 8, 9, 3],
           [0, 3, 6, 8, 2],
           range(5)]
zz_list # 1-dimension

zz_array = np.array([[5, 7, 8, 9, 3],
                     [0, 3, 6, 8, 2],
                     range(5)])
zz_array # 2-dimensional

# What is range(5)?
list(range(5))
list(range(3, 5))

# for reference:
import keyword
keyword.kwlist

# pandas DataFrames - Data containers, pt 4

# Get a DataFrame when impoting a file

# Or from a dict:
foo1 = [True, False, False, True, True, False]
foo2 = ["Liver", "Brain", "Testes", "Muscle", "Intestine", "Heart"]
foo3 = [13, 88, 1233, 55, 233, 18]

# Collect list into a dataframe
foo_df = pd.DataFrame({'healthy': foo1, 'tissue': foo2, 'quantity': foo3})
foo_df
type(foo_df)

# Or from a list of keys and values:
list_names = ['healthy', 'tissue', 'quantity']
# A list of lists
list_cols = [foo1, foo2, foo3]
# zip put the key/value pairs together
pd.DataFrame(dict(zip(list_names, list_cols)))

# Access information by:
# Name (here)
# Position (later, indexing)

# columns
foo_df.columns
# rows
foo_df.index

foo_df['healthy'] # Series
type(foo_df['healthy'])

foo_df[['healthy']] # DataFrame
type(foo_df[['healthy']])

foo_df.healthy # a Series

foo_df[['quantity', 'healthy']] # DataFrame

# each column is a Series
# DataFrames are build upon np.arrays
# i.e. Series can only be ONE type!

foo_df.info()

quantity_list = foo3.copy()
# quantity_list.mean() # no!
np.mean(quantity_list) # yes :)
# quantity_list/100 # no!

quantity_array = np.array(foo3)
quantity_array.mean()
quantity_array/100
quantity_array.astype("str")
# quantity_array.name # no!

quantity_Series = foo_df['quantity']
quantity_Series.mean()
quantity_Series/100
quantity_Series.astype("str")
quantity_Series.name

test_Series = pd.Series(quantity_array)
test_Series.name = "hello"
test_Series

# four main data containers
# list - 1-dimensional, heterogenous type
# dictionary - key/value pairs
# np.array - n-dimensional, homogenous type
# pd.DataFrame - 2-dimensional, collection of Series

# plus these special ones
# tuple - 1-dimensiona, immutable, heterogenous type
# Series - Special case of np.array in a DF

# Beginning with two lists
cities = ['Munich', 'Paris', 'Amsterdam', 'Madrid', 'Istanbul']
dist = [584, 1054, 653, 2301, 2191]

# Exercise 4.1: Make a dictionary manually, then a DF
pd.DataFrame({'cities':cities,
              'dist':dist})

# Exercise 4.2: 
list_names = ['cities', 'dist']
list_values = [cities, dist]

pd.DataFrame(dict(zip(list_names, list_values)))

# Broadcasting
# We already saw...
foo_df.quantity / 100
foo_df['quantity'] / 100

foo_df['new'] = 0
foo_df

foo_df['new'] = range(6, 12)
foo_df

foo_df['new'] = ['A', 'B', 'C', 'a', 'b', 'c']
foo_df

# Removing a column
foo_df = foo_df.drop('new', axis = 1)
# axis 0 = rows  
# axis 1 = columns
foo_df
foo_df_2 = foo_df.copy()
foo_df_2.drop(['healthy', 'tissue'], axis = 1)

# mtcars case study
mtcars = pd.read_csv('./data/mtcars.csv')
mtcars.info()
# import os
# os.listdir()

# Calculate the correlation between mpg and wt and test if it is significant
mtcars['mpg'].corr(mtcars['wt'])
mtcars.mpg.corr(mtcars['wt'])

# Visualize the relationship in an XY scatter plot
sns.scatterplot('wt', 'mpg', data=mtcars)
sns.regplot('wt', 'mpg', data=mtcars)

# Calculate linear model
reg1 = sm.OLS(endog = mtcars['mpg'], 
              exog = mtcars['wt'])
regResults = reg1.fit()
regResults.summary()
0.720 # R^2 from model
-0.8676**2 # R^2 from correlation function

model = ols("mpg ~ wt", mtcars)
# call the fit method
results = model.fit()
# results
print(results)
results.params
dir(results)
results.summary()

# Convert weight from pounds to kg
mtcars['wt_kg'] = (mtcars['wt']*1000)*0.4536