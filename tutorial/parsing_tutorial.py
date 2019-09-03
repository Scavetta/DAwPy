import pandas as pd # For DataFrame and handling

foo1 = [True, False, False, True, True, False]
foo2 = ["Liver", "Brain", "Testes", "Muscle", "Intestine", "Heart"]
foo3 = [13, 88, 1233, 55, 233, 18]

# Collect list into a dataframe
foo_df = pd.DataFrame({'healthy': foo1, 'tissue': foo2, 'quantity': foo3})
foo_df

## Parsing data

# before: find information by name
foo_df['tissue'] # Series
foo_df[['tissue']] # DataFrame

foo_df.tissue

# here: find information by positon, or search criteria

# Use iloc ("index location") [] here is for indexing
# no ,
# use [ <rows> , <cols> ]
foo_df.iloc[0] # first row as a Series, short cut for foo_df.iloc[0,:]
foo_df.iloc[[0]] # first row as a DataFrame
foo_df.iloc[[4,1]] # first & second row

# Using ranges and access col with ,
foo_df.iloc[0,:] # first row, all columns
foo_df.iloc[[4,1],:] # first & second row, all columns

foo_df.iloc[:,:2] # all rows, first 2 cols 

""" Indexing in Python begins at 0!
If no , is present then we retrieve rows
We specify a range using the : operator, as per start:end. If using , this must be included
The end position is exclusive, i.e. not included in the series!
If no start or end position is given, then we take the beginning to the end, respectively.
Using a negative number, i.e. -1 begins counting in the reverse direction. """

foo_df.iloc[-1, :]

# Exercise 5.1: 
# The 2nd to 3rd rows?
foo_df.iloc[[1,2]]
foo_df.iloc[1:3]

# The last 2 rows?
foo_df.iloc[[-1,-2]]
foo_df.iloc[-3:]

# From the 4th to the last row? But without hard-coding, i.e. regardless of how many rows my data frame contains
foo_df.iloc[3:]

# A random row in foo_df? (sample)
foo_df.sample()
foo_df.iloc[:,:2].sample(2)

# The first 3 rows, without using iloc (head)
foo_df.head(3)

# The last 3 rows, without using iloc (tail)
foo_df.tail(3)

# For lists, you don't need .iloc
list(range(6))[:3]
[2,4,6,7,9][3]
# here, [:3] and [3] are index calls onto lists []

# foo_df[[1,2]]

# List all the possible objects we can use inside iloc[]

foo_df.iloc[[True, False, True, False, True, False]]

# Integers? yes!
# Floats? no!
# Characters? no! (use .loc[], or just call [])

# Boolean? yes! as a list :)
foo_df.iloc[[True, False, True, False, True, False]]
foo_df.iloc[[False, False, False, False, True, False]]

# Boolean? as a Series? if you make it an array or list
foo_df.iloc[list(foo_df.healthy)]
foo_df.iloc[np.array(foo_df.healthy)]
# foo_df.iloc[foo_df.healthy] # no

# A heterogenous list? no e.g. [1, 'a', True]
# A homogenous list? yes, see above, with integers, boolean

# Before we move onto Booleans...
# so far we used [ <start> : <end> ], i.e.
list(range(10))[3:6]
# but, it's really... [ <start> : <end> : <step> ]
list(range(10))[1::3]

# Exercise 5.3 (Indexing at intervals) 
# Use indexing to obtain all the odd and even
# rows only from the foo_df data frame
# Even rows: 0,2,4
# foo_df.iloc[1:4]
# foo_df.iloc[[1,2]]
foo_df.iloc[::2,:]

# Odd rows: 1,3,5
foo_df.iloc[1::2,:]

# Logical Expressions
# Asking and combining Yes/No questions
# Composed of:
# 1 - Relational operators - Asking
# 2 - Logical operators - Combining
# Answers will ALWAYS be Boolean :)

# 1 - Relational operators
# == equivalency, are things equal
# != non-equivalency
# >, <, >=, <=

# 2 - Logical operators
# & and
# | or

# Get a specific tissue: heart
foo_df[foo_df.tissue == "Heart"]

# Get a specific tissue and quantity:
foo_df[(foo_df.tissue == "Heart") | (foo_df.quantity == 233)]

# Exercise 5.4 Subset for boolean data:
# Only “healthy” samples.
foo_df[foo_df.healthy == True]
foo_df[foo_df.healthy]
# Only “unhealthy” samples.
foo_df[foo_df.healthy != True]

# Exercise 5.5 Subset for numerical data:
# Only low quantity samples, those below 100.
foo_df[foo_df.quantity < 100]
# Quantity between 100 and 1000,
foo_df[(foo_df.quantity > 100) & (foo_df.quantity < 1000)]
# Quantity below 100 and beyond 1000.
foo_df[(foo_df.quantity < 100) | (foo_df.quantity > 1000)]

# Exercise 5.6 Subset for strings:
# Only “Heart” samples
# “Heart” and “Liver” samples
foo_df[(foo_df.tissue == "Heart") | (foo_df.tissue == "Liver")]

mySearch = ["Heart", "Liver"]
foo_df[foo_df.tissue == mySearch]

"Heart" in list(foo_df.tissue)

# Everything except “Intestines”
foo_df[(foo_df.tissue != "Intestine")]
