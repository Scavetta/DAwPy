# Intro to R
# Rick Scavetta
# 26 Mar 2018
# IMPRS-LS DA workshop

# Clear workspace
rm(list = ls())

# Load packages
library(dplyr)
library(ggplot2)
library(purrr)
library(tidyr)

# Basic R notation
n <- log2(8) # the log 2 of 8
n # shortcut print(n)

# A simple workflow
# The data:
data(PlantGrowth)

# export for python
# write.csv(PlantGrowth, "plant_growth.csv", row.names = FALSE)


# Summary statistics
# What are the groups? i.e. levels
# How many?
levels(PlantGrowth$group)
nlevels(PlantGrowth$group)

# mean:
mean(PlantGrowth$weight)

# group-wise means:
tapply(PlantGrowth$weight, PlantGrowth$group, mean)

# use dplyr:
# use shift+ctrl+m for %>%
PlantGrowth %>%
  group_by(group) %>%
  summarise(avg = mean(weight))

# Visualisation: use ggplot2
# Specify axes (scales) using aes()
# Add visuals using a geom_ function and +
ggplot(PlantGrowth, aes(group, weight)) +
  geom_point(position = position_jitter(0.2),
             shape = 1)

ggplot(PlantGrowth, aes(group, weight)) +
  geom_boxplot()

# Statistics:
# make a linear model
plant.lm <- lm(weight ~ group, data = PlantGrowth)

# T-tests:
plant.lm # short cut for print(plant.lm)
summary(plant.lm)

# get access to all results:
plant.lm$coefficients

# ANOVA:
plant.anova <- anova(plant.lm)
plant.anova$`Pr(>F)`[1]

# Another way to set up an ANOVA:

plant.aov <- aov(weight ~ group, data = PlantGrowth)
summary(plant.aov) # get p-value,
# summary(aov()) is the same as anova(lm())

# All pair-wise t-test:
TukeyHSD(plant.aov)

# Reproduce in Markdown with:
chickwts

# Element 2: Functions
# Everything that happens, is because of a function

# Arithmetic operators
34 + 6
# just to know... this is the same as:
`+`(34, 6)

# Order of operations
# BEDMAS - Brackets, expon, div, mult, add, sub
2 - 3/4 # 1.25
(2 - 3)/4 # -0.25

# use () in math order and function input

# Make some objects:
n <- 34
p <- 6

n + p

# Exercise 1, p27:
1.12 * 3 - 0.4
m <- 1.12
b <- -0.4
m * 3 + b
m * 8 + b
m * 9 + b
m * 23 + b

# Generic form of functions
# fun_name(args)

# args may be named or unnamed
# args may be none to many
# e.g.
log2(8)
log2(x = 8)
log(8, base = 2)
log(8, 2)
log(base = 2, 8) # confusing :/

# Positional matching, versus
# Named arguments

# Some basic and common functions:
# Combine unnamed arguments: c()

xx <- c(3, 8, 9, 23)
xx

myNames <- c("healthy", "tissue", "quantity")
myNames

# Sequential Numbers:
seq(from = 1, to = 100, by = 7)
foo1 <- seq(1, 100, 7) # same as above

# use objects in functions:
foo2 <- seq(1, n, p)
foo2

# The : operator for seq(x, y, 1)
1:10
seq(1, 10, 1)

# Type of math functions:
# 1 - transformation: Every value is transformed in the SAME way
# +, -, *,
# Normalisation, Z-scores
# log, log10, log2, sqrt
# Multiply every element by 3
log(foo1)
scale(foo1) # Z-scores

# 2 - aggregration
# Returns a SINGLE (or a small number of) value(s)
# mean, sd, median, IQR, range, n
# sum(), prod()
sum(foo1) # add all values
prod(6:1) # 6-factorial: 6!
length(foo1) # n

# Exercise 2, page 30:
foo2 + 100 # trans
foo2 * 3 # trans
foo2 + foo2 # trans
sum(foo2) + foo2 # agg, followed by trans

1:3 + foo2 # trans, with vector recycling

1:4 + foo2 # trans, with warning

######### fundamental concept ############
######### Vector recycling ###############

# Exercise 3, p 30:
m
b
xx
m * xx + b

# Exercise 4, p 30:
m2 <- c(0, 1.12)

# goal:
0 * xx + b # Where m is 0
1.12 * xx + b # where m is 1.12

m2 * xx + b

# one solution:
# for loops == :(, but not really prefered in R
for(i in m2) {
  print(i * xx - 0.4)
}



# making your own functions:
equation <- function(x, m = 1.12) {
  m * x - 0.4
}

# use it:
rm(b) # remove b
equation(1:100)

# How to reiterate:
b <- -0.4
# define an anonymous function in lapply
lapply(m2, function(x) xx * x + b)
# or a named function:
lapply(m2, function(x) equation(xx, x))

# use sapply to simplify to the
sapply(m2, function(x) xx * x + b)
sapply(m2, function(x) equation(xx, x))

# A newer way using purrr
map(m2, ~ equation(xx, .) )
map(m2, ~ xx * . + b)

# m2 %>%
# map(~ xx * . + b) %>%
# map_dfr(~ as.data.frame(t(as.matrix(.))))

# Element 3: Objects
# anything that exists is an objects

# Vectors - 1D, homogenous
# e.g.
xx
m2
foo1
foo2

# (user defined) Atomic vector types:
# Logical (boolean/binary) T/F, TRUE/FALSE, 1/0
# Integer - 0,1,2,3,
# Double - 3.14, 6.8,...
# Character (string) - All A-Z, a-z, 0-9, white space, symbols

typeof(foo1)
typeof(myNames)

foo3 <- c("Liver", "Brain", "Testes",
          "Muscle", "Intestine", "Heart")

foo4 <- c(T, F, F, T,T, F)
typeof(foo4)

# as an aside:
test <- c(1:10, "bob")
test
as.numeric(test)
# Major problem # 1: wrong type!
# Typical solution: Coercion
test <- test[-11] # remove last element, still chr
mean(test)
test <- as.integer(test)
mean(test)

# Lists - 1D, heterogenous
# Data Frames - 2D, heterogenous where every
#               element is a vector of the same length
# heterogenous: every element can be a different type.
# rows == observations
# cols == variables

# Make it from scratch:
foo.df <- data.frame(foo4, foo3, foo2)
foo.df

# attributes:
attributes(foo.df)
typeof(foo.df)
# Class determines how other functions deal
# with an object: object-oriented programming
summary(foo.df)
summary(plant.lm)

print(foo.df)
print(plant.lm)

typeof(plant.lm)
class(plant.lm)

# Access attributes with accessor functions
# e.g.
class(foo.df)
names(foo.df) <- myNames

# Major Problem # 2: Wrong class (structure)
# coerce to right class

# Major Problem # 3: Wrong format (also structure)
# rearrange (see dplyr and tidyr later)

# Exploring data frames:
str(foo.df) # look at structure!
glimpse(foo.df) # from dplyr
summary(foo.df)

dim(foo.df)
nrow(foo.df)
ncol(foo.df)

# The floating point trap:
0.3/3
0.1 == 0.3/3 # test of equivalency
all.equal(0.1, 0.3/3)

typeof(foo.df)
class(foo.df)

# sapply(foo.df, typeof)

# Element 4: Logical Expressions
# Relational operators: Ask YES/NO questions
# == equivalency
# != non-equivalency
# >, <, >=, <=
# !x negation of x, where x is a logical vector
# e.g.
foo4 # logical
!foo4 # negation
n
p
n > p
n < p
n == p

# The results are ALWAYS logical vectors!

# Logical operators: combine YES/NO questions
# & AND
# | OR
# %in% WITHIN

# classic funct: subset()
# dplyr method: filter()

# apply to logical data:
# only healthy
subset(foo.df, foo.df$healthy == TRUE)
subset(foo.df, healthy == TRUE)
subset(foo.df, healthy)

foo.df %>%
  filter(healthy)

# unhealthy
subset(foo.df, !healthy)
subset(foo.df, healthy == FALSE)

# apply to numerical data:
# Below 10:
subset(foo.df, quantity < 10)

foo.df %>%
  filter(quantity < 10)

# Ranges: 10-20
subset(foo.df, quantity > 10 & quantity < 20)

# Meaningless
subset(foo.df, quantity > 10 | quantity < 20)

# Extreme values beyond 10, 20:
subset(foo.df, quantity < 10 | quantity > 20)

# Impossible
subset(foo.df, quantity < 10 & quantity > 20)

# Apply to character data:
# NO pattern matching!
# Heart samples:
subset(foo.df, tissue == "Heart")

# 2 or more: Liver & Heart
# good, but inefficient for many searches
subset(foo.df, tissue == "Heart" | tissue == "Liver")

# TERRIBLE - NEVER do it.
subset(foo.df, tissue == c("Heart", "Liver"))
subset(foo.df, tissue == c("Liver", "Heart"))

# The proper way: alternative for multiple |
subset(foo.df, tissue %in% c("Heart", "Liver"))
subset(foo.df, tissue %in% c("Liver", "Heart"))

# everything except heart & liver:
subset(foo.df, !(tissue %in% c("Heart", "Liver")))

length(foo1)

# Element 5: Indexing
# Find info according to position using []

# Vectors:
foo2
foo.df$quantity

# Using numbers:
foo.df$quantity[3] # 3rd element
foo.df$quantity[p] # pth element
foo.df$quantity[3:p] # 3rd to the pth element

foo.df$quantity[3:length(foo.df$quantity)] # 3rd to the last value
foo.df$quantity[3:nrow(foo.df)] # 3rd to the last value

foo.df$quantity[p:3] # pth to the 3rd element

# Using logical vectors:
foo2[foo2 < 10]

# Data frames:
# 2D: [<rows> , <columns>]
foo.df[3,] # 3rd row, all columns
foo.df[3, 3] # 3rd row, 3rd column
foo.df[3, "quantity"] # 3rd row, 3rd column
foo.df[3:p, c("healthy", "tissue")] # select by name
foo.df[3:p, -2] # 3rd to pth, exclude col 2
foo.df[3:p, names(foo.df) != "quantity"] # Use logical vectors

# mixing dimensions:
foo.df$quantity[3,] # 1D obj, 2D index :/
foo.df[3] # 2D obj, 1D index shortcut to col selection
foo.df[,3] # 3rd col as a vector (!)

# choose one col base on critera from another:
# all the tissue with low quantity (< 10)
foo.df[foo.df$quantity < 10, "tissue"]
# the same as:
foo.df$tissue[foo.df$quantity < 10]

# This is just a more flexible version of subset!
subset(foo.df, quantity < 10, select = "tissue")

foo.df %>%
  filter(quantity < 10) %>%
  select("tissue")

?intersect

# Element 8: Factor Variables (with levels)
# Categorical variables (with groups)
# aka discrete or qualitative

# e.g.
PlantGrowth$group

# factors are a special class of type integer
typeof(PlantGrowth$group)
class(PlantGrowth$group)
# with labels:
# e.g.
foo3 # character
foo.df$tissue # factor
# each label gets an integer ID
str(foo.df)

# some problems:
xx <- c(23:27, "bob")

test <- data.frame(xx)
str(test)
mean(test$xx)
# Get rid of characters
test$yy <- as.double(as.character(test$xx))
mean(test$yy, na.rm = T)

# Elements 9 & 10: Tidy data & Split-apply-combine
# work on a new data set:
PlayData <- data.frame(type = rep(c("A", "B"), each = 2),
                       time = 1:2,
                       height = seq(10, 40, 10),
                       width = seq(50, 80, 10))

PlayData

# To rearrange our data, use tidyr::gather()
# four arguments: DF, key, value, either the ID or MEASURE vars
# key & value are OUTPUT names, NOT INPUT
gather(PlayData, key, value, -c(type, time)) # with ID
PlayData.t <- gather(PlayData, key, value, c(height, width)) # with MEASURE

# Scenario: 1: Compare by key (height/width)
# just use the original data set
PlayData$height/PlayData$width

# Scenario 2: Compare by time (2-1)
# Two solutions
# Solution 1 - revert to messy data and get proper variable names:
spread(PlayData.t, time, value) # time is now the "key" variable

# Solution 2 - Split-apply-combine with dplyr
# 1st, split/group according to some variable(s)
# 2nd, apply some function
# 3rd, combine together
PlayData.t %>%
  group_by(type, key) %>%
  summarise(growth = diff(value))

# from:
diff(c(40, 50, 75, 100))
