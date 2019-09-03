# import libraries
import numpy as np
import seaborn as sns
import pandas as pd

# Exercise 6.2 - Import data
protein = pd.read_csv('./data/Protein.txt', delimiter = '\t')

# Examine our data
protein.describe()
protein.columns
protein.head()
protein['Intensity.H']
protein.info()

# Exercise 6.8 - Remove contaminants
protein['Contaminant']
# How many contaminants are present in the data-set?
protein.Contaminant.value_counts()
# len(protein[protein.Contaminant == "+"])
# protein[protein.Contaminant == "+"].shape
sum(protein.Contaminant == "+")

# What fraction of the total do they represent?
sum(protein.Contaminant == "+")/len(protein)

# Remove them
protein = protein.loc[protein.Contaminant != "+"]

# Exercise 6.2 - log10
protein[['Intensity.L', 'Intensity.M', 'Intensity.H']] = \
    np.log10(protein[['Intensity.L', 'Intensity.M', 'Intensity.H']])
protein.describe()

# Add log10s
protein['Intensity.M.L'] = protein['Intensity.L'] + protein['Intensity.M']
protein['Intensity.H.M'] = protein['Intensity.H'] + protein['Intensity.M']
protein.info()

# Exercise 6.3
sns.scatterplot('Ratio.H.M', 'Intensity.H.M', data = protein, alpha = 0.5)
# sns.catplot('Ratio.H.M', 'Intensity.H.M', data = protein, alpha = 0.5)

# Exercise 6.4 - log2
protein[['Ratio.H.M', 'Ratio.M.L']] = \
    np.log2(protein[['Ratio.H.M', 'Ratio.M.L']])

# Exercise 6.5
sns.scatterplot('Ratio.H.M', 'Intensity.H.M', data = protein, alpha = 0.5)

# Exercise 6.7 - Normalization
protein['Ratio.H.M'] = protein['Ratio.H.M'] - np.mean(protein['Ratio.H.M'])
protein['Ratio.M.L'] = protein['Ratio.M.L'] - np.mean(protein['Ratio.M.L'])
sns.scatterplot('Ratio.H.M', 'Intensity.H.M', data = protein, alpha = 0.5)

# Exercise 6.9 - Find protein values
myStr = ['GOGA7_MOUSE', 'PSA6_MOUSE', 'S10AB_MOUSE']
protein.Uniprot
protein[['Description', 'Ratio.H.M', 'Ratio.M.L']].loc[protein.Uniprot.isin(myStr)]

# Pattern matching with regular expressions
protein[protein.Uniprot.str.contains('GOGA7|PSA6|S10AB', regex = True) == True]

# Exercise 6.10 - Find significant hits
HM_Sig = protein[protein['Ratio.H.M.Sig'] <= 0.05]
HM_Sig.shape

# Exercise 6.11 - Find extreme values
HM_Extreme = protein[abs(protein['Ratio.H.M']) >= 2]
HM_Extreme.shape

# Exercise 6.12 - Find top 20 values
#HM_top20 = protein.sort_values('Ratio.H.M', na_position = 'first').tail(20)
HM_top20 = protein.sort_values('Ratio.H.M', ascending = False).head(20)
HM_top20 = HM_top20[['Uniprot', 'Ratio.H.M']]
HM_top20.shape

ML_top20 = protein.sort_values('Ratio.M.L', ascending = False).head(20)
ML_top20 = ML_top20[['Uniprot', 'Ratio.M.L']]
ML_top20.shape

# Exercise 6.13 - Find intersections
# What's the intersection?
np.intersect1d(HM_top20.Uniprot, ML_top20.Uniprot).shape

# What's in HM that's not in ML?
np.setdiff1d(HM_top20.Uniprot, ML_top20.Uniprot).shape

# What's in ML that's not in HM?
np.setdiff1d(ML_top20.Uniprot, HM_top20.Uniprot).shape

# What's the total unique set?
np.union1d(HM_top20.Uniprot, ML_top20.Uniprot).shape
