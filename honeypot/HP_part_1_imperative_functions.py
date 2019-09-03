# Honeypot Case Study

# import necessary packages
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import chisquare

# For cosine matching
SCORE_THRESHOLD = 0.50

# B - Read in templates
df_templates = pd.read_csv('./data/message_templates.csv')
df_templates.columns
df_templates.message
df_templates['message'] = [BeautifulSoup( \
    '' if not text else text).get_text() for text in df_templates['message']]
df_templates['type'] = 'template'
df_templates.info() # A pandas DataFrame, 15x4

# C - Read in invites
df_invites = pd.read_csv('./data/Interview_requests.csv') 
df_invites['message'] = [BeautifulSoup( \
            '' if not text else text).get_text() for text in df_invites['message']]
df_invites['type'] = 'invite'
df_invites.info() # A pandas DataFrame, 165x5

# D - Create recruiter_list
recruiter_list = list(df_invites.recruiter_id.unique())

# D - Concatenate the two DataFrames as all_messages
all_messages = pd.concat([df_invites, df_templates], ignore_index=True, sort=True)
all_messages.info() # a pandas DataFrame, 180x5

# Add columns to all_messages
all_messages['frequency'] = 0
all_messages['template'] = None

all_messages.info() # a pandas DataFrame, 180x7

# E - Create countvectors
# Make an instance of the count vectorizer class
CV = CountVectorizer()

# Call a method (also know as an instance function) on the instance
countvectors = CV.fit_transform(all_messages['message'])
print(countvectors)

# CV.get_feature_names()
# countvectors.toarray()

# G - Find used templates
for recruiter_id in recruiter_list:
    usages = {}

    
    template_indexes = all_messages[(all_messages['recruiter_id'] == recruiter_id) & (all_messages['type'] == 'template')].index.tolist()
    
    invite_indexes = all_messages[(all_messages['recruiter_id'] == recruiter_id) & (all_messages['type'] == 'invite')].index.tolist()
    
    matches = cosine_similarity(countvectors[template_indexes], countvectors[invite_indexes])
    
    matches_zipped = dict(zip(template_indexes, matches))
    
    for template_id in matches_zipped:
        scores = matches_zipped[template_id]
        scores_zipped = dict(zip(invite_indexes, scores))
        template_used = []
    
    for invite_id in scores_zipped:
        if scores_zipped[invite_id] > SCORE_THRESHOLD:
            template_used.append(invite_id)
            usages[template_id] = template_used

print(usages)

all_messages.iloc[179]
all_messages.iloc[usages[179]]

# H - Join usages and all_messages together
for result in usages:
    usecase = usages[result]
    all_messages.loc[result, 'frequency'] = len(usecase)
    template_id = all_messages.loc[result, 'id']

    for case in usecase:
        all_messages.loc[case, 'template'] = template_id

final_data = all_messages.copy()

# Export the final_data dataframe


# Perform a chi square on accepted vs template


