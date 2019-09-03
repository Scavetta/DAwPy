from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = ['first document',
          'second document.',
          'third text',
          'first text',
         ]

# Make an instance of the count vectorizer class
vectorizer = CountVectorizer()

# Call a method (also know as an instance function) on the instance
X = vectorizer.fit_transform(corpus)
print(X)

vectorizer.get_feature_names()
X.toarray()

cosine_similarity(X)
