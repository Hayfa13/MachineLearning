import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load the dataset from a CSV file
df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\spam (or) ham.csv")

# Display the first few rows of the DataFrame
print(df.head())

# Display statistics for the 'Category' column grouped by 'spam' and 'ham'
print(df.groupby('Class').describe())

# Create a new 'spam' column with 1 for 'spam' and 0 for 'ham'
df['spam'] = df['Class'].apply(lambda x: 1 if x == 'spam' else 0)

# Display the DataFrame with the new 'spam' column
print(df.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.sms, df.spam, test_size=0.25)

# Create a CountVectorizer to convert text data into numerical form
v = CountVectorizer()

# Fit and transform the training data to obtain the term-document matrix
X_train_count = v.fit_transform(X_train.values)

# Fit a Multinomial Naive Bayes classifier on the term-document matrix
reg = MultinomialNB()
reg.fit(X_train_count, y_train)

# Sample emails for prediction
emails = [
    'Hey mohan, can we get together to watch a football game tomorrow?',
    'Up to 20% discount on parking, an exclusive offer just for you. Dont miss this reward!'
]

# Transform the sample emails using the CountVectorizer
emails_count = v.transform(emails)

# Predict whether the emails are spam or not
print(reg.predict(emails_count))

# Evaluate the classifier on the test data and calculate the accuracy
X_test_count = v.transform(X_test)
print("The score is", reg.score(X_test_count, y_test))

# Create a pipeline with CountVectorizer and MultinomialNB
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

# Fit the pipeline on the training data
clf.fit(X_train, y_train)

# Calculate and print the accuracy on the test data using the pipeline
print(clf.score(X_test, y_test))

# Predict whether the sample emails are spam or not using the pipeline
print(clf.predict(emails))
