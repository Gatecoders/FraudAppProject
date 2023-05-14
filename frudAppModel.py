import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('./training.csv')

# Define a function to preprocess the data
def preprocess_data(df):
    # Remove package name as it's not relevant
    df = df.drop('package_name', axis=1)
    
    # Convert text to lowercase
    df['review'] = df['review'].str.strip().str.lower()
    
    return df

# Preprocess the data
df = preprocess_data(df)

# Split into training and testing data
x = df['review']
y = df['polarity']
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)

# Vectorize text reviews to numbers
vectorizer = CountVectorizer(stop_words='english')
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# Train a binary classifier to predict positive or not
binary_classifier = MultinomialNB()
binary_classifier.fit(x_train, y_train)
y_pred = binary_classifier.predict(x_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save the vectorizer and the model
joblib.dump(binary_classifier, './model.pkl')
joblib.dump(vectorizer, './vectorizer.pkl')
