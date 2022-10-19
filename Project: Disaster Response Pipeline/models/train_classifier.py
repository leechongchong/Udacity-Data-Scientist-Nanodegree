# Import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])

from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pickle

# load data
def load_data(database_filepath):
    """
    Loads data from SQLite database.
    
    Parameters:
    database_filepath: filepath to the database
    
    Returns:
    X: features
    y: target
    category_names: categorical name for labeling
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('DisasterResponse', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns
    return X,y,category_names


# tokenizing the text data
def tokenize(text):
     '''
    Paraneters: messages for tokenization.
    Return:
        clean_tokens: output upon tokenization.
    '''
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

# build a machine learning pipeline
def build_model():
    """
    building message classifierand tuning model using GridSearchCV.
    
    Returns:
    cv: Classifier 
    """  
        
    # set up pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
    # fine tune parameters 
    parameters = {
    'clf__estimator__n_estimators' : [5, 10]
    }
    
    cv = GridSearchCV(pipeline, parameters, verbose=3)
    
    return cv

# Evaluate the model: show the accuracy, precision, and recall of the tuned model.
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates model performance and generate corresponding report. 
    
    Parameters:
    model: classifier
    X_test: test messages
    Y_test: categories/labels for test messages
    
    Returns:
    classification report for each column
    """
    
    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))

# save and export the model as a pickle file
def save_model(model, model_filepath):
    """ Exports the final model as a pickle file."""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
