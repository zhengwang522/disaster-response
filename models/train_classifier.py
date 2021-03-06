import sys
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle


def load_data(database_filepath):
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df['message'].values
    Y = df.iloc[:, 4:].values
    category_names = df.columns[4:]
    
    return X, Y, category_names


def tokenize(text):
    
    #replace punctuation with blanks and tokenize the text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = text.split()
    
    # convert words to their roots if any
    lemmed = [WordNetLemmatizer().lemmatize(tok) for tok in tokens]

    return lemmed

#class StartingVerbExtractor(BaseEstimator, TransformerMixin):
#
#    def starting_verb(self, text):
#        sentence_list = nltk.sent_tokenize(text)
#        for sentence in sentence_list:
#            pos_tags = nltk.pos_tag(tokenize(sentence))
#            first_word, first_tag = pos_tags[0]
#            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
#                return True
#        return False
#
#    def fit(self, x, y=None):
#        return self
#
#    def transform(self, X):
#        X_tagged = pd.Series(X).apply(self.starting_verb)
#        return pd.DataFrame(X_tagged)
    
    
def build_model():
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])) #,

            #('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

   
    parameters = {
        #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        #'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        #'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3],
        #'features__transformer_weights': (
        #    {'text_pipeline': 1, 'starting_verb': 0.5},
        #    {'text_pipeline': 0.5, 'starting_verb': 1},
        #    {'text_pipeline': 0.8, 'starting_verb': 1},
        #)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for j in range(Y_test.shape[1]):
        print(classification_report(Y_test[:,j], Y_pred[:,j], labels=np.unique(Y_pred[:,j])))
    

def save_model(model, model_filepath):

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