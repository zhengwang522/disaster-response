import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # read datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets using the common id
    df = pd.merge(messages, categories, on='id',how='inner')
    return df

def clean_data(df):
    # example record in categories column: related-1;request-0;offer-0;aid_related-0;
    # split column values by ";" so that each value becomes a separate column
    # use names before the dash as column names
    # clean up the column values as only numbers remained
    categories = df['categories'].str.split(';', expand=True)
    row1 = categories.iloc[0, :]
    category_colnames = row1.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for col in categories:
        categories[col] = categories[col].str[-1]
        categories[col] = categories[col].astype(int)
    # drop original column from df and combine new features into it
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # remove duplicate records
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    engine = create_engine(database_filename)
    df.to_sql('messages', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()