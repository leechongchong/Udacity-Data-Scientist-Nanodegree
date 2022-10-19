# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# load and merge datasets
def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges datasets from 2 filepaths.
    
    Parameters:
    messages_filepath: messages csv file
    categories_filepath: categories csv file
    
    Returns:
    df: dataframe containing messages_filepath and categories_filepath merged
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    return df


def clean_data(df):
     """
    Cleans the dataframe.
    
    Parameters:
    df: DataFrame
    
    Returns:
    df: Cleaned DataFrame
    
    """
    # Split categories into separate category columns 
    categories = df.categories.str.split(';',expand=True)     
    row = categories.iloc[0]  
    category_colnames = [i[:-2] for i in row]
    categories.columns = category_colnames  
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    categories['related'].replace(2, 1, inplace=True) 
    
    # Replace categories column in df with new category columns
    df.drop('categories', axis = 1, inplace=True)
    
    # Remove duplicates
    df = pd.concat([df, categories], axis = 1)
    df.drop_duplicates(inplace=True)
    return df

# Save the clean dataset into an sqlite database
def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterResponse', engine, index=False)


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
