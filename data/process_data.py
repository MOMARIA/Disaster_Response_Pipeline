import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads a DataFrame by merging message and category data from specified file paths.
    
    Parameters:
    - messages_file_path: String representing the file path of the messages data.
    - categories_file_path: String representing the file path of the categories data.
    
    Returns:
    - DataFrame: A Pandas DataFrame containing the merged dataset from the provided messages and categories file paths.
    """
    # message dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df

    

def clean_data(df):
    """
    Cleans the input Pandas DataFrame.
    
    Parameters:
    - df: A Pandas DataFrame that needs to be cleaned.
    
    Returns:
    - DataFrame: A cleaned Pandas DataFrame.
    """
    categories = df["categories"].str.split(";", expand=True)
    # Select the first row of the categories dataframe
    row = categories.iloc[0,:]
    category_colnames = [r.split('-')[0] for r in row]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df.drop(columns=["categories"], inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    df['related'] = df['related'].map({0:0, 1:1, 2:1})
    return df


def save_data(df, database_filename):
    """
    Saves the cleaned data to a specified database file.
    
    Parameters:
    - df: A Pandas DataFrame containing the cleaned data for messages and categories.
    - filename: String representing the name of the output database file.
    
    Returns:
    - None
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('message', engine, index = False, if_exists='replace')


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
        print("Provide the file paths for the messages and categories datasets as the first and second arguments, respectively. Also, specify the database file path where you'd like to save the cleaned data as the third argument.\n\nUsage Example: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db")
        
if __name__ == '__main__':
    main()
