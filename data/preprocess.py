import pandas as pd
import os

def load_data(data_dir):
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    movies_path = os.path.join(data_dir, 'movies.csv')
    
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    return ratings, movies

def preprocess_data(ratings, movies):
    df = pd.merge(ratings, movies, on='movieId')
    return df

if __name__ == "__main__":
    DATA_DIR = "data/raw/ml-latest-small"
    if not os.path.exists(DATA_DIR):
        print(f"data dir: {DATA_DIR} not found. Run download.py first.")
    else:
        ratings, movies = load_data(DATA_DIR)
        print(f"loaded {len(ratings)} ratings and {len(movies)} movies.")
        df = preprocess_data(ratings, movies)
        print(df.head())
