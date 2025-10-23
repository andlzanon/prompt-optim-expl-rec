import pandas as pd

# Ratings
ratings = pd.read_csv(
    'ml-1m_dat/ratings.dat',
    sep='::', engine='python',
    names=['user_id', 'movie_id', 'rating', 'timestamp'],
    encoding='latin-1'
)
ratings.to_csv('ml-1m_csv/ratings.csv', index=False)

# Movies
movies = pd.read_csv(
    'ml-1m/movies.dat',
    sep='::', engine='python',
    names=['movie_id', 'title', 'genres'],
    encoding='latin-1'
)
movies.to_csv('ml-1m_csv/movies.csv', index=False)

# Users
users = pd.read_csv(
    'ml-1m/users.dat',
    sep='::', engine='python',
    names=['user_id', 'gender', 'age', 'occupation', 'zip'],
    encoding='latin-1'
)
users.to_csv('ml-1m_csv/users.csv', index=False)

print("✅ All MovieLens .dat files converted to CSV successfully!")
