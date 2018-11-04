import pandas as pd 
import numpy as np
import datetime, re, pickle
import os.path


def _read_movielens_old(directory):
    # directory = os.path.dirname(os.path.realpath("__file__")) + '/' + directory
    ratings_df = pd.read_csv(directory, sep='::', header=None, engine='python')
    ratings_df.columns = ['User','Movie','Rating','Timestamp']
    movies_df = pd.read_csv(directory.replace('ratings.dat','movies.dat'), sep='::', header=None, engine='python')
    movies_df.columns = ['MovieID','MovieName','MovieGenre']
    return (ratings_df, movies_df)

def _preprocess_dataframe_1M(ratings_df, movies_df, all_movies, min_positive_score):    
    # Converting Timestamps to datetime format
    ratings_df['Timestamp'] = ratings_df['Timestamp'].apply(lambda x: datetime.datetime(1970,1,1,0,0,0) + datetime.timedelta(seconds=x))
    
    # Filter Ratings - Preprocessing 1
    downloads_df = pd.read_csv(all_movies, header=None, delimiter=r'\s+')
    downloads_df = downloads_df[downloads_df[3].str.contains('.mp4')]
    downloads = list(downloads_df[3].apply(lambda x: int(x.rsplit('_',1)[1].split('.')[0])))
    filtered_movies = list(set(ratings_df.Movie.unique()) - (set(ratings_df.Movie.unique()) - set(downloads)))
    print(ratings_df.shape)
    # Filter Ratings that has no corresponding trailer in database
    ratings_df = ratings_df[ratings_df.Movie.isin(filtered_movies)]
    print(ratings_df.shape)
    # Filter Ratings that has higher than min_positive_score (default is 3.0)
    ratings_df = ratings_df[ratings_df.Rating>=min_positive_score]
    print(ratings_df.shape)

    # Train Test Split - Preprocessing 2
    downloads_df['year'] = downloads_df[3].map(
        lambda x: re.search(r"\((200[0|1|2|3])\)", x)).dropna().map(
        lambda x: x[1])
    downloads_df['movieID'] = downloads_df[3].apply(lambda x: int(x.rsplit('_',1)[1].split('.')[0]))
    downloads_df = downloads_df.dropna()
    test_ratings_df = pd.merge(ratings_df, downloads_df[['movieID','year']], how='left', left_on='Movie', right_on='movieID').dropna()
    assert len(test_ratings_df.year.unique()) == 1
    training_ratings_df = ratings_df.merge(test_ratings_df[['User','Movie','Rating','Timestamp']], 
                                           indicator=True, how='outer')
    training_ratings_df = training_ratings_df[training_ratings_df['_merge'] == 'left_only'][['User','Movie','Rating','Timestamp']]
    assert (test_ratings_df.shape[0]+training_ratings_df.shape[0] == ratings_df.shape[0]) # Check split is valid

    return (training_ratings_df, test_ratings_df)


def _preprocess_dataframe_10M(ratings_df, movies_df, all_movies, min_positive_score):    
    # Converting Timestamps to datetime format
    ratings_df['Timestamp'] = ratings_df['Timestamp'].apply(lambda x: datetime.datetime(1970,1,1,0,0,0) + datetime.timedelta(seconds=x))
    
    # Filter Ratings - Preprocessing 1
    downloads_df = pd.read_csv(all_movies, header=None, delimiter=r'\s+')
    downloads_df = downloads_df[downloads_df[3].str.contains('.mp4')]
    downloads = list(downloads_df[3].apply(lambda x: int(x.rsplit('_',1)[1].split('.')[0])))
    filtered_movies = list(set(ratings_df.Movie.unique()) - (set(ratings_df.Movie.unique()) - set(downloads)))
    print(ratings_df.shape)
    # Filter Ratings that has no corresponding trailer in database
    ratings_df = ratings_df[ratings_df.Movie.isin(filtered_movies)]
    print(ratings_df.shape)
    # Filter Ratings that has higher than min_positive_score (default is 3.0)
    ratings_df = ratings_df[ratings_df.Rating>=min_positive_score]
    print(ratings_df.shape)

    # Train Test Split - Preprocessing 2
    downloads_df['year'] = downloads_df[3].map(
        lambda x: re.search(r"\((200[7|8|9])\)", x)).dropna().map(
        lambda x: x[1])
    downloads_df['movieID'] = downloads_df[3].apply(lambda x: int(x.rsplit('_',1)[1].split('.')[0]))
    downloads_df = downloads_df.dropna()
    test_ratings_df = pd.merge(ratings_df, downloads_df[['movieID','year']], how='left', left_on='Movie', right_on='movieID').dropna()
    assert len(test_ratings_df.year.unique()) == 2
    training_ratings_df = ratings_df.merge(test_ratings_df[['User','Movie','Rating','Timestamp']], 
                                           indicator=True, how='outer')
    training_ratings_df = training_ratings_df[training_ratings_df['_merge'] == 'left_only'][['User','Movie','Rating','Timestamp']]
    training_ratings_df = training_ratings_df[training_ratings_df.Timestamp < datetime.datetime(2007,1,1)] # Removing all data in 2007/8/9 in training list
    assert (test_ratings_df.shape[0]+training_ratings_df.shape[0] < ratings_df.shape[0]) # Check split is valid

    return (training_ratings_df, test_ratings_df)


def get_movielens_1M(directory='ml-1m/ratings.dat', all_movies_dir = 'all_movies.txt', pickles_dir='pickles/', min_positive_score=4.):
    # print(os.path.dirname(os.path.realpath("__file__")))
    if os.path.exists(pickles_dir+str(min_positive_score)+'MovieLens1M.pickle'):
        with open(pickles_dir+str(min_positive_score)+'MovieLens1M.pickle', 'rb') as handle:
            data = pickle.load(handle)

    else:
        (ratings, movies) = _read_movielens_old(directory)
        (training_ratings_df, test_ratings_df) = _preprocess_dataframe_1M(ratings, movies, all_movies_dir, min_positive_score)

        data = {'training':training_ratings_df,
                'test': test_ratings_df[['User', 'Movie', 'Rating', 'Timestamp']],
                'Titles': movies}
        with open(pickles_dir+str(min_positive_score)+'MovieLens1M.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    return data



def get_movielens_10M(directory='ml-10m/ratings.dat', all_movies_dir = 'all_movies.txt', pickles_dir='pickles/', min_positive_score=4.):
    if os.path.exists(pickles_dir+str(min_positive_score)+'MovieLens10M.pickle'):
        with open(pickles_dir+str(min_positive_score)+'MovieLens10M.pickle', 'rb') as handle:
            data = pickle.load(handle)

    else:
        (ratings, movies) = _read_movielens_old(directory)
        (training_ratings_df, test_ratings_df) = _preprocess_dataframe_10M(ratings, movies, all_movies_dir, min_positive_score)

        data = {'training':training_ratings_df,
                'test': test_ratings_df[['User', 'Movie', 'Rating', 'Timestamp']],
                'Titles': movies}
        with open(pickles_dir+str(min_positive_score)+'MovieLens1M.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    return data


def _read_movielens_new(directory):
    ratings_df = pd.read_csv(directory, sep=',', header=None)
    ratings_df.columns = ['User','Movie','Rating','Timestamp']
    movies_df = pd.read_csv(directory.replace('ratings.csv','movies.csv'), sep=',', header=None)
    movies_df.columns = ['MovieID','MovieName','MovieGenre']
    return (ratings_df, movies_df)


def get_movielens_20M(directory='ml-20m/ratings.dat', all_movies_dir = 'all_movies.txt', pickles_dir='pickles/', min_positive_score=4.):
    if os.path.exists(pickles_dir+str(min_positive_score)+'MovieLens20M.pickle'):
        with open(pickles_dir+str(min_positive_score)+'MovieLens20M.pickle', 'rb') as handle:
            data = pickle.load(handle)

    else:
        (ratings, movies) = _read_movielens_new(directory)
        (training_ratings_df, test_ratings_df) = _preprocess_dataframe_10M(ratings, movies, all_movies_dir, min_positive_score)

        data = {'training':training_ratings_df,
                'test': test_ratings_df[['User', 'Movie', 'Rating', 'Timestamp']],
                'Titles': movies}
        with open(pickles_dir+str(min_positive_score)+'MovieLens20M.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    return data











