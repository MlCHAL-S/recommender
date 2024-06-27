import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict, List


class Recommender:
    def __init__(self, ratings_path, movies_path):
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)

    @staticmethod
    def calculate_stats_and_bayesian_average(ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates movie statistics and Bayesian average rating.

        Args:
            ratings_df: DataFrame containing the ratings data

        Returns:
            movie_stats_df: DataFrame containing the movie statistics
        """
        movie_stats_df = ratings_df.groupby('movieId')['rating'].agg(['count', 'mean'])
        confidence = movie_stats_df['count'].mean()
        prior = movie_stats_df['mean'].mean()

        def bayesian_average(x):
            return round((confidence * prior + x.sum()) / (confidence + x.count()), 3)

        bayesian_avg_ratings = ratings_df.groupby('movieId')['rating'].agg(bayesian_average).reset_index()
        bayesian_avg_ratings.columns = ['movieId', 'bayesian_avg']
        movie_stats_df = movie_stats_df.merge(bayesian_avg_ratings, on='movieId')
        return movie_stats_df

    @staticmethod
    def update_ratings_with_bayesian_avg(ratings_df: pd.DataFrame, bayesian_avg_ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Updates the ratings DataFrame with Bayesian average ratings.

        Args:
            ratings_df: DataFrame containing the original ratings data
            bayesian_avg_ratings: DataFrame containing the Bayesian average ratings

        Returns:
            updated_ratings_df: DataFrame containing the updated ratings
        """
        updated_ratings_df = ratings_df.copy()
        updated_ratings_df = updated_ratings_df.merge(bayesian_avg_ratings[['movieId', 'bayesian_avg']], on='movieId')
        updated_ratings_df['rating'] = updated_ratings_df['bayesian_avg']
        updated_ratings_df = updated_ratings_df.drop(columns=['bayesian_avg'])
        return updated_ratings_df

    @staticmethod
    def create_sparse_matrix(df: pd.DataFrame) \
            -> Tuple[csr_matrix, Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
        """
        Generates a sparse matrix from the ratings DataFrame.

        Args:
            df: DataFrame containing userId, movieId, and rating columns

        Returns:
            sparse_matrix: sparse matrix
            user_to_index: dict mapping user IDs to user indices
            index_to_user: dict mapping user indices to user IDs
            movie_to_index: dict mapping movie IDs to movie indices
            index_to_movie: dict mapping movie indices to movie IDs
        """
        num_users = df['userId'].nunique()
        num_movies = df['movieId'].nunique()

        user_to_index = dict(zip(np.unique(df["userId"]), list(range(num_users))))
        movie_to_index = dict(zip(np.unique(df["movieId"]), list(range(num_movies))))

        index_to_user = dict(zip(list(range(num_users)), np.unique(df["userId"])))
        index_to_movie = dict(zip(list(range(num_movies)), np.unique(df["movieId"])))

        user_indices = [user_to_index[i] for i in df['userId']]
        movie_indices = [movie_to_index[i] for i in df['movieId']]

        sparse_matrix = csr_matrix((df["rating"], (user_indices, movie_indices)), shape=(num_users, num_movies))

        return sparse_matrix, user_to_index, movie_to_index, index_to_user, index_to_movie

    @staticmethod
    def find_similar_movies(movie_id: int, sparse_matrix: csr_matrix, movie_to_index: Dict[int, int],
                            index_to_movie: Dict[int, int], k: int, metric: str = 'cosine') \
            -> List[int]:
        """
        Finds k-nearest neighbours for a given movie ID.

        Args:
            movie_id: ID of the movie of interest
            sparse_matrix: user-item utility matrix
            movie_to_index: dict mapping movie IDs to movie indices
            index_to_movie: dict mapping movie indices to movie IDs
            k: number of similar movies to retrieve
            metric: distance metric for kNN calculations

        Returns:
            similar_movie_ids: list of k similar movie IDs
        """
        sparse_matrix = sparse_matrix.T
        neighbour_ids = []

        movie_index = movie_to_index[movie_id]
        movie_vector = sparse_matrix[movie_index]
        if isinstance(movie_vector, np.ndarray):
            movie_vector = movie_vector.reshape(1, -1)

        knn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric=metric)
        knn.fit(sparse_matrix)
        neighbours = knn.kneighbors(movie_vector, return_distance=False)
        for i in range(1, k + 1):  # Start from 1 to skip the movie itself
            neighbour_index = neighbours.item(i)
            neighbour_ids.append(index_to_movie[neighbour_index])

        return neighbour_ids

    def recommend(self, movie_id: int) -> List[str]:
        """
        Recommends similar movies based on the provided movie ID.

        Args:
            movie_id: The ID of the movie for which to find similar movies.

        Returns:
            recommendations: List of recommended movie titles.
        """
        movie_stats = self.calculate_stats_and_bayesian_average(self.ratings)
        movie_stats = movie_stats.merge(self.movies[['movieId', 'title']])
        movie_stats = movie_stats.sort_values('bayesian_avg', ascending=False)

        bayesian_avg_ratings = movie_stats[['movieId', 'bayesian_avg']]
        updated_ratings = self.update_ratings_with_bayesian_avg(self.ratings, bayesian_avg_ratings)

        sparse_matrix, user_to_index, movie_to_index, index_to_user, index_to_movie = self.create_sparse_matrix(
            updated_ratings)

        movie_titles = dict(zip(self.movies['movieId'], self.movies['title']))

        similar_movies = self.find_similar_movies(movie_id, sparse_matrix, movie_to_index, index_to_movie,
                                                  metric='cosine', k=10)

        recommendations = []
        for similar_movie_id in similar_movies:
            recommendations.append(movie_titles[similar_movie_id])

        return recommendations
