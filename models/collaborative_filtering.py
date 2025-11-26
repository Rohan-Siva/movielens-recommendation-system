import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

class KNNRecommender:
    def __init__(self, k=5, metric='cosine'):
        self.k = k
        self.metric = metric
        self.model = NearestNeighbors(n_neighbors=k, metric=metric, algorithm='brute')
        self.user_item_matrix = None
        self.user_map = None
        self.item_map = None
        self.reverse_item_map = None

    def fit(self, ratings_df):
        self.user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.user_map = {i: u for i, u in enumerate(self.user_item_matrix.index)}
        self.item_map = {m: i for i, m in enumerate(self.user_item_matrix.columns)}
        self.reverse_item_map = {i: m for m, i in self.item_map.items()}
        
        self.sparse_matrix = csr_matrix(self.user_item_matrix.values)
        
        self.model.fit(self.sparse_matrix)

    def recommend(self, user_id, n_recommendations=10):
        if user_id not in self.user_item_matrix.index:
            return []
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        distances, indices = self.model.kneighbors(self.user_item_matrix.iloc[user_idx, :].values.reshape(1, -1), n_neighbors=self.k+1)
        
        similar_users_indices = indices.flatten()[1:] 
        
        similar_users_ratings = self.user_item_matrix.iloc[similar_users_indices]
        mean_ratings = similar_users_ratings.mean(axis=0)
        
        user_rated_items = self.user_item_matrix.loc[user_id]
        mean_ratings[user_rated_items > 0] = 0
        
        top_items_indices = mean_ratings.argsort()[::-1][:n_recommendations]
        top_items = [self.reverse_item_map[i] for i in top_items_indices]
        
        return top_items

if __name__ == "__main__":
    import os
    data_dir = "data/raw/ml-latest-small"
    if os.path.exists(data_dir):
        ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
        model = KNNRecommender(k=20)
        print("Fitting model...")
        model.fit(ratings)
        print("Model fitted.")
        
        test_user = ratings['userId'].iloc[0]
        recs = model.recommend(test_user)
        print(f"Recommendations for user {test_user}: {recs}")
