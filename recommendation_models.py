# recommendation_models.py
# Multiple recommendation algorithms

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFilter:
    """
    User-based and Item-based Collaborative Filtering
    """
    def __init__(self, method='user'):
        self.method = method
        self.similarity_matrix = None
        
    def fit(self, interaction_matrix):
        """
        Calculate similarity matrix
        """
        print(f"Training {self.method}-based collaborative filter...")
        
        if self.method == 'user':
            self.similarity_matrix = cosine_similarity(interaction_matrix)
        else:  # item-based
            self.similarity_matrix = cosine_similarity(interaction_matrix.T)
        
        return self
    
    def predict(self, user_id, item_id, interaction_matrix, n_neighbors=10):
        """
        Predict rating for user-item pair
        """
        if self.method == 'user':
            user_idx = interaction_matrix.index.get_loc(user_id)
            item_idx = interaction_matrix.columns.get_loc(item_id)
            
            # Find similar users
            user_similarities = self.similarity_matrix[user_idx]
            similar_users = np.argsort(user_similarities)[::-1][1:n_neighbors+1]
            
            # Weighted average of similar users' ratings
            ratings = []
            weights = []
            for similar_user in similar_users:
                rating = interaction_matrix.iloc[similar_user, item_idx]
                if rating > 0:
                    ratings.append(rating)
                    weights.append(user_similarities[similar_user])
            
            if ratings:
                return np.average(ratings, weights=weights)
            return 3.0  # Default rating
            
    def recommend(self, user_id, interaction_matrix, n_recommendations=10):
        """
        Get top-N recommendations for a user
        """
        try:
            user_idx = interaction_matrix.index.get_loc(user_id)
        except:
            # User not found, return random recommendations
            all_items = interaction_matrix.columns.tolist()
            return np.random.choice(all_items, size=min(n_recommendations, len(all_items)), replace=False).tolist()
        
        user_ratings = interaction_matrix.iloc[user_idx]
        
        # Find unrated items
        unrated_items = user_ratings[user_ratings == 0].index
        
        if len(unrated_items) == 0:
            return []
        
        # Predict ratings for unrated items
        predictions = {}
        for item in unrated_items[:100]:  # Limit for performance
            pred = self.predict(user_id, item, interaction_matrix)
            if pred is not None:
                predictions[item] = pred
            else:
                predictions[item] = 3.0  # Default rating
        
        # Sort and return top N
        if predictions:
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            return [item for item, _ in sorted_predictions[:n_recommendations]]
        else:
            # Return random items if no predictions
            return unrated_items[:n_recommendations].tolist()

class ContentBasedFilter:
    """
    Content-based filtering using movie features
    """
    def __init__(self):
        self.movie_features = None
        self.feature_matrix = None
        
    def fit(self, movies_df):
        """
        Prepare content features
        """
        print("Training content-based filter...")
        
        # Select relevant features
        feature_cols = [col for col in movies_df.columns if 
                       col.startswith('genre_') or col in ['visual_style', 'pace', 'emotional_tone']]
        
        self.movie_features = movies_df[['movie_id'] + feature_cols]
        self.feature_matrix = cosine_similarity(movies_df[feature_cols])
        
        return self
    
    def get_similar_movies(self, movie_id, movies_df, n_similar=10):
        """
        Find similar movies based on content
        """
        movie_idx = movies_df[movies_df['movie_id'] == movie_id].index[0]
        similarities = self.feature_matrix[movie_idx]
        
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        similar_movies = movies_df.iloc[similar_indices]['movie_id'].tolist()
        
        return similar_movies
    
    def recommend(self, user_history, movies_df, n_recommendations=10):
        """
        Recommend based on user's viewing history
        """
        # Get user's favorite movies
        top_movies = user_history.nlargest(5, 'rating')['movie_id'].tolist()
        
        recommendations = []
        for movie in top_movies:
            similar = self.get_similar_movies(movie, movies_df, n_similar=5)
            recommendations.extend(similar)
        
        # Remove duplicates and already watched
        recommendations = list(set(recommendations) - set(user_history['movie_id'].tolist()))
        
        return recommendations[:n_recommendations]

class MatrixFactorization:
    """
    Matrix factorization using NMF and SVD
    """
    def __init__(self, method='nmf', n_factors=20):
        self.method = method
        self.n_factors = n_factors
        self.model = None
        self.user_features = None
        self.item_features = None
        
    def fit(self, interaction_matrix):
        """
        Factorize the interaction matrix
        """
        print(f"Training matrix factorization ({self.method})...")
        
        # Fill NaN with 0
        matrix = interaction_matrix.fillna(0)
        
        if self.method == 'nmf':
            self.model = NMF(n_components=self.n_factors, init='random', random_state=42)
            self.user_features = self.model.fit_transform(matrix)
            self.item_features = self.model.components_.T
        else:  # SVD
            self.model = TruncatedSVD(n_components=self.n_factors, random_state=42)
            self.user_features = self.model.fit_transform(matrix)
            self.item_features = self.model.components_.T
        
        return self
    
    def predict(self, user_idx, item_idx):
        """
        Predict rating for user-item pair
        """
        return np.dot(self.user_features[user_idx], self.item_features[item_idx])
    
    def recommend(self, user_id, interaction_matrix, n_recommendations=10):
        """
        Get recommendations for a user
        """
        user_idx = interaction_matrix.index.get_loc(user_id)
        user_vector = self.user_features[user_idx]
        
        # Calculate scores for all items
        scores = np.dot(self.item_features, user_vector)
        
        # Filter out already watched items
        watched_items = interaction_matrix.iloc[user_idx][interaction_matrix.iloc[user_idx] > 0].index
        
        item_scores = pd.Series(scores, index=interaction_matrix.columns)
        item_scores = item_scores[~item_scores.index.isin(watched_items)]
        
        return item_scores.nlargest(n_recommendations).index.tolist()

class HybridRecommender:
    """
    Hybrid recommender combining multiple methods
    """
    def __init__(self):
        self.cf_model = CollaborativeFilter(method='item')
        self.cb_model = ContentBasedFilter()
        self.mf_model = MatrixFactorization(method='svd')
        self.weights = {'cf': 0.4, 'cb': 0.3, 'mf': 0.3}
        
    def fit(self, interaction_matrix, movies_df):
        """
        Train all component models
        """
        print("Training hybrid recommender...")
        
        self.cf_model.fit(interaction_matrix)
        self.cb_model.fit(movies_df)
        self.mf_model.fit(interaction_matrix)
        
        return self
    
    def recommend(self, user_id, interaction_matrix, movies_df, n_recommendations=10):
        """
        Get hybrid recommendations
        """
        rec_scores = {}
        
        # Get recommendations from each model safely
        try:
            cf_recs = self.cf_model.recommend(user_id, interaction_matrix, n_recommendations*2)
            for i, movie in enumerate(cf_recs):
                rec_scores[movie] = rec_scores.get(movie, 0) + self.weights['cf'] * (len(cf_recs) - i)
        except:
            cf_recs = []
        
        try:
            user_history = interaction_matrix.loc[user_id]
            user_history = pd.DataFrame({'movie_id': user_history.index, 'rating': user_history.values})
            user_history = user_history[user_history['rating'] > 0]
            if not user_history.empty:
                cb_recs = self.cb_model.recommend(user_history, movies_df, n_recommendations*2)
                for i, movie in enumerate(cb_recs):
                    rec_scores[movie] = rec_scores.get(movie, 0) + self.weights['cb'] * (len(cb_recs) - i)
        except:
            cb_recs = []
        
        try:
            mf_recs = self.mf_model.recommend(user_id, interaction_matrix, n_recommendations*2)
            for i, movie in enumerate(mf_recs):
                rec_scores[movie] = rec_scores.get(movie, 0) + self.weights['mf'] * (len(mf_recs) - i)
        except:
            mf_recs = []
        
        # If no recommendations from any model, return random movies
        if not rec_scores:
            all_movies = interaction_matrix.columns.tolist()
            watched = interaction_matrix.loc[user_id][interaction_matrix.loc[user_id] > 0].index.tolist()
            unwatched = list(set(all_movies) - set(watched))
            return np.random.choice(unwatched, size=min(n_recommendations, len(unwatched)), replace=False).tolist()
        
        # Sort and return top N
        sorted_recs = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)
        return [movie for movie, _ in sorted_recs[:n_recommendations]]
    
class SessionBasedRecommender:
    """
    Session-based recommendations using sequential patterns
    """
    def __init__(self):
        self.session_patterns = {}
        
    def fit(self, session_sequences):
        """
        Learn session patterns
        """
        print("Training session-based recommender...")
        
        # Build item-to-item transition probabilities
        transitions = {}
        
        for session in session_sequences:
            for i in range(len(session) - 1):
                current_item = session[i]
                next_item = session[i + 1]
                
                if current_item not in transitions:
                    transitions[current_item] = {}
                
                if next_item not in transitions[current_item]:
                    transitions[current_item][next_item] = 0
                    
                transitions[current_item][next_item] += 1
        
        # Normalize to probabilities
        for item in transitions:
            total = sum(transitions[item].values())
            for next_item in transitions[item]:
                transitions[item][next_item] /= total
        
        self.session_patterns = transitions
        return self
    
    def predict_next(self, current_items, n_recommendations=5):
        """
        Predict next items in session
        """
        predictions = {}
        
        for item in current_items[-3:]:  # Use last 3 items
            if item in self.session_patterns:
                for next_item, prob in self.session_patterns[item].items():
                    if next_item not in current_items:
                        predictions[next_item] = predictions.get(next_item, 0) + prob
        
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_predictions[:n_recommendations]]

def evaluate_recommendations(recommendations, test_interactions, k=10):
    """
    Evaluate recommendation quality
    """
    precision_scores = []
    recall_scores = []
    
    for user_id, user_recs in recommendations.items():
        # Get actual items user interacted with in test set
        actual_items = test_interactions[test_interactions['user_id'] == user_id]['movie_id'].tolist()
        
        if actual_items and user_recs:
            # Calculate precision@k and recall@k
            relevant_recommended = set(user_recs[:k]) & set(actual_items)
            
            precision = len(relevant_recommended) / min(k, len(user_recs)) if user_recs else 0
            recall = len(relevant_recommended) / len(actual_items) if actual_items else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
    
    # Calculate averages
    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_recall = np.mean(recall_scores) if recall_scores else 0
    
    return {
        'precision@10': avg_precision,
        'recall@10': avg_recall,
        'f1@k': 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-10)
    }