# feature_engineering.py
# Advanced feature engineering for recommendation system

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import sparse

def create_user_features(users_df, interactions_df):
    """
    Create comprehensive user features
    """
    print("Creating user features...")
    
    # Aggregate viewing statistics
    user_stats = interactions_df.groupby('user_id').agg({
        'rating': ['mean', 'std', 'count'],
        'completion_rate': ['mean', 'std'],
        'watch_duration_minutes': ['sum', 'mean'],
        'from_recommendation': 'mean'
    }).round(3)
    
    user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
    
    # Merge with user profiles
    users_enhanced = users_df.merge(user_stats, left_on='user_id', right_index=True, how='left')
    
    # Time-based features
    time_features = interactions_df.groupby(['user_id', 'time_of_day']).size().unstack(fill_value=0)
    time_features = time_features.div(time_features.sum(axis=1), axis=0)
    time_features.columns = [f'pref_time_{col}' for col in time_features.columns]
    
    users_enhanced = users_enhanced.merge(time_features, left_on='user_id', right_index=True, how='left')
    
    # Device preferences
    device_features = interactions_df.groupby(['user_id', 'device']).size().unstack(fill_value=0)
    device_features = device_features.div(device_features.sum(axis=1) + 1, axis=0)
    device_features.columns = [f'device_{col}' for col in device_features.columns]
    
    users_enhanced = users_enhanced.merge(device_features, left_on='user_id', right_index=True, how='left')
    
    return users_enhanced

def create_movie_features(movies_df, interactions_df):
    """
    Create enhanced movie features
    """
    print("Creating movie features...")
    
    # Aggregate movie statistics
    movie_stats = interactions_df.groupby('movie_id').agg({
        'rating': ['mean', 'count'],
        'completion_rate': 'mean',
        'watch_duration_minutes': 'mean'
    }).round(3)
    
    movie_stats.columns = ['_'.join(col).strip() for col in movie_stats.columns]
    movie_stats.rename(columns={'rating_mean': 'avg_user_rating', 
                                'rating_count': 'total_views'}, inplace=True)
    
    # Merge with movie catalog
    movies_enhanced = movies_df.merge(movie_stats, left_on='movie_id', right_index=True, how='left')
    
    # Genre encoding
    genre_dummies = movies_enhanced['genres'].str.get_dummies(sep='|')
    genre_dummies.columns = [f'genre_{col}' for col in genre_dummies.columns]
    
    movies_enhanced = pd.concat([movies_enhanced, genre_dummies], axis=1)
    
    # Popularity tiers
    movies_enhanced['popularity_tier'] = pd.qcut(movies_enhanced['popularity_score'].fillna(0), 
                                                  q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Content quality score
    movies_enhanced['quality_score'] = (
        movies_enhanced['imdb_rating'] * 0.4 +
        movies_enhanced['avg_user_rating'].fillna(5) * 0.4 +
        movies_enhanced['completion_rate_mean'].fillna(0.5) * 10 * 0.2
    )
    
    return movies_enhanced

def create_interaction_matrix(interactions_df):
    """
    Create user-item interaction matrices
    """
    print("Creating interaction matrices...")
    
    # Create rating matrix
    rating_matrix = interactions_df.pivot_table(
        index='user_id',
        columns='movie_id',
        values='rating',
        fill_value=0
    )
    
    # Create implicit feedback matrix (watched or not)
    implicit_matrix = interactions_df.pivot_table(
        index='user_id',
        columns='movie_id',
        values='completion_rate',
        aggfunc='max',
        fill_value=0
    )
    implicit_matrix = (implicit_matrix > 0.3).astype(int)  # Watched at least 30%
    
    # Create weighted interaction matrix
    interactions_df['interaction_strength'] = (
        interactions_df['rating'].fillna(3) / 5 * 0.5 +
        interactions_df['completion_rate'] * 0.5
    )
    
    weighted_matrix = interactions_df.pivot_table(
        index='user_id',
        columns='movie_id',
        values='interaction_strength',
        fill_value=0
    )
    
    return rating_matrix, implicit_matrix, weighted_matrix

def create_session_features(sessions_df):
    """
    Extract session-based patterns
    """
    print("Creating session features...")
    
    # Session statistics
    session_stats = sessions_df.groupby('session_id').agg({
        'movie_id': 'count',
        'duration_seconds': ['sum', 'mean'],
        'action': lambda x: (x == 'add_to_list').sum()
    })
    
    session_stats.columns = ['movies_browsed', 'total_duration', 'avg_duration', 'items_added']
    
    # Sequential patterns (simplified)
    session_sequences = sessions_df.sort_values(['session_id', 'timestamp']).groupby('session_id')['movie_id'].apply(list)
    
    return session_stats, session_sequences

def create_graph_features(interactions_df, movies_df):
    """
    Create graph-based features for GNN
    """
    print("Creating graph features...")
    
    # User-Movie bipartite graph edges
    edges = interactions_df[interactions_df['rating'].notna()][['user_id', 'movie_id', 'rating']]
    
    # Movie-Movie similarity graph (based on shared viewers)
    movie_viewers = interactions_df.groupby('movie_id')['user_id'].apply(set)
    
    movie_similarity = []
    movies_list = movie_viewers.index.tolist()
    
    for i, movie1 in enumerate(movies_list[:100]):  # Limit for performance
        for movie2 in movies_list[i+1:min(i+101, len(movies_list))]:
            shared_viewers = len(movie_viewers[movie1] & movie_viewers[movie2])
            if shared_viewers > 5:  # Threshold
                similarity = shared_viewers / min(len(movie_viewers[movie1]), len(movie_viewers[movie2]))
                movie_similarity.append({
                    'movie1': movie1,
                    'movie2': movie2,
                    'similarity': similarity
                })
    
    movie_similarity_df = pd.DataFrame(movie_similarity)
    
    return edges, movie_similarity_df

def engineer_all_features(users_df, movies_df, interactions_df, sessions_df):
    """
    Apply all feature engineering
    """
    # Create enhanced features
    users_enhanced = create_user_features(users_df, interactions_df)
    movies_enhanced = create_movie_features(movies_df, interactions_df)
    
    # Create interaction matrices
    rating_matrix, implicit_matrix, weighted_matrix = create_interaction_matrix(interactions_df)
    
    # Create session features
    session_stats, session_sequences = create_session_features(sessions_df)
    
    # Create graph features
    graph_edges, movie_similarity = create_graph_features(interactions_df, movies_df)
    
    print(f"Created {len(users_enhanced.columns)} user features")
    print(f"Created {len(movies_enhanced.columns)} movie features")
    print(f"Interaction matrix shape: {rating_matrix.shape}")
    
    return {
        'users': users_enhanced,
        'movies': movies_enhanced,
        'rating_matrix': rating_matrix,
        'implicit_matrix': implicit_matrix,
        'weighted_matrix': weighted_matrix,
        'session_stats': session_stats,
        'session_sequences': session_sequences,
        'graph_edges': graph_edges,
        'movie_similarity': movie_similarity
    }