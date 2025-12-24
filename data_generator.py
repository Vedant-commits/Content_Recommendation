# data_generator.py
# Generate realistic user interaction data for movie recommendation system

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)

def generate_movie_catalog(n_movies=2000):
    """
    Generate movie/show catalog with rich metadata
    """
    print("Generating movie catalog...")
    
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Romance', 
              'Thriller', 'Documentary', 'Animation', 'Fantasy']
    
    # Generate random names for directors and actors
    first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emma', 'Chris', 'Lisa', 
                   'Robert', 'Maria', 'James', 'Patricia', 'Richard', 'Jennifer', 'William',
                   'Daniel', 'Nancy', 'Paul', 'Karen', 'Mark', 'Betty', 'Donald', 'Helen']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 
                  'Davis', 'Rodriguez', 'Martinez', 'Wilson', 'Anderson', 'Taylor', 'Thomas',
                  'Hernandez', 'Moore', 'Martin', 'Jackson', 'Thompson', 'White', 'Lopez']
    
    directors = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(100)]
    actors = [f"{random.choice(first_names)} {random.choice(last_names)} {i}" for i in range(300)]
    
    movies = []
    
    # Title components for generating movie names
    title_prefixes = ['The', 'A', '']
    title_adjectives = ['Amazing', 'Incredible', 'Dark', 'Bright', 'Last', 'First', 
                       'Hidden', 'Lost', 'Forbidden', 'Secret', 'Ultimate', 'Final',
                       'Eternal', 'Golden', 'Silver', 'Mystic', 'Ancient', 'Modern']
    title_nouns = ['Story', 'Journey', 'Adventure', 'Mystery', 'Legacy', 'Chronicles',
                   'Knight', 'Dragon', 'Storm', 'Dream', 'Shadow', 'Light', 'Quest',
                   'Warrior', 'Kingdom', 'Empire', 'Legend', 'Prophecy', 'Destiny']
    
    for i in range(n_movies):
        movie_id = f'MOV_{str(i).zfill(4)}'
        
        # Determine content type
        content_type = np.random.choice(['Movie', 'TV Show'], p=[0.7, 0.3])
        
        # Generate title
        prefix = np.random.choice(title_prefixes)
        adj = np.random.choice(title_adjectives)
        noun = np.random.choice(title_nouns)
        title = f"{prefix} {adj} {noun}".strip()
        # Add number to make titles unique
        if i % 10 == 0:
            title += f" {i//10 + 1}"
        
        year = np.random.randint(1990, 2024)
        
        # Genres (can have multiple)
        n_genres = np.random.randint(1, 4)
        movie_genres = np.random.choice(genres, n_genres, replace=False)
        
        # Cast and crew
        director = np.random.choice(directors)
        n_actors = np.random.randint(3, 8)
        movie_actors = np.random.choice(actors, n_actors, replace=False)
        
        # Quality metrics
        imdb_rating = np.random.beta(7, 3) * 10  # Skewed towards higher ratings
        popularity_score = np.random.exponential(50)
        
        # Content features
        runtime = np.random.normal(120, 30) if content_type == 'Movie' else np.random.randint(6, 10) * 10
        runtime = max(60, min(200, runtime))
        
        # Language and country
        languages = ['English', 'Spanish', 'French', 'Korean', 'Japanese', 'Hindi']
        language = np.random.choice(languages, p=[0.5, 0.15, 0.1, 0.1, 0.1, 0.05])
        
        # Viewing features
        maturity_rating = np.random.choice(['G', 'PG', 'PG-13', 'R'], p=[0.1, 0.2, 0.5, 0.2])
        
        # Embeddings features (simulated content features)
        visual_style = np.random.random()  # 0=dark, 1=bright
        pace = np.random.random()  # 0=slow, 1=fast
        emotional_tone = np.random.random()  # 0=serious, 1=lighthearted
        
        movies.append({
            'movie_id': movie_id,
            'title': title[:50],  # Truncate long titles
            'content_type': content_type,
            'year': year,
            'genres': '|'.join(movie_genres),
            'director': director,
            'cast': '|'.join(movie_actors[:5]),  # Top 5 cast
            'imdb_rating': round(imdb_rating, 1),
            'popularity_score': round(popularity_score, 2),
            'runtime_minutes': int(runtime),
            'language': language,
            'maturity_rating': maturity_rating,
            'visual_style': round(visual_style, 3),
            'pace': round(pace, 3),
            'emotional_tone': round(emotional_tone, 3)
        })
    
    return pd.DataFrame(movies)

def generate_user_profiles(n_users=5000):
    """
    Generate user profiles with preferences
    """
    print("Generating user profiles...")
    
    users = []
    
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55+']
    subscription_tiers = ['Basic', 'Standard', 'Premium']
    
    for i in range(n_users):
        user_id = f'USER_{str(i).zfill(5)}'
        
        # Demographics
        age_group = np.random.choice(age_groups)
        subscription = np.random.choice(subscription_tiers, p=[0.3, 0.5, 0.2])
        
        # Viewing preferences based on age
        if age_group == '18-24':
            genre_preferences = {'Action': 0.3, 'Comedy': 0.25, 'Horror': 0.2}
            avg_viewing_hours = np.random.normal(3, 1)
        elif age_group == '25-34':
            genre_preferences = {'Drama': 0.3, 'Thriller': 0.25, 'Comedy': 0.2}
            avg_viewing_hours = np.random.normal(2.5, 0.8)
        elif age_group == '35-44':
            genre_preferences = {'Drama': 0.35, 'Documentary': 0.2, 'Comedy': 0.15}
            avg_viewing_hours = np.random.normal(2, 0.7)
        elif age_group == '45-54':
            genre_preferences = {'Drama': 0.3, 'Documentary': 0.25, 'Romance': 0.15}
            avg_viewing_hours = np.random.normal(2, 0.6)
        else:  # 55+
            genre_preferences = {'Documentary': 0.3, 'Drama': 0.25, 'Romance': 0.2}
            avg_viewing_hours = np.random.normal(3, 1)
        
        # Account details
        account_age_days = np.random.randint(30, 1825)  # 1 month to 5 years
        devices_used = np.random.choice(['TV', 'Mobile', 'Tablet', 'Computer'], 
                                       np.random.randint(1, 4), replace=False)
        
        # Viewing behavior
        binge_watcher = np.random.choice([0, 1], p=[0.6, 0.4])
        weekend_viewer = np.random.choice([0, 1], p=[0.4, 0.6])
        
        users.append({
            'user_id': user_id,
            'age_group': age_group,
            'subscription_tier': subscription,
            'account_age_days': account_age_days,
            'avg_daily_viewing_hours': round(max(0, avg_viewing_hours), 2),
            'preferred_genre': max(genre_preferences, key=genre_preferences.get),
            'genre_affinity_scores': str(genre_preferences),
            'devices': '|'.join(devices_used),
            'binge_watcher': binge_watcher,
            'weekend_viewer': weekend_viewer
        })
    
    return pd.DataFrame(users)

def generate_viewing_history(users_df, movies_df, n_interactions=100000):
    """
    Generate user-movie interaction data
    """
    print("Generating viewing history...")
    
    interactions = []
    
    for _ in range(n_interactions):
        # Select user and movie
        user = users_df.sample(1).iloc[0]
        
        # Bias movie selection based on user preferences
        movie = movies_df.sample(1).iloc[0]
        
        # Viewing details
        watch_date = datetime.now() - timedelta(days=np.random.randint(0, 365))
        
        # Completion rate affected by match quality
        genre_match = user['preferred_genre'] in movie['genres']
        base_completion = 0.7 if genre_match else 0.4
        completion_rate = min(1.0, max(0.1, np.random.normal(base_completion, 0.2)))
        
        # Rating (1-5) influenced by completion and match
        if completion_rate > 0.8:
            rating = np.random.choice([4, 5], p=[0.4, 0.6])
        elif completion_rate > 0.5:
            rating = np.random.choice([3, 4], p=[0.6, 0.4])
        else:
            rating = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3]) if np.random.random() > 0.5 else None
        
        # Session details
        device = np.random.choice(user['devices'].split('|'))
        time_of_day = np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'],
                                       p=[0.1, 0.2, 0.4, 0.3])
        
        # Engagement metrics
        did_search = np.random.choice([0, 1], p=[0.7, 0.3])
        from_recommendation = np.random.choice([0, 1], p=[0.4, 0.6])
        
        interactions.append({
            'interaction_id': f'INT_{str(_).zfill(6)}',
            'user_id': user['user_id'],
            'movie_id': movie['movie_id'],
            'watch_date': watch_date,
            'completion_rate': round(completion_rate, 3),
            'rating': rating,
            'device': device,
            'time_of_day': time_of_day,
            'watch_duration_minutes': int(movie['runtime_minutes'] * completion_rate),
            'did_search': did_search,
            'from_recommendation': from_recommendation
        })
    
    return pd.DataFrame(interactions)

def generate_session_data(users_df, movies_df, n_sessions=10000):
    """
    Generate session-based browsing data
    """
    print("Generating session data...")
    
    sessions = []
    
    for i in range(n_sessions):
        user = users_df.sample(1).iloc[0]
        session_id = f'SES_{str(i).zfill(5)}'
        
        # Session details
        session_start = datetime.now() - timedelta(days=np.random.randint(0, 30))
        n_movies_viewed = np.random.poisson(5) + 1  # At least 1
        
        # Movies browsed in this session
        movies_browsed = movies_df.sample(min(n_movies_viewed, len(movies_df)))
        
        for idx, (_, movie) in enumerate(movies_browsed.iterrows()):
            sessions.append({
                'session_id': session_id,
                'user_id': user['user_id'],
                'movie_id': movie['movie_id'],
                'timestamp': session_start + timedelta(seconds=idx * np.random.randint(10, 300)),
                'action': np.random.choice(['view', 'hover', 'click', 'add_to_list'],
                                         p=[0.5, 0.2, 0.2, 0.1]),
                'duration_seconds': np.random.randint(1, 300)
            })
    
    return pd.DataFrame(sessions)

if __name__ == "__main__":
    # Generate all data
    movies = generate_movie_catalog(2000)
    print(f"Created {len(movies)} movies/shows")
    
    users = generate_user_profiles(5000)
    print(f"Created {len(users)} user profiles")
    
    interactions = generate_viewing_history(users, movies, 100000)
    print(f"Created {len(interactions)} viewing interactions")
    
    sessions = generate_session_data(users, movies, 10000)
    print(f"Created {len(sessions)} session records")
    
    # Save samples
    print("\nSample movie:")
    print(movies[['title', 'genres', 'imdb_rating']].head(3))
    
    print("\nSample interactions:")
    print(interactions[['user_id', 'movie_id', 'rating', 'completion_rate']].head(3))