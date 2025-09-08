import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_process_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')

    # Convert JSON-like strings to lists
    def convert_json_to_list(text):
        try:
            data = ast.literal_eval(text)
            return [item['name'] for item in data]
        except:
            return []

    movies['genres'] = movies['genres'].apply(convert_json_to_list)
    movies['keywords'] = movies['keywords'].apply(convert_json_to_list)
    movies['production_companies'] = movies['production_companies'].apply(convert_json_to_list)
    movies['production_countries'] = movies['production_countries'].apply(convert_json_to_list)
    movies['spoken_languages'] = movies['spoken_languages'].apply(convert_json_to_list)
    credits['cast'] = credits['cast'].apply(convert_json_to_list)
    credits['crew'] = credits['crew'].apply(convert_json_to_list)

    movies = movies.merge(credits, left_on='id', right_on='movie_id')

    # Combine features into one column
    def combine_features(row):
        return ' '.join(row['genres']) + ' ' + ' '.join(row['keywords']) + ' ' + ' '.join(row['production_companies'])+' '+' '.join(row['production_countries'])+' '+' '.join(row['cast']) + ' ' + ' '.join(row['crew'])

    movies['combined_features'] = movies.apply(combine_features, axis=1)

    # Vectorize combined features and compute cosine similarity
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return movies, cosine_sim
