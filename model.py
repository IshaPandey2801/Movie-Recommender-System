import pandas as pd

def recommend_movies(title, cosine_sim, movies, top_n=10):
    indices = pd.Series(movies.index, index=movies['original_title']).drop_duplicates()
    idx = indices.get(title)

    if idx is None:
        return f"Movie '{title}' not found in dataset."

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]

    return movies['original_title'].iloc[movie_indices].tolist()
