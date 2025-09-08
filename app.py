# Save this script as app.py
from model import recommend_movies
from data import load_and_process_data

movies, cosine_sim = load_and_process_data()

import streamlit as st

# Use previously defined recommend_movies function and movies DataFrame

st.title("Movie Recommender System")

movie_title = st.text_input("Enter a movie title:")

if st.button("Recommend"):
    if movie_title:
        recommendations = recommend_movies(movie_title,cosine_sim, movies)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.write(f"Top recommendations similar to '{movie_title}':")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
    else:
        st.warning("Please enter a movie title.")