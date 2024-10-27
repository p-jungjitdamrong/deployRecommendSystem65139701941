
# prompt: # Generate app for streamlit.io use above content with recommendation_movie_svd.pkl

import streamlit as st
import pickle
import pandas as pd

# Load the saved model and data
with open('recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

def get_recommendations(user_id, top_n=10):
    """
    Generates top N movie recommendations for a given user ID.

    Args:
        user_id: The ID of the user for whom to generate recommendations.
        top_n: The number of top recommendations to return.

    Returns:
        A list of top N movie recommendations.
    """
    rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
    unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
    pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
    sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
    top_recommendations = sorted_predictions[:top_n]
    recommendations = []
    for recommendation in top_recommendations:
        movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
        recommendations.append((movie_title, recommendation.est))
    return recommendations

# Streamlit app
st.title("Movie Recommendation System")

# Get user input for user ID
user_id = st.number_input("Enter your user ID:", min_value=1, value=1)

# Generate recommendations
if st.button("Get Recommendations"):
    recommendations = get_recommendations(user_id)
    if recommendations:
        st.subheader("Top 10 Recommended Movies:")
        for movie_title, est_rating in recommendations:
            st.write(f"- {movie_title} (Estimated Rating: {est_rating:.2f})")
    else:
        st.write("No recommendations found for this user.")

