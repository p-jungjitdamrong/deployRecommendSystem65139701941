import streamlit as st
import pickle

# Load data back from the file
@st.cache_data
def load_model():
    with open('./recommendation_movie_svd.pkl', 'rb') as file:
        svd_model, movie_ratings, movies = pickle.load(file)
    return svd_model, movie_ratings, movies

# Function to get top N recommendations for a user
def get_top_recommendations(user_id, svd_model, movie_ratings, movies, top_n=10):
    rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
    unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
    pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
    sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
    top_recommendations = sorted_predictions[:top_n]
    return [(movies[movies['movieId'] == rec.iid]['title'].values[0], rec.est) for rec in top_recommendations]

# Streamlit app
st.title("Movie Recommendation System")

# Load the SVD model and data
svd_model, movie_ratings, movies = load_model()

number_movie = st.number_input("Enter Number Movie to recommend:", min_value=1, step=1, value=10)

# Input for user ID
user_id = st.number_input("Enter User ID:", min_value=1, step=1)

# Button to get recommendations
if st.button("Get Recommendations"):
    if user_id:
        top_recommendations = get_top_recommendations(user_id, svd_model, movie_ratings, movies, number_movie)
        st.subheader(f"Top 10 Movie Recommendations for User {user_id}:")
        for title, rating in top_recommendations:
            st.write(f"{title} (Estimated Rating: {rating:.2f})")
    else:
        st.write("Please enter a valid User ID.")
