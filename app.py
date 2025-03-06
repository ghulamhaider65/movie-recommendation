import streamlit as st
import pandas as pd
import pickle
import requests
from scipy.sparse import load_npz

# Custom CSS for improved UI
st.markdown(
    """
    <style>
    body {
        background-color: #1f1f2e;
        color: white;
    }
    .stApp {
        background-image: linear-gradient(to right, #6a11cb, #2575fc);
        color: white;
    }
    .css-1d391kg {
        background-color: rgba(0, 0, 0, 0.6) !important;
        border-radius: 10px;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App Title
st.title("🎬 Movie Recommendation System")
st.write("Created by **Ghulam Haider**")
st.markdown("[LinkedIn Profile](https://www.linkedin.com/in/ghulam-haider-586177297/)")

# Function to fetch movie poster
def fetch_poster(movie_id):
    api_key = "8265bd1679663a7ea12ac168da84d2e8"
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US")
    data = response.json()
    return f"https://image.tmdb.org/t/p/w500/{data.get('poster_path', '')}"

# Load models efficiently
try:
    similarity = load_npz("similarity.npz")  # Load sparse matrix
    movies_dict = pickle.load(open("movies_dict.pkl", "rb"))
    movies = pd.DataFrame(movies_dict)
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Recommendation function
def recommend(movie):
    if movie not in movies['title'].values:
        return [], []

    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index].toarray().flatten()
    movies_list = sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_posters

# Dropdown to select a movie
selected_movie_name = st.selectbox("Select a movie:", movies['title'].values)

if st.button("Recommend"):
    names, posters = recommend(selected_movie_name)

    if names:
        cols = st.columns(len(names))
        for idx, col in enumerate(cols):
            with col:
                st.text(names[idx])
                st.image(posters[idx])
    else:
        st.warning("No recommendations found. Try a different movie.")

