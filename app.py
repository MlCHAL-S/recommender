import streamlit as st
import pandas as pd
from recommender import Recommender

# Initialize Recommender
recommender = Recommender('data/ratings.csv', 'data/movies.csv')

movies = pd.read_csv('data/movies.csv')

home_css = """
            <style>
            [data-testid="stAppViewContainer"] {
            background: radial-gradient(#711F22, black);
              background-size: 400% 400%;
              animation: gradient 20s ease infinite;
              height: 100vh;
            }

            @keyframes gradient {
              0% {
                background-position: 0% 50%;
              }
              50% {
                background-position: 100% 50%;
              }
              100% {
                background-position: 0% 50%;
              }
            }

            /* Hide the header */
            header {visibility: hidden;}

            /* Hide the footer */
            footer {visibility: hidden;}
            </style>
             """

recommender_css = """
            <style>
            [data-testid="stAppViewContainer"] {
            background: radial-gradient(#3c3c3c, black);
              background-size: 400% 400%;
              animation: gradient 15s ease infinite;
              height: 100vh;
            }

            @keyframes gradient {
              0% {
                background-position: 0% 50%;
              }
              50% {
                background-position: 100% 50%;
              }
              100% {
                background-position: 0% 50%;
              }
            }

            /* Hide the header */
            header {visibility: hidden;}

            /* Hide the footer */
            footer {visibility: hidden;}
            </style>
             """


def home_page():
    st.markdown(home_css, unsafe_allow_html=True)
    st.title("Welcome to the Movie Recommender System")
    st.write("""
        This application serves the purpose of recommending 10 movies based on the chosen movie.
        
    
        **How it works:**
        - **Collaborative Filtering**: Finds patterns from user behavior to suggest movies.
        - **k-Nearest Neighbors (k-NN)**: Uses the k-NN algorithm to find similar movies based on ratings.
        - **MovieLens Dataset**: Utilizes the MovieLens dataset for ratings and movie information.
        - **Streamlit**: Built with Streamlit for a simple and interactive interface.

        Click the button below to get movie recommendations.
    """)
    if st.button("Continue"):
        st.session_state.page = "recommendations"
        st.experimental_rerun()  # Trigger rerun immediately to reflect the state change


def recommendations_page():
    st.markdown(recommender_css, unsafe_allow_html=True)
    st.title("Movie Recommendations")

    st.write("Find other interesting movies based on your choice.")

    # Dropdown search bar
    movie_titles = recommender.movies['title'].tolist()
    selected_movie_title = st.selectbox("Type in or search for a movie title", ["Search for a movie"] + movie_titles,
                                        index=0)

    if selected_movie_title == "Search for a movie":
        st.write("Recommendations will appear here.")
    else:
        movie_id = recommender.movies[recommender.movies['title'] == selected_movie_title].iloc[0]['movieId']
        recommendations = recommender.recommend(movie_id)
        st.write(f"Because you watched {selected_movie_title}, you might also like:")
        st.write('')
        for movie in recommendations:
            st.write(movie)


# Define the main function to control the app's flow
def main():
    if 'page' not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "recommendations":
        recommendations_page()


if __name__ == "__main__":
    main()
