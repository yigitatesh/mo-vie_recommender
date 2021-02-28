import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from sklearn.neighbors import KNeighborsClassifier

# LOAD DATA
X = load_npz("X_data.npz")
y = pd.read_csv("y_data.csv")

# CREATE MODEL
knn = KNeighborsClassifier().fit(X, y.values.ravel())

# FUNCTIONS
def get_similar_movies(movie_name, X, y, knn, n_movies=10):
    """Returns n similar to the given movie"""
    index = y.loc[y["title"].str.lower() == movie_name].index[0]
    movie_data = X[index, :].toarray()
    distances, indices = knn.kneighbors(movie_data, n_neighbors=n_movies+1)
    
    movies = []
    for i in np.squeeze(indices):
        movie = y.iloc[i]["title"]
        movies.append(movie)
    
    return movies[1:]

def main(X, y, knn):
    run = True
    while run:
        # get movie name
        movie_name = input("\nEnter movie name to find similar movies (q to quit): ")
        
        if movie_name == "q":
            run = False
            continue
        
        movie_name = movie_name.lower()
        
        # name is in dataset
        if movie_name in y["title"].str.lower().values:
            n_movies = int(input("How many similar movies (integer): "))
            movies = get_similar_movies(movie_name, X, y, knn, n_movies=n_movies)
            
            # inform user
            print("\nSimilar Movies:")
            for i, movie in enumerate(movies):
                print(str(i+1) + ". " + movie)
        
        # some movie names contains searched movie name
        elif y["title"].str.lower().str.contains(movie_name).any():
            similar_names = y[y["title"].str.lower().str.contains(movie_name)].values.ravel()

            # print similar names
            print("\nMovie: \"{}\" is not founded!".format(movie_name))
            print("Similar Movie Names you can look:")
            for i, movie in enumerate(similar_names):
                print(str(i+1) + ". " + movie)
        
        # not founded at all
        else:
            print("\nMovie: \"{}\" or a similar movie name is not founded!".format(movie_name))
            print("Try Again.")

if __name__ == "__main__":
    main(X, y, knn)