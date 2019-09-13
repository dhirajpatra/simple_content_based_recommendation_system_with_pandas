import pandas as pd

pd.set_option('max_rows', 20)
pd.set_option('mode.chained_assignment', None)  # default='warn'

movies_data = 'movies.csv'
ratings_data = 'movie_ratings.csv'

# Defining additional NaN identifiers.
missing_values = ['na', '--', '?', '-', 'None', 'none', 'non']
movies_df = pd.read_csv(movies_data, na_values=missing_values)
ratings_df = pd.read_csv(ratings_data, na_values=missing_values)

# movies_df.shape

# Using regular expressions to find a year stored between parentheses
# We specify the parentheses so we don't conflict with movies that have years in their titles.
movies_df['year'] = movies_df.title.str.extract('(\d\d\d\d)', expand=False)

# Removing the years from the 'title' column.
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')

# Applying the strip function to get rid of any ending white space characters
# that may have appeared, using lambda function.
movies_df['title'] = movies_df.title.apply(lambda x: x.strip())

# Every genre is separated by a | so we simply have to call the split function on |.
movies_df['genres'] = movies_df.genres.str.split('|')

# movies_df.info
movies_df.isna().sum()

# Filling year NaN values with zeros
movies_df.year.fillna(0, inplace=True)

# Converting columns year from obj to int16 and movieId from int64 to int32 to save memory.
movies_df.year = movies_df.year.astype('int16')
movies_df.movieId = movies_df.movieId.astype('int32')
# movies_df.dtypes

# First let's make a copy of the movies_df.
movies_with_genres = movies_df.copy(deep=True)

# Let's iterate through movies_df, then append the movie genres as columns of 1s or 0s.
# 1 if that column contains movies in the genre at the present index and 0 if not.
x = []
for index, row in movies_df.iterrows():
    x.append(index)
    for genre in row['genres']:
        movies_with_genres.at[index, genre] = 1

# Confirm that every row has been iterated and acted upon.
print(len(x) == len(movies_df))
# movies_with_genres.head(3)

# Filling in the NaN values with 0 to show that a movie doesn't have that column's genre.
movies_with_genres = movies_with_genres.fillna(0)
movies_with_genres.head(3)

# print out the shape and first five rows of ratings data.
# ratings_df.head()

# Dropping the timestamp column
ratings_df.drop('timestamp', axis=1, inplace=True)
# Confirming the drop
# ratings_df.head(3)

# Let's confirm the right data types exist per column in ratings data_set
print(ratings_df.dtypes)
print(ratings_df.isna().sum())

# Notice: Feel free to add or remove movies from the list of dictionaries below!
# Just be sure to write it in with capital letters and if a movie starts with a “The”,
# like “The Avengers” then write it in like this: ‘Avengers, The’ .
# Creating Lawrence’s Profile
# so on a scale of 0 to 5, with 0 min and 5 max, see Lawrence's movie ratings below.
Lawrence_movie_ratings = [
    {'title': 'Predator', 'rating': 4.9},
    {'title': 'Final Destination', 'rating': 4.9},
    {'title': 'Mission Impossible', 'rating': 4},
    {'title': "Beverly Hills Cop", 'rating': 3},
    {'title': 'Exorcist, The', 'rating': 4.8},
    {'title': 'Waiting to Exhale', 'rating': 3.9},
    {'title': 'Avengers, The', 'rating': 4.5},
    {'title': 'Omen, The', 'rating': 5.0}
]
Lawrence_movie_ratings = pd.DataFrame(Lawrence_movie_ratings)
# Lawrence_movie_ratings.head()

# Extracting movie Ids from movies_df and updating lawrence_movie_ratings with movie Ids.
Lawrence_movie_Id = movies_df[movies_df['title'].isin(Lawrence_movie_ratings['title'])]
# Merging Lawrence movie Id and ratings into the lawrence_movie_ratings data frame.
# This action implicitly merges both data frames by the title column.
Lawrence_movie_ratings = pd.merge(Lawrence_movie_Id, Lawrence_movie_ratings)
# Display the merged and updated data frame.
# Lawrence_movie_ratings

# Dropping information we don't need such as year and genres
Lawrence_movie_ratings = Lawrence_movie_ratings.drop(['genres', 'year'], 1)

# Final profile for Lawrence
print(Lawrence_movie_ratings)

# filter the selection by outputing movies that exist in both
# Lawrence_movie_ratings and movies_with_genres.
Lawrence_genres_df = movies_with_genres[movies_with_genres.movieId.isin(Lawrence_movie_ratings.movieId)]
# Lawrence_genres_df

# First, let's reset index to default and drop the existing index.
Lawrence_genres_df.reset_index(drop=True, inplace=True)

# Next, let's drop redundant columns
Lawrence_genres_df.drop(['movieId', 'title', 'genres', 'year'], axis=1, inplace=True)

# let's confirm the shapes of our data frames to guide us as we do matrix multiplication.
print('Shape of Lawrence_movie_ratings is:', Lawrence_movie_ratings.shape)
print('Shape of Lawrence_genres_df is:', Lawrence_genres_df.shape)

# Let's find the dot product of transpose of Lawrence_genres_df by Lawrence rating column.
Lawrence_profile = Lawrence_genres_df.T.dot(Lawrence_movie_ratings.rating)
# Lawrence_profile

# let's set the index to the movieId.
movies_with_genres = movies_with_genres.set_index(movies_with_genres.movieId)
# movies_with_genres.head()

# Deleting four unnecessary columns.
movies_with_genres.drop(['movieId', 'title', 'genres', 'year'], axis=1, inplace=True)

# Multiply the genres by the weights and then take the weighted average.
recommendation_table_df = (movies_with_genres.dot(Lawrence_profile) / Lawrence_profile.sum())

# Let's sort values from great to small
recommendation_table_df.sort_values(ascending=False, inplace=True)
# recommendation_table_df.head()

# first we make a copy of the original movies_df
copy = movies_df.copy(deep=True)

# Then we set its index to movieId
copy = copy.set_index('movieId', drop=True)

# Next we enlist the top 20 recommended movieIds we defined above
top_20_index = recommendation_table_df.index[:20].tolist()

# finally we slice these indices from the copied movies df and save in a variable
recommended_movies = copy.loc[top_20_index, :]

# Now we can display the top 20 movies in descending order of preference
print(recommended_movies)