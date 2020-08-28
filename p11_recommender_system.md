## Recommender System

#### Author: Hao Zheng

While we talk about marketing, most of us will think about the diverse ads on the television. However, in real world, marketing is everywhere. Especially with the help from programming languages, advertisement can reach all corners of your life in an easier way without letting you notice it. For example, personizing the replying message so that customers feel connected.

In this project, we will lead you through a combination of text processing and marketing that mainly focus on the content optimization aspect: the recommender system and let you understand how the system actually works. We encourage you to create your own Jupytor notebook and follow along. You can also download this notebook together with any affiliated data in the [Notebooks and Data](https://github.com/Master-of-Business-Analytics/Notebooks_and_Data) GitHub repository. Alternatively, if you do not have Python or Jupyter Notebook installed yet, you may experiment with a virtual notebook by launching Binder or Syzygy below (learn more about these two tools in the [Resource](https://analytics-at-sauder.github.io/resource.html) tab). 

<a href="https://ubc.syzygy.ca/jupyter/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAnalytics-at-Sauder%2FProject_11_Recommender_System&urlpath=tree%2FProject_11_Recommender_System%2Fp11_recommender_system.ipynb&branch=master" target="_blank" class="button">Launch Syzygy (UBC)</a>

<a href="https://pims.syzygy.ca/jupyter/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAnalytics-at-Sauder%2FProject_11_Recommender_System&urlpath=tree%2FProject_11_Recommender_System%2Fp11_recommender_system.ipynb&branch=master" target="_blank" class="button">Launch Syzygy (Google)</a>

<a href="https://mybinder.org/v2/gh/Analytics-at-Sauder/Project_11_Recommender_System/master?filepath=p11_recommender_system.ipynb" target="_blank" class="button">Launch Binder</a>

## Business Problem

---

Here we can use the movie industry as an example to illustrate how the recommender system can actually be applied into a business context. The traditional movie content provider systems don’t care about the general taste of the customers because how they get their revenue is irrelevant to their ability to tell their customers’ taste. On the other words, the traditional movie seller only focus on the most welcomed movie and try to sell as many as possible. However, with the introduction of the age of internet, how current movie sellers makes money actually changes. Their income is now highly correlated with how long custoemrs spend on the site to watch the movie. So here the recommender system is required to make sure customers got the best recommendation and spend more time on the website.

We will use the movie rating dataset to try to duplicate that process.



```python
# Import the packages and read in the data

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
df1=pd.read_csv('data/tmdb_5000_credits.csv')
df2=pd.read_csv('data/tmdb_5000_movies.csv')
```

## Explore the data

---

The data is clean so here we do not want to go through the cleaning process again. But it is still useful to look at the data before starting to play around with it.


```python
# Explore the column names to find out what is in the dataframe
print("The first dataframe includes the information about: " )
for i in df1.columns: print(i, end  =", ")
print("\nThe second dataframe includes the information about: ")
for i in df2.columns: print(i, end  =", ")
```

    The first dataframe includes the information about: 
    movie_id, title, cast, crew, 
    The second dataframe includes the information about: 
    budget, genres, homepage, id, keywords, original_language, original_title, overview, popularity, production_companies, production_countries, release_date, revenue, runtime, spoken_languages, status, tagline, title, vote_average, vote_count, 

The first dataframe includes four different columns the reflects on the general production information about the movie whereas the second dataframe includes detailed information like genres and popularity. We can see that both dataset includes the unique identifier for the movie, so we can try to combine two datasets for simplicities.


```python
# Change the columns name to id so it is ready to merge
df1.columns = ['id','tittle','cast','crew']

# Merge two dataset on the unique identifier
new_df= pd.merge(df1,df2,on='id')
new_df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>tittle</th>
      <th>cast</th>
      <th>crew</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>...</th>
      <th>production_countries</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19995</td>
      <td>Avatar</td>
      <td>[{"cast_id": 242, "character": "Jake Sully", "...</td>
      <td>[{"credit_id": "52fe48009251416c750aca23", "de...</td>
      <td>237000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.avatarmovie.com/</td>
      <td>[{"id": 1463, "name": "culture clash"}, {"id":...</td>
      <td>en</td>
      <td>Avatar</td>
      <td>...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2009-12-10</td>
      <td>2787965087</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>285</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>[{"cast_id": 4, "character": "Captain Jack Spa...</td>
      <td>[{"credit_id": "52fe4232c3a36847f800b579", "de...</td>
      <td>300000000</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
      <td>http://disney.go.com/disneypictures/pirates/</td>
      <td>[{"id": 270, "name": "ocean"}, {"id": 726, "na...</td>
      <td>en</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2007-05-19</td>
      <td>961000000</td>
      <td>169.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>At the end of the world, the adventure begins.</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>6.9</td>
      <td>4500</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 23 columns</p>
</div>



## Building Recommender System

---

### Method 1: Demographic Filtering

This is the fundamental method that we will try to use here. In this method, we are giving users movie recommendation based on the genre of the movie. Generally, the movie with higher popularity will be liked by more people. So what we need to do this method is: 1. Find out a scientific way to reflect the popularity of the movie 2.Recommend the most popular movie.

The two major factors that we are using here will be vote_average and the vote_count. Vote average reflects the overall opinion whereas the vote count reflect how accurate the average score is. There are countless way to calculate for the "real" score, so feel free to think about your own method.

The final score calculation that we would use here would be:

average vote score + (vote score of selected film - average vote score) * ((vote count of selected film -  average vote count)/average vote count) ^ 2



```python
# Define the method to calculate the score:

am = new_df['vote_average'].mean()
bm = new_df['vote_count'].mean()

def rating(x):
    a = x['vote_average']
    b = x['vote_count']
    
    return am + (a - am)* ((b - bm)/bm) ** 2

```


```python
# Apply the function to the entire dataframe

## Create a new df for method 1
new_df1 = new_df.copy()

new_df1['score_cal'] = new_df.apply(rating,axis = 1)
new_df1.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>tittle</th>
      <th>cast</th>
      <th>crew</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>...</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>score_cal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19995</td>
      <td>Avatar</td>
      <td>[{"cast_id": 242, "character": "Jake Sully", "...</td>
      <td>[{"credit_id": "52fe48009251416c750aca23", "de...</td>
      <td>237000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.avatarmovie.com/</td>
      <td>[{"id": 1463, "name": "culture clash"}, {"id":...</td>
      <td>en</td>
      <td>Avatar</td>
      <td>...</td>
      <td>2009-12-10</td>
      <td>2787965087</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
      <td>293.111430</td>
    </tr>
    <tr>
      <th>1</th>
      <td>285</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>[{"cast_id": 4, "character": "Captain Jack Spa...</td>
      <td>[{"credit_id": "52fe4232c3a36847f800b579", "de...</td>
      <td>300000000</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
      <td>http://disney.go.com/disneypictures/pirates/</td>
      <td>[{"id": 270, "name": "ocean"}, {"id": 726, "na...</td>
      <td>en</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>...</td>
      <td>2007-05-19</td>
      <td>961000000</td>
      <td>169.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>At the end of the world, the adventure begins.</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>6.9</td>
      <td>4500</td>
      <td>30.704168</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 24 columns</p>
</div>




```python
# Print out the top five films
new_df1 = new_df1.sort_values('score_cal', ascending=False)
new_df1[['title', 'score_cal']].head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>score_cal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>96</th>
      <td>Inception</td>
      <td>725.141884</td>
    </tr>
    <tr>
      <th>65</th>
      <td>The Dark Knight</td>
      <td>572.233379</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Interstellar</td>
      <td>442.582874</td>
    </tr>
    <tr>
      <th>662</th>
      <td>Fight Club</td>
      <td>358.708821</td>
    </tr>
    <tr>
      <th>16</th>
      <td>The Avengers</td>
      <td>343.465619</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualize the result

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(new_df1['title'][0:5],new_df1['score_cal'][0:5])
plt.grid(True)
plt.show()
```


![png](output_11_0.png)


Based on the calculated results, we will be able to recomend Inception, The Dark Knight, Intersetllar, Fight Club and The Avengers to all the people because these are the most welcomed popular movies.

However, This is not an ideal way of making a recommender system because there is no personalized recommendation in it. All the people will receive the same recommendation no matter which type of genres they prefer. So we might want to try on a different method to include personal preference into our system.

### Method 2: Content Based Filtering

This method will not focus on recommending similar film based on personal taste. Which means, if you just watched an action movie with english subtitle, the next movie recommended for you is likely to be another action moview with english subtitle. I will show you how to build such a system.

In this recommender system, I will use genres and keywords to help me find out similar films. In the dataset, the both genres and keywords are stored as string value, so we need to do some text extraction before moving to next step.


```python
# Extract important words form selected_columns

from ast import literal_eval

new_df2 = new_df.copy()
selected_columns = ["genres","keywords"]

for feature in selected_columns:
    new_df2[feature] = new_df2[feature].apply(literal_eval)
```


```python
# Return the top 3 elements/entire list(if there are less than 3 elements) and change all element to low case
## Reference: 
## https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system/#Content-Based-Filtering

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names
    return []

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
```


```python
# Apply the defined method so it is ready for further processing

for feature in selected_columns:
    new_df2[feature] = new_df2[feature].apply(get_list)
    
for feature in selected_columns:
    new_df2[feature] = new_df2[feature].apply(clean_data)

```

In the next step, we use the cosines similarity to find which movies to recommend. On the other hand, the cosines similarity is the new score we use in this second method.


```python
# Concat the two feature together 

for i, r in new_df2.iterrows():
     new_df2['test'][i] = " ".join(new_df2["genres"][i]) + " ".join(new_df2["keywords"][i])

# Use Vectorizer to change words into matrix

from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(new_df2['test'])

# Introduce cosines similarity
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
```

    /Users/haozheng/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    


```python
#Construct a reverse map of indices and movie titles
indices = pd.Series(new_df2.index, index=new_df2['title']).drop_duplicates()

# Use the cosines similarity to find out which movies to recommend
def get_recom(title,cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]

```


```python
get_recom('Avatar', cosine_sim2)
```




    85        Captain America: The Winter Soldier
    2444                          Damnation Alley
    71      The Mummy: Tomb of the Dragon Emperor
    83                                 The Lovers
    518                          Inspector Gadget
    600                              Killer Elite
    678                              Dragon Blade
    786                         The Monkey King 2
    1273                              Extreme Ops
    1324                         Virgin Territory
    Name: title, dtype: object



Here we have the recommending list that is based on user's preference. This is going to perform way better than the first method. However, we can still continue to improve it.

## Next Step

---

Apart from these two method, there is a third way out there which is called the collaborative filtering which combines content based filtering and demographic filtering. In this method, you can combine the result from two methods giving different weights to them. 

For example, if avater is the 4th movie using content based filtering and the 20th movie using demographic filtering in a database the consist of 100 movies. We can generate a new score with reverse ranking method if two method have same weights in our calculation:

0.5 * (100 - 4) + 0.5 * (100 - 20)

We can use the new score to rank the movie again just like we did in demographic filtering method.

Can you try to do it yourself?

## Reference

---

https://www.kaggle.com/tmdb/tmdb-movie-metadata


https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system/?select=tmdb_5000_movies.csv
