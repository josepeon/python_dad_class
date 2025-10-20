import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests, time

api = "https://ghibliapi.vercel.app"

def get_data(endpoint):
    response = requests.get(f"{api}/{endpoint}")
    return response.json()

films_data = get_data("films")
people_data = get_data("people")
films_df = pd.DataFrame(films_data)
people_df = pd.DataFrame(people_data)       

# Data Cleaning
films_df['release_date'] = pd.to_datetime(films_df['release_date'], format='%Y')
people_df['age'] = pd.to_numeric(people_df['age'], errors='coerce') 
people_df['age'].fillna(people_df['age'].median(), inplace=True)
people_df['gender'].replace({'NA': 'Unknown', '': 'Unknown'}, inplace=True) 
# Merging Data
merged_df = pd.merge(people_df, films_df, left_on='films', right_on='url', how='left', suffixes=('_person', '_film'))       
# Data Exploration
print("Films DataFrame Info:")
print(films_df.info())
print("\nPeople DataFrame Info:")
print(people_df.info())
print("\nMerged DataFrame Info:")
print(merged_df.info()) 
print("\nFilms DataFrame Description:")
print(films_df.describe())
print("\nPeople DataFrame Description:")
print(people_df.describe())
print("\nMerged DataFrame Description:")
print(merged_df.describe())
# Visualizations
plt.figure(figsize=(10, 6))
plt.hist(films_df['release_date'].dt.year, bins=range(1980, 2030, 5), color='skyblue', edgecolor='black')
plt.title('Distribution of Film Release Years')
plt.xlabel('Year')
plt.ylabel('Number of Films')
plt.grid(axis='y')
plt.show()
plt.figure(figsize=(10, 6))
plt.hist(people_df['age'], bins=20, color='salmon', edgecolor='black')
plt.title('Distribution of Ages of Characters')
plt.xlabel('Age')
plt.ylabel('Number of Characters')
plt.grid(axis='y')
plt.show()
plt.figure(figsize=(10, 6))     
gender_counts = people_df['gender'].value_counts()
plt.bar(gender_counts.index, gender