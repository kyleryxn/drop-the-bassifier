# Drop the Bassifier

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

### Authors
[Hillel Gersten](https://github.com/hillelg1), [Johnathan Sanchez](https://github.com/johnathansanchez16), 
[Kyle Schoenhardt](https://github.com/kyleryxn/)

## Overview

A machine learning-based project that classifies music genres using audio features provided by Spotify datasets. This 
project is particularly focused on classifying genres within the realm of Electronic Dance Music (EDM), but it can be 
extended to classify other genres as well.

By leveraging Spotify's audio features like danceability, energy, tempo, loudness, and more, the project analyzes a 
track’s characteristics to predict its genre. The core goal is to create a system that can take a song (via its Spotify 
track ID or its audio features) and classify it into a predefined set of genres or sub-genres.

## Datasets

### 1. [Classic Hits Dataset](https://www.kaggle.com/datasets/thebumpkin/10400-classic-hits-10-genres-1923-to-2023)

- The dataset is a comprehensive collection of 15,150 classic hits from 3,083 artists, spanning a century of music 
history from 1923 to 2023. This diverse dataset is divided into 19 distinct genres, showcasing the evolution of popular 
music across different eras and styles. Each track in the dataset is enriched with Spotify audio features, offering 
detailed insights into the acoustic properties, rhythm, tempo, and other musical characteristics.


## Challenges & Solutions

| Challenge   | Solution    |
|-------------|-------------|
| Challenge 1 | Solution 1  |

## Risks

### 1. Data Availability and Quality

- **Risk:** The dataset(s) may have missing or inconsistent data
- **Mitigation:** 
  - Perform data cleaning and imputation for missing values
  - Consider using a fallback strategy to handle missing or incomplete features (e.g., exclude tracks or impute values 
  based on similar tracks)

### 2. Genre Overlap and Ambiguity

- **Risk:** Some songs may exhibit features of multiple genres, making it difficult for the model to classify them accurately
- **Mitigation:** 
  - Consider using a **multi-label classification** approach for songs that span multiple genres
  - Experiment with different feature sets or introduce ensemble models to capture genre overlaps better

### 3. Data Bias and Imbalance

- **Risk:** Dataset(s) may be biased toward certain genres (e.g., more tracks of "house" than "trance"), which can lead 
to an imbalanced model that performs poorly on underrepresented genres
- **Mitigation:**
  - Apply techniques such as oversampling/undersampling or SMOTE to balance the dataset
  - Use class-weighted models to account for imbalance

### 4. Interpretability and User Trust

- **Risk:** Users may not fully understand how the model arrives at a genre classification, leading to reduced trust 
in the predictions
- **Mitigation:**
  - Use interpretability tools like SHAP or LIME to explain model predictions
  - Clearly communicate the limitations of the model in the README and documentation
  - Provide a way to show which features most influence the model's decisions for transparency

## Resources

| Member    | Resource |
|-----------|----------|
| Hillel    |          |
| Johnathan |          |
| Kyle      |          |

## Implementation

### 1. Data Collection
- Gather data on songs' audio features and analysis

### 2. Data Preprocessing and Cleaning
- Clean data and identify key metrics that will be needed then integrate the data while handling missing data and 
outliers

### 3. Model Development
- Build and train models that prioritize key metrics

### 4. Model Testing
- Test our models for accuracy

## License

Project is licensed under the GNU General Public License v3.0. See the [LICENSE](./LICENSE) for full details.

## Acknowledgements

We would like to thank the following organizations and people:

### 1. [CUNY Tech Prep](https://cunytechprep.org/)
- For all their guidance and leadership.

### 2. [Kaggle](https://www.kaggle.com/)
- For providing an excellent platform to access diverse datasets, including the datasets used in this project. 
Kaggle's community-driven datasets and competitions were instrumental in shaping the data analysis and machine 
learning aspects of this project.

### 3. [Spotify](https://open.spotify.com/)
- Without Spotify's wealth of music metadata, this project wouldn't have been able to explore genre classification with 
such depth and accuracy.