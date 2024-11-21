# Drop the Bassifier

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkyleryxn%2Fdrop-the-bassifier&count_bg=%231DB954&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits&edge_flat=false)](https://hits.seeyoufarm.com)

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
*Provided by Kyle*

- The dataset is a comprehensive collection of 15,150 classic hits from 3,083 artists, spanning a century of music 
history from 1923 to 2023. This diverse dataset is divided into 19 distinct genres, showcasing the evolution of popular 
music across different eras and styles. Each track in the dataset is enriched with Spotify audio features, offering 
detailed insights into the acoustic properties, rhythm, tempo, and other musical characteristics.

### 2. [Million Song Dataset](http://millionsongdataset.com/)
*Provided by Hillel*

- The Million Song Dataset is a freely-available collection of audio features and metadata for a million contemporary 
popular music tracks. The dataset does not include any audio, only the derived features. Its purposes are to encourage 
research on algorithms that scale to commercial sizes; to provide a reference dataset for evaluating research; as a 
shortcut alternative to creating a large dataset with APIs.

### 3. [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
*Provided by Johnathan*

- Music. Experts have been trying for a long time to understand sound and what differenciates one song from another. 
How to visualize sound. What makes a tone different from another. This data hopefully can give the opportunity to do 
just that.

## Challenges & Solutions

| Challenge                               | Solution                                                                                        |
|-----------------------------------------|-------------------------------------------------------------------------------------------------|
| Inexperience in Machine learning models | Use of resouces to learn which existing models will suit this project better                    |
| Finding patterns in different music     | Converting audio files into spectrographs can give us frequency and amplitude over time         |
| Accuracy of our model                   | After developing the model we can use songs not included in the training set to train our model |
 

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

| Member    | Resource                                                                                                                                                                                                                                                                                                                                        |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Hillel    |                                                                                                                                                                                                                                                                                                                                                 |
| Johnathan | [Music Genre Classification using CNN](https://www.clairvoyant.ai/blog/music-genre-classification-using-cnn), [Rhythm Tips for Identifying Music Genres by Ear](https://www.musical-u.com/learn/rhythm-tips-for-identifying-music-genres-by-ear/#:~:text=There%20are%20some%20genres%20that,distinguishes%20liquid%20dubstep%20and%20darkstep.) |
| Kyle      | [Music Genre Classification System Introduction](https://www.youtube.com/watch?v=KW6585XMV3c&list=PLvz5lCwTgdXCd200WNDupTMo15DP9iryv)                                                                                                                                                                                                           |

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

## Tech Stack

### Core Development
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/en/3.0.x/)

### Exploratory Data Analysis (EDA) and Model Experimentation
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)

### Web Development
[![Jinja](https://img.shields.io/badge/Jinja-B41717?style=for-the-badge&logo=jinja&logoColor=white)](https://developer.mozilla.org/en-US/docs/Glossary/HTML5)
[![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Glossary/HTML5)
[![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)
[![SCSS](https://img.shields.io/badge/SCSS-CC6699?style=for-the-badge&logo=sass&logoColor=white)](https://sass-lang.com/)

### Version Control
[![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)](https://git-scm.com/)

### Team Collaboration
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/)
[![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)](https://slack.com/)
[![Trello](https://img.shields.io/badge/Trello-0052CC?style=for-the-badge&logo=trello&logoColor=white)](https://trello.com/)

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

### 3. [Million Song Dataset](http://millionsongdataset.com/)
*Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. 
The Million Song Dataset. In Proceedings of the 12th International Society
for Music Information Retrieval Conference (ISMIR 2011), 2011.*

- For going to the trouble of providing a free, public dataset perfectly suited for our project's needs.
