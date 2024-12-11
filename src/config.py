import os

# Get the project's root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the model's path relative to the project root
MODEL_PATH = os.path.join(BASE_DIR, 'notebooks', 'model_training', 'genre_model.joblib')
