import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load Dataset (Fixing File Encoding Issue)
file_path = r"C:\Users\kris9\Downloads\imdb.csv"  # Change this if needed

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {file_path}. Please check the file path.")

try:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Fix UnicodeDecodeError
    print("✅ Dataset Loaded Successfully!\n")
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='latin1')  # Try an alternative encoding
    print("✅ Dataset Loaded with 'latin1' Encoding!\n")

# Step 2: Explore Dataset
print("🔹 First 5 rows of the dataset:")
print(df.head())
print("\n🔹 Dataset Info:")
print(df.info())

# Step 3: Handle Missing Values
print("\n🔹 Missing Values in Dataset:")
print(df.isnull().sum())

# Drop unnecessary columns if they exist
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Selecting Features & Target Variable
df.columns = df.columns.str.strip().str.lower()  # Convert all column names to lowercase
target = 'rating'

# Verify if the required columns exist
required_features = {'genre', 'director', 'duration', 'year', 'actor 1', 'actor 2', 'actor 3'}
missing_features = required_features - set(df.columns)
if missing_features:
    raise ValueError(f"❌ Missing required columns in dataset: {missing_features}")

# Creating 'actors' column by combining 'actor 1', 'actor 2', 'actor 3'
df['actors'] = df[['actor 1', 'actor 2', 'actor 3']].fillna('').agg(' '.join, axis=1).str.strip()

features = ['genre', 'director', 'actors', 'duration', 'year']

# Drop rows where target is missing
df = df.dropna(subset=[target])

# Separate Features (X) and Target (y)
X = df[features]
y = df[target]

# Step 4: Preprocessing Pipeline
numerical_features = ['duration', 'year']
categorical_features = ['genre', 'director', 'actors']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', dtype=str)),  # Ensure dtype compatibility
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Step 5: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)
print("\n✅ Model Training Complete!")

# Step 7: Evaluate Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 8: Predict on New Data (Example)
new_movie = pd.DataFrame({
    'genre': ['Action'],
    'director': ['Christopher Nolan'],
    'actors': ['Leonardo DiCaprio'],
    'duration': [150],
    'year': [2024]
})

# Ensure the new movie data matches the training data format
new_movie_pred = model.predict(new_movie)

print(f"\n🎬 Predicted Rating for New Movie: {new_movie_pred[0]:.2f}")
