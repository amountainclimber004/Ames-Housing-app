import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
df = pd.read_excel('Ames_Housing.xlsx')

# Drop rows where SalePrice is missing
df = df.dropna(subset=['SalePrice'])

# Define features and target
numeric_features = ['GrLivArea', 'TotalBsmtSF', 'BedroomAbvGr']
categorical_features = ['Neighborhood', 'House Style']

X = df[numeric_features + categorical_features].copy()  # Select only required columns
y = df['SalePrice']

# Handle missing values
X[numeric_features] = X[numeric_features].fillna(0)  # Fill numeric NaN with 0
X[categorical_features] = X[categorical_features].fillna('Unknown').astype(str)  # Fill categorical NaN with 'Unknown'

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split data (Ensuring reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
])

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'housing_model.pkl')

print("Model saved successfully as 'housing_model.pkl'")

