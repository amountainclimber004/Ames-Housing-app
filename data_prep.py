import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_excel('Ames_Housing.xlsx')

# Handle missing values
df = df.dropna(subset=['SalePrice'])

# Select features and target
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Define preprocessing
numeric_features = ['GrLivArea', 'TotalBsmtSF', 'BedroomAbvGr']
categorical_features = ['Neighborhood', 'HouseStyle']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Train model
model.fit(X_train, y_train)

# Save model
import joblib
joblib.dump(model, 'housing_model.pkl')
