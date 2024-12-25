import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load training data
training_data = pd.read_excel('TrainingData.xlsx')

# Drop 'Home Team' and 'Away Team' columns for readability
training_data = training_data.drop(['Home Team', 'Away Team'], axis=1)

# Features (X) and Targets (y)
X = training_data.drop(['Home Score', 'Away Score'], axis=1)  # Drop target columns
y = training_data[['Home Score', 'Away Score']]  # Target columns

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Load prediction data
prediction_data = pd.read_excel('PredictionData.xlsx')

# Drop 'Home Team' and 'Away Team' columns for readability
prediction_data = prediction_data.drop(['Home Team', 'Away Team'], axis=1)

# Ensure prediction_data matches the training features
prediction_features = prediction_data[X.columns]  # Select only columns used for training

# Predict using the trained model
predictions = model.predict(prediction_features)

# Add predictions to the Excel file
prediction_data['Predicted Home Score'] = predictions[:, 0]
prediction_data['Predicted Away Score'] = predictions[:, 1]

# Save the file with predictions
prediction_data.to_excel('PredictionData_with_Predictions.xlsx', index=False)