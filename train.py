import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("Housing.csv")  # Make sure this file is in your folder

# Features and target
X = data[['area']]  # Assuming your CSV has 'area' column
y = data['price']   # And a 'price' column

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model to a .pkl file
joblib.dump(model, 'model.pkl')
print("✅ Model saved as model.pkl")
