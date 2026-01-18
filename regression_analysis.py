import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
# Step 1: Load dataset
# --------------------------------------------------
df = pd.read_csv("churn-bigml-20.csv")

print("Dataset loaded successfully")
print(df.head())

# --------------------------------------------------
# Step 2: Select features for regression
# Predicting Total day charge based on Total day minutes
# --------------------------------------------------
X = df[["Total day minutes"]]
y = df["Total day charge"]

# --------------------------------------------------
# Step 3: Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Step 4: Train Linear Regression model
# --------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------------
# Step 5: Make predictions
# --------------------------------------------------
y_pred = model.predict(X_test)

# --------------------------------------------------
# Step 6: Evaluate model
# --------------------------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print("Mean Squared Error (MSE):", mse)
print("R-squared Score:", r2)

# --------------------------------------------------
# Step 7: Visualize regression results
# --------------------------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(X_test, y_test, color="blue", label="Actual Data")
plt.plot(X_test, y_pred, color="red", label="Regression Line")
plt.xlabel("Total Day Minutes")
plt.ylabel("Total Day Charge")
plt.title("Linear Regression: Day Minutes vs Day Charges")
plt.legend()
plt.tight_layout()
plt.savefig("regression_result.png")
plt.show()

print("\n✅ Regression analysis completed successfully!")
