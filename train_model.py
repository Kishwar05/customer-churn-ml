import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("churn.csv")

# Drop rows with missing TotalCharges (or convert to numeric safely)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Convert target variable to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical variables
df['gender_Male'] = (df['gender'] == 'Male').astype(int)
df['PaymentMethod_Electronic check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
df['PaymentMethod_Mailed check'] = (df['PaymentMethod'] == 'Mailed check').astype(int)
df['PaymentMethod_Bank transfer (automatic)'] = (df['PaymentMethod'] == 'Bank transfer (automatic)').astype(int)
df['PaymentMethod_Credit card (automatic)'] = (df['PaymentMethod'] == 'Credit card (automatic)').astype(int)
df['Contract_Two year'] = (df['Contract'] == 'Two year').astype(int)
df['InternetService_Fiber optic'] = (df['InternetService'] == 'Fiber optic').astype(int)

# Feature list - must match what is used in Streamlit app
features = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'Contract_Two year', 'InternetService_Fiber optic'
]

X = df[features]
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "best_model.pkl")
print("✅ Model trained and saved as 'best_model.pkl'")
