# Step 1: Import libraries and load data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
train = pd.read_csv(r"C:\Users\anany\Downloads\archive (1)\train.csv")
test = pd.read_csv(r"C:\Users\anany\Downloads\archive (1)\test.csv")
sample = pd.read_csv(r"C:\Users\anany\Downloads\archive (1)\sample_submission.csv")

print(train.info())

# Step 2: Handle missing values
for df in [train, test]:
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())


# Step 3: Encode categorical variables
combined = pd.concat([train.drop('Label', axis=1), test], axis=0)
combined = pd.get_dummies(combined)

train_encoded = combined.iloc[:len(train), :]
test_encoded = combined.iloc[len(train):, :]
train_encoded['Label'] = train['Label'].values

# Step 4: Normalize/standardize features
X = train_encoded.drop('Label', axis=1)
y = train_encoded['Label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_encoded)


# Step 5: Visualize and remove outliers (optional)
numeric_cols = X.columns[:2]  # Use first two numeric features for boxplots

for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=X[col])
    plt.title(f'Boxplot - {col}')
    plt.savefig(f"{col}_boxplot.png")
    plt.close()


def remove_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[col] >= lower) & (data[col] <= upper)]

# Skip this step for scaled data — applied only if using original

# Step 6: Train-test split and modeling
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"✅ Model trained. Validation Accuracy: {acc:.4f}")
# Step 7: Predict on test and create submission
predictions = model.predict(test_scaled)
submission = sample.copy()
submission['Label'] = predictions
submission.to_csv("submission.csv", index=False)
