import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix

file_path = '/content/Copper_Set (1).xlsx'  # Replace with your file path
data = pd.ExcelFile(file_path)
df = data.parse('Result 1')

df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
df['item_date'] = pd.to_datetime(df['item_date'], errors='coerce')
df['delivery date'] = pd.to_datetime(df['delivery date'], errors='coerce')
df['material_ref'] = df['material_ref'].replace(to_replace=r'^00000.*', value=None, regex=True)
df.fillna({
    'country': df['country'].mode()[0],
    'application': df['application'].mode()[0],
    'thickness': df['thickness'].median(),
    'width': df['width'].median(),
    'material_ref': 'Unknown'
}, inplace=True)
df.dropna(subset=['selling_price', 'status'], inplace=True)

label_encoders = {}
for col in ['country', 'application', 'material_ref', 'product_ref', 'item type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

regression_target = 'selling_price'
classification_target = 'status'
X = df.drop(columns=[regression_target, classification_target, 'id', 'item_date', 'delivery date', 'customer'])

status_encoder = LabelEncoder()
df[classification_target] = status_encoder.fit_transform(df[classification_target])
y_reg = df[regression_target]
y_class = df[classification_target]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)
X_train_class_scaled = scaler.fit_transform(X_train_class)
X_test_class_scaled = scaler.transform(X_test_class)

X_train_reg_scaled = pd.DataFrame(X_train_reg_scaled).fillna(0).values
X_test_reg_scaled = pd.DataFrame(X_test_reg_scaled).fillna(0).values
X_train_class_scaled = pd.DataFrame(X_train_class_scaled).fillna(0).values
X_test_class_scaled = pd.DataFrame(X_test_class_scaled).fillna(0).values

pca = PCA(n_components=0.95) 
X_train_reg_pca = pca.fit_transform(X_train_reg_scaled)
X_test_reg_pca = pca.transform(X_test_reg_scaled)
X_train_class_pca = pca.fit_transform(X_train_class_scaled)
X_test_class_pca = pca.transform(X_test_class_scaled)

regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train_reg_pca, y_train_reg)
y_pred_reg = regressor.predict(X_test_reg_pca)
reg_mse = mean_squared_error(y_test_reg, y_pred_reg)
print("Regression Model MSE:", reg_mse)

classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_class_pca, y_train_class)
y_pred_class = classifier.predict(X_test_class_pca)
class_acc = accuracy_score(y_test_class, y_pred_class)
print("Classification Model Accuracy:", class_acc)
print("Classification Report:\n", classification_report(y_test_class, y_pred_class))

conf_matrix = confusion_matrix(y_test_class, y_pred_class)
print("Confusion Matrix:\n", conf_matrix)

def predict_outcome(classifier, regressor, scaler, pca, user_data):
    user_data_scaled = scaler.transform(user_data)
    user_data_pca = pca.transform(user_data_scaled)
    prediction_class = classifier.predict(user_data_pca)
    prediction_price = regressor.predict(user_data_pca)
    return prediction_class, prediction_price

user_data = np.array([[30153963, 30, 6, 28, 952, 628377, 5.9, -0.96, 6.46, 1, 4, 2021]])
classification_prediction, price_prediction = predict_outcome(classifier, regressor, scaler, pca, user_data)

if classification_prediction[0] == 1:
    print("Outcome: Win")
else:
    print("Outcome: Lose")

print(f"Predicted Price: {price_prediction[0]}")
