import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATASET
# =========================
data = pd.read_csv("Book.csv")

# FEATURES & TARGET
X = data[['Temperature', 'Moisture', 'Humidity', 'Light']]
y = data['Risk']   # Low / Medium / High

# =========================
# ENCODE TARGET
# =========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)   # Low=0, Medium=1, High=2

# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# =========================
# XGBOOST MODEL
# =========================
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# =========================
# TRAIN
# =========================
model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# =========================
# SAVE MODEL
# =========================
model.save_model("xgb_risk_model.json")
