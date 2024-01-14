# Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import time

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
start_time = time.time()
# Define individual models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=42)
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)

# Create a Voting Classifier
voting_model = VotingClassifier(estimators=[(
    'RandomForest', rf_model), ('XGBoost', xgb_model), ('AdaBoost', ada_model)], voting='soft')

# Stacking ensemble
stacking_models = [
    ('RandomForest', rf_model),
    ('XGBoost', xgb_model),
    ('AdaBoost', ada_model)
]
stacking_model = StackingClassifier(
    estimators=stacking_models, final_estimator=LogisticRegression())

# Evaluate individual models
models = [rf_model, xgb_model, ada_model, voting_model, stacking_model]
model_names = ["Random Forest", "XGBoost", "AdaBoost", "Voting", "Stacking"]

# for model, model_name in zip(models, model_names):
#     start_time = time.time()
#     kfold = KFold(n_splits=5)
#     results = cross_val_score(model, X_train, y_train,
#                               cv=kfold, scoring='accuracy')
#     end_time = time.time()
#     print(f"{model_name} - Mean Accuracy: {results.mean()}")
  # 记录程序结束运行时间
start_time = time.time()

# Train and evaluate the Voting Classifier
voting_model.fit(X_train, y_train)
y_pred = voting_model.predict(X_test)
voting_accuracy = accuracy_score(y_test, y_pred)

end_time = time.time()
print('cost %f second' % (end_time - start_time))
print(f"Voting Classifier Accuracy: {voting_accuracy}")

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
f1 = f1_score(y_test, y_pred, average='macro')
print("F1-Score:", f1)


start_time = time.time()
# Train and evaluate the Stacking Classifier
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
stacking_accuracy = accuracy_score(y_test, y_pred)
print(f"Stacking Classifier Accuracy: {stacking_accuracy}")
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
f1 = f1_score(y_test, y_pred, average='macro')
print("F1-Score:", f1)
end_time = time.time()
print('cost %f second' % (end_time - start_time))