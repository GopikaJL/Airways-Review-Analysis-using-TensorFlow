from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Training Accuracy :", model.score(X_train, y_train))
print("Validation Accuracy :", model.score(X_test, y_test))
cm = confusion_matrix(y_test, y_pred)
print(cm)
!pip install xgboost
from xgboost import XGBClassifier
model = XGBClassifier(n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Training Accuracy :", model.score(X_train, y_train))
print("Validation Accuracy :", model.score(X_test, y_test))
cm = confusion_matrix(y_test, y_pred)
print(cm)
