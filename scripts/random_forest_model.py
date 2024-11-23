import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import X_train, X_test, y_train, y_test
from sklearn.ensemble import RandomForestClassifier

# Step 1: Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Step 2: Make predictions
y_pred_rf = rf_model.predict(X_test)

# Step 3: Get classification report and confusion matrix
report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Step 4: Convert to DataFrame and export to Excel
df_report_rf = pd.DataFrame(report_rf).transpose()
df_report_rf.to_excel("rf_classification_report.xlsx", sheet_name="Random Forest Report")

# Export confusion matrix to Excel
df_conf_matrix_rf = pd.DataFrame(conf_matrix_rf)
df_conf_matrix_rf.to_excel("rf_confusion_matrix.xlsx", sheet_name="Random Forest Confusion Matrix")

print("Random Forest results have been exported to Excel.")
