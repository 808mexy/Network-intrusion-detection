import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import X_train, X_test, y_train, y_test
from sklearn.neural_network import MLPClassifier

# Step 1: Train the MLP model
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp_model.fit(X_train, y_train)

# Step 2: Make predictions
y_pred_mlp = mlp_model.predict(X_test)

# Step 3: Get classification report and confusion matrix
report_mlp = classification_report(y_test, y_pred_mlp, output_dict=True)
conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)

# Step 4: Convert to DataFrame and export to Excel
df_report_mlp = pd.DataFrame(report_mlp).transpose()
df_report_mlp.to_excel("mlp_classification_report.xlsx", sheet_name="MLP Report")

# Export confusion matrix to Excel
df_conf_matrix_mlp = pd.DataFrame(conf_matrix_mlp)
df_conf_matrix_mlp.to_excel("mlp_confusion_matrix.xlsx", sheet_name="MLP Confusion Matrix")

print("MLP results have been exported to Excel.")
