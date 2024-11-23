import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import X_train, X_test, y_train, y_test
from sklearn.svm import SVC
import time

# Use a smaller subset for quick testing
X_train_sample = X_train[:1000]
y_train_sample = y_train[:1000]

# Track the time taken
start_time = time.time()
print("Step 1: Training the SVM model...")

# Use SVC without probability for faster training
svm_model = SVC(kernel='linear', C=1.0, class_weight='balanced', probability=False)
svm_model.fit(X_train_sample, y_train_sample)

print(f"Model training complete. Time taken: {time.time() - start_time:.2f} seconds")

print("Step 2: Making predictions...")
y_pred_svm = svm_model.predict(X_test)

print("Predictions complete. Step 3: Generating report and confusion matrix...")
report_svm = classification_report(y_test, y_pred_svm, output_dict=True, zero_division=1)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

print("Exporting results to Excel...")
df_report_svm = pd.DataFrame(report_svm).transpose()
df_report_svm.to_excel("svm_classification_report.xlsx", sheet_name="SVM Report")

df_conf_matrix_svm = pd.DataFrame(conf_matrix_svm)
df_conf_matrix_svm.to_excel("svm_confusion_matrix.xlsx", sheet_name="SVM Confusion Matrix")

print("SVM results have been exported to Excel.")
