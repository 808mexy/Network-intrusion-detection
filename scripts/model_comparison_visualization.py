import matplotlib.pyplot as plt
import numpy as np

models = ['SVM', 'Random Forest', 'MLP']
precision = [0.99, 0.99, 0.99]  # Replace with actual values
recall = [0.99, 0.99, 0.99]
f1_score = [0.99, 0.99, 0.99]

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots()
ax.bar(x - width, precision, width, label='Precision')
ax.bar(x, recall, width, label='Recall')
ax.bar(x + width, f1_score, width, label='F1-Score')

ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

plt.show()
