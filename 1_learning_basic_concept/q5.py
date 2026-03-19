import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset("iris")

# Compute correlation matrix for numeric columns
corr = df.corr(numeric_only=True)

# Plot correlation heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Iris Features")
plt.tight_layout()
plt.show()
