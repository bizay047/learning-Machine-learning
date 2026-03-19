import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris dataset
df = sns.load_dataset("iris")

# Plot distribution of a feature (sepal_length)
plt.hist(df["sepal_length"], bins=15, edgecolor="black")
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
