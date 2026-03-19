import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset("iris")

# Scatter plot: Sepal Length vs Petal Length
sns.scatterplot(
    data=df,
    x="sepal_length",
    y="petal_length",
    hue="species"        # color by species
)

plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.tight_layout()
plt.show()

