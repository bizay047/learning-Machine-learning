import pandas as pd
import seaborn as sns

# Load Iris dataset using seaborn and convert to pandas DataFrame
df = pd.read_csv("Iris.csv")

print("Dataset Loaded Successfully!")
print(df.head(149))     # show first 5 rows
# print("\nShape of dataset:", df.shape)
