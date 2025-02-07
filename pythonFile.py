#make sure imported packages are downloaded
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("fredgraph.csv")

# Define capital elasticity
alpha = 0.03

# Convert date column to datetime format and extract only the year
df["date"] = pd.to_datetime(df["date"]).dt.year

# Set base year
base_year = 2017

# Compute log(A_t)
df["log_A_t"] = np.log(df["rGDP"]) - alpha * np.log(df["capital"]) - (1 - alpha) * np.log(df["hrs_work"])

# Convert log(A_t) to A_t
df["A_t"] = np.exp(df["log_A_t"])

# Normalize A_t to 1 in the base year
base_A = df.loc[df["date"] == base_year, "A_t"].values[0]
df["A_t"] = df["A_t"] / base_A

# Display the results
print(df[["date", "A_t"]])

# Save results to a CSV file
df.to_csv("solow_results.csv", index=False)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["A_t"], marker="o", linestyle="-", color="b", label="TFP (A_t)")
plt.axhline(y=1, color="r", linestyle="--", label=f"Base Year {base_year}")
plt.xlabel("Year")
plt.ylabel("Normalized Total Factor Productivity (A_t)")
plt.title("TFP (A_t) Over Time")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()