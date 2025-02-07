# make sure the following packages are downloaded and then run the command to import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("fredgraph.csv")

# Define capital elasticity
alpha = 0.03

# Convert date column to datetime format and extract only the year
df["date"] = pd.to_datetime(df["date"]).dt.year

# Set base year
base_year = 2017

# Compute log(A_t) using the residual
df["log_A_t"] = np.log(df["rGDP"]) - alpha * np.log(df["capital"]) - (1 - alpha) * np.log(df["hrs_work"])

# Convert log(A_t) to A_t
df["A_t"] = np.exp(df["log_A_t"])

# Normalize A_t to 1 in the base year
base_A = df.loc[df["date"] == base_year, "A_t"].values[0]
df["A_t"] = df["A_t"] / base_A

# Display the results in the console with date and value of A_t
print(df[["date", "A_t"]])

# Save results to a CSV file
df.to_csv("solow_results.csv", index=False)

# Plot A_t
plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["A_t"], marker="o", linestyle="-", color="b", label="TFP (A_t)")
plt.xlabel("Year") # x label
plt.ylabel("Normalized Total Factor Productivity (A_t)") # y label
plt.title("TFP (A_t) Over Time") # title of plot
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Part 2
