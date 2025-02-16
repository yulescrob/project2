# make sure the following packages are downloaded and then run the command to import

#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from scipy.optimize import minimize

'''
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
#df.to_csv("solow_results.csv", index=False)

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
delta = 0.037 # depreciation
savings_rate = 0.038 # savings rate

# Extract base year values
base_data = df[df["date"] == base_year]
if base_data.empty:
    raise ValueError(f"Base year {base_year} not found in data.")

K0 = base_data["capital"].values[0]  # Initial capital stock
L0 = base_data["hrs_work"].values[0]  # Initial labor force
A0 = base_data["A_t"].values[0]  # TFP in base year
k0 = K0 / L0  # Initial capital per worker

# Simulate capital per worker over time
years = 200  # Number of periods to simulate
kt_path = [k0]

for t in range(1, years):
    kt_next = savings_rate * A0 * (kt_path[-1] ** alpha) + (1 - delta) * kt_path[-1]
    kt_path.append(kt_next)

    # Check for convergence to steady-state
    if abs(kt_path[-1] - kt_path[-2]) < 1e-6:  # Stopping condition
        print(f"Converged at year {t}")
        break

# Compute steady-state capital per worker
k_star = (savings_rate * A0 / delta) ** (1 / (1 - alpha))
print(f"Steady-state capital per worker: {k_star:.2f}")

# Plot capital per worker over time
plt.figure(figsize=(10, 5))
plt.plot(range(len(kt_path)), kt_path, marker="o", linestyle="-", color="b", label="Capital per Worker (k_t)")
plt.axhline(y=k_star, color="r", linestyle="--", label=f"Steady-State k* = {k_star:.2f}")
plt.xlabel("Years")
plt.ylabel("Capital per Worker (k_t)")
plt.title("Solow Model Simulation: Capital per Worker Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Calculating aggregate output
Yt_path = []

for t in range(1, years):
    # Update capital per worker
    kt_next = savings_rate * A0 * (kt_path[-1] ** alpha) + (1 - delta) * kt_path[-1]
    kt_path.append(kt_next)

    # Compute aggregate capital
    Kt = kt_next * L0  # K_t = k_t * L_t (assuming constant L_t)

    # Compute aggregate output Y_t
    Yt = A0 * (Kt ** alpha) * (L0 ** (1 - alpha))
    Yt_path.append(Yt)

    # Check for convergence
    if len(Yt_path) > 1 and abs(Yt_path[-1] - Yt_path[-2]) < 1e-6:
        print(f"Output converged at year {t}")
        break

plt.figure(figsize=(10, 5))
plt.plot(range(len(Yt_path)), Yt_path, marker="o", linestyle="-", color="g", label="Aggregate Output (Y_t)")
plt.xlabel("Years")
plt.ylabel("Output (Y_t)")
plt.title("Solow Model Simulation: Aggregate Output Over Time")
plt.legend()
plt.grid(True)
plt.show()


# Trying different

'''

# Part 4 - Optimization
def utility(params):
    c1, c2 = params
    return -(np.log(c1) + beta * np.log(c2))  # Negative function to use minimize

def constraint1(params):
    c1, s = params
    return w1 - c1 - s  # Budget constraint in period 1

def constraint2(params):
    c1, s = params
    c2 = s * (1 + r) # Second constraint
    return c2 - beta * (1 + r) * c1  # Derived relationship between c1 and c2

# Given parameters
beta = 0.95
r = 0.05
w1 = 100

# Initial guess
initial_guess = [w1 / 2, w1 / 2]

# Constraints
eq_constraints = ({'type': 'eq', 'fun': constraint1},
                  {'type': 'eq', 'fun': constraint2})

# Solve optimization
solution = minimize(utility, initial_guess, constraints=eq_constraints, bounds=((1e-5, w1), (1e-5, w1)))

c1_opt, s_opt = solution.x
c2_opt = s_opt * (1 + r)

print(f"Optimal c1: {c1_opt:.4f}")
print(f"Optimal c2: {c2_opt:.4f}")

# Loops to see changes in beta and r
beta_values = [0.90, 0.95, 1.00]  # Different discount factors
r_values = [0.03, 0.05, 0.07]  # Different interest rates
w1 = 100
w2_values = [0, 100, 10000]  # Different future wealth values

for beta in beta_values:
    for r in r_values:
        for w2 in w2_values:
            # Initial guess
            initial_guess = [w1 / 2, w1 / 2]

            # Constraints
            eq_constraints = ({'type': 'eq', 'fun': constraint1},
                              {'type': 'eq', 'fun': constraint2})

            # Solve optimization
            solution = minimize(utility, initial_guess, constraints=eq_constraints, bounds=((1e-5, w1), (1e-5, w1)))

            c1_opt, s_opt = solution.x
            c2_opt = s_opt * (1 + r) + w2

            print(f"beta: {beta}, r: {r}, w2: {w2}")
            print(f"Optimal c1: {c1_opt:.4f}, Optimal c2: {c2_opt:.4f}\n")

