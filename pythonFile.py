# make sure the following packages are downloaded and then run the command to import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from IPython.display import display

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

# Compute initial capital per worker (k0)
df['k_t'] = df['capital'] / df['hrs_work']

# Simulate  path of k_t
num_years = 100  # Simulate for 100 years
k_path = [df['k_t'].iloc[0]]  # Start with initial k_t
A_t = df['A_t'].iloc[0]  # Use initial TFP

if not np.issubdtype(df['date'].dtype, np.datetime64):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Get the starting year from the dataset
start_year = df['date'].dt.year.iloc[0]
years = [start_year + t for t in range(num_years)]
for t in range(num_years - 1):
    k_next = savings_rate * A_t * (k_path[-1] ** alpha) + (1 - delta) * k_path[-1]
    k_path.append(k_next)

# Compute aggregate output path Y_t
Y_path = [A_t * (k ** alpha) * df['hrs_work'].iloc[0] for k in k_path]

yt_df = pd.DataFrame({'Year': years, 'Simulated_Yt': Y_path})
yt_df.to_csv("simulated_Yt.csv", index=False)
# Plot the path of k_t and Y_t
plt.figure(figsize=(10,5))
plt.plot(years, Y_path, label='Aggregate Output (Y_t)', color='b', linestyle='dashed')
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Aggregate Output')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(years, k_path, label='Capital per Worker (k_t)', color='g')
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Simulated Paths of Capital per Worker')
plt.legend()
plt.grid()
plt.show()

df.to_csv("tfp_results.csv", index=False)

# Part 3

# Reload dataset (assuming previous computations are lost)
# Since the file path is not available, I will define sample TFP values for computation
A_t_values = np.linspace(0.8, 1.2, 10)  # Simulating a range of TFP values

# Define function to compute steady-state capital k*
def steady_state_k(s, A, alpha, delta):
    """Computes steady-state capital per worker (k*) using the Solow model equation."""
    return (savings_rate * A / delta) ** (1 / (1 - alpha))

# Compute steady-state capital per worker for different TFP values
steady_state_k_values = [steady_state_k(savings_rate, A_t, alpha, delta) for A_t in A_t_values]

# Create DataFrame to store results
steady_state_df = pd.DataFrame({
    "TFP (A_t)": A_t_values,
    "Steady-State k*": steady_state_k_values
})

# Display the dataframe
display(steady_state_df)


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

