import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#load CSV files
fourier_df = pd.read_csv("FourierResults.csv")
coefficients_df = pd.read_csv("coefficients.csv")

#extract columns
coefficients = coefficients_df["Coefficient"].values
average_fourier_equations = fourier_df["Fourier_Avg"].dropna().astype(str).tolist()
stddev_fourier_equations = fourier_df["Fourier_Std"].dropna().astype(str).tolist()

#print lengths (debugging purposes...)
print(f"Length of coefficients: {len(coefficients)}")
print(f"Length of average Fourier equations: {len(average_fourier_equations)}")
print(f"Length of stddev Fourier equations: {len(stddev_fourier_equations)}")

#check data lengths match
assert len(coefficients) == len(average_fourier_equations) == len(stddev_fourier_equations), "Mismatch in data lengths"


#convert raw eqns into executable functions
def parse_fourier_equation(raw_equation):
    #converting fourier eqn into a usable eqn
    equation = raw_equation.split("=")[-1].strip()
    equation = equation.replace("t", "days")
    equation = equation.replace("cos", "np.cos").replace("π", "np.pi")
    equation = equation.replace("2np.pi", "2 * np.pi")
    return lambda days: eval(equation, {"np": np, "days": days})


#convert Fourier equations into executable functions
basis_functions = [parse_fourier_equation(eq) for eq in average_fourier_equations]
stddev_functions = [parse_fourier_equation(eq) for eq in stddev_fourier_equations]

#linear spacing between days
days = np.linspace(0, 365, 365)

production_factor = 1.00 #production scale

#compute fourier avg output
Y_avg = production_factor * np.sum([coef * func(days) for coef, func in zip(coefficients, basis_functions)], axis=0)

#compute fourier stdeviation
Y_std = production_factor * np.sum([coef * func(days) for coef, func in zip(coefficients, stddev_functions)], axis=0)


#load demand func
def demand_function(days):
    return (4454.708901 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            202.979616 * np.cos(2 * np.pi * -0.002740 * days + 0.261259) +
            202.979616 * np.cos(2 * np.pi * 0.002740 * days + -0.261259) +
            133.571826 * np.cos(2 * np.pi * -0.142466 * days + -2.989836) +
            133.571826 * np.cos(2 * np.pi * 0.142466 * days + 2.989836))


#daily demand
daily_demand = demand_function(days)

#number of monte carlo sims
num_simulations = 5

#store max surplus and deficit values
print("\nCalculating energy surplus/deficit in MWh for multiple simulations...\n")
plt.figure(figsize=(12, 6))

for i in range(num_simulations):
    noise = np.random.normal(scale=Y_std) #apply standard deviation as noise
    simulated_output = np.maximum(Y_avg + noise, 0) #ensure values do not go below 0

    #compute cumulative surplus/deficit
    cumulative_balance = np.cumsum(simulated_output - daily_demand)

    #print max surplus and deficit
    max_surplus = np.max(cumulative_balance)
    max_deficit = np.min(cumulative_balance)
    print(
        f"For Simulation {i + 1}: Maximum surplus is {max_surplus:.2f} MWh, maximum deficit is {max_deficit:.2f} MWh.")

    #plot cumulative energy balance
    plt.plot(days, cumulative_balance, label=f"Simulation {i + 1}")

print("\nAll simulations completed.\n")

#plot formatting
plt.axhline(y=0, color='black', linestyle='dashed')  # Reference line for balance
plt.xlabel("Day of the Year")
plt.ylabel("Cumulative Energy Balance (MWh)")
plt.title(
    f"Cumulative Energy Surplus/Deficit Over the Year for {num_simulations} Simulations ({production_factor}x Production)")
plt.legend()
plt.grid()
plt.show()