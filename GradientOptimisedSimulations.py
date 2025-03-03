import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg') #tkagg fixes rendering errors for me
import matplotlib.pyplot as plt

#load CSV files
fourier_df = pd.read_csv("FourierResults.csv")
coefficients_df = pd.read_csv("coefficients.csv")

#remove any fully empty or corrupted rows
fourier_df = fourier_df.dropna(subset=["File", "Column"])
coefficients_df = coefficients_df.dropna()

#ensure all column values are treated as strings to prevent errors
fourier_df["Column"] = fourier_df["Column"].astype(str)

#extract installation type from FourierResults.csv (skip empty rows)
installation_types = []
for col in fourier_df["Column"]:
    if col.strip(): #ignore empty/invalid rows
        installation_types.append("Wind" if "wdsp^3" in col else "Solar")

#ensure the installation types list matches the expected data length
if len(installation_types) != len(coefficients_df):
    raise ValueError(f"Mismatch in data lengths: "
                     f"{len(coefficients_df)} coefficients vs {len(installation_types)} installations")

#extract columns
coefficients = coefficients_df["Coefficient"].values
average_fourier_equations = fourier_df["Fourier_Avg"].dropna().astype(str).tolist()
stddev_fourier_equations = fourier_df["Fourier_Std"].dropna().astype(str).tolist()

#ensure all extracted lists are the same length
assert len(coefficients) == len(average_fourier_equations) == len(stddev_fourier_equations) == len(installation_types), \
    "Mismatch in data lengths"

#convert raw Fourier equations into executable functions
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

production_factor = 1.3 #production scale

#compute fourier avg output separately for Wind and Solar
Y_wind_avg = production_factor * np.sum(
    [coef * func(days) for coef, func, install_type in zip(coefficients, basis_functions, installation_types) if install_type == "Wind"], axis=0)

Y_solar_avg = production_factor * np.sum(
    [coef * func(days) for coef, func, install_type in zip(coefficients, basis_functions, installation_types) if install_type == "Solar"], axis=0)

#compute standard deviation separately for Wind and Solar (including production_factor)
Y_wind_std = production_factor * np.sum(
    [coef * func(days) for coef, func, install_type in zip(coefficients, stddev_functions, installation_types) if install_type == "Wind"], axis=0)

Y_solar_std = production_factor * np.sum(
    [coef * func(days) for coef, func, install_type in zip(coefficients, stddev_functions, installation_types) if install_type == "Solar"], axis=0)

#load demand function
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
print("\nCalculating energy surplus/deficit in MWd for multiple simulations...\n")
plt.figure(figsize=(12, 6))

for i in range(num_simulations):
    wind_noise = np.random.normal(scale=Y_wind_std) #apply standard deviation as noise to wind
    solar_noise = np.random.normal(scale=Y_solar_std) #apply standard deviation as noise to solar

    wind_output = np.maximum(Y_wind_avg + wind_noise, 0)  #ensure wind values do not go below 0
    solar_output = np.maximum(Y_solar_avg + solar_noise, 0) #ensure solar values do not go below 0

    total_output = wind_output + solar_output #total energy production

    #compute cumulative surplus/deficit
    #cumulative_balance = np.cumsum(total_output - daily_demand)

    storage_efficiency = 0.4
    #find daily changes in cumulative balance
    daily_changes = total_output - daily_demand
    daily_changes = np.where(daily_changes > 0, daily_changes * storage_efficiency, daily_changes)
    #multiplied by storage efficiency if positive change detected

    cumulative_balance = np.cumsum(daily_changes) #cumulative sum

    #find max surplus change (max energy injection needed)
    max_surplus_change = np.max(daily_changes)

    #find max deficit change (max energy withdrawal needed)
    max_deficit_change = np.min(daily_changes)

    #print max surplus and deficit
    max_surplus = np.max(cumulative_balance)
    max_deficit = np.min(cumulative_balance)
    print(
        f"For Simulation {i + 1}: Maximum surplus is {max_surplus:.2f} MWd, maximum deficit is {max_deficit:.2f} MWd.")
    print(
        f"               Max daily average surplus: {max_surplus_change:.2f} MW, Max daily average deficit: {max_deficit_change:.2f} MW."
    )

    #plot cumulative energy balance
    plt.plot(days, cumulative_balance, label=f"Simulation {i + 1}")

print("\nAll simulations completed.\n")

#plot formatting
plt.axhline(y=0, color='black', linestyle='dashed') #balance line
plt.xlabel("Day of the Year")
plt.ylabel("Cumulative Energy Balance (MWd)")
plt.title(
    f"Cumulative Energy Surplus/Deficit Over the Year for {num_simulations} Simulations ({production_factor}x Production)")
plt.legend()
plt.grid()
plt.show()