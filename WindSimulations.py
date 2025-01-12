import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def run_wind_simulation():
    print("Running wind simulation...")

    #define monthly wind production and stdev
    months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    wind_capacity = 0.0029 * months**3 + 0.0321 * months**2 - 0.9065 * months + 13.076
    wind_std_dev = 0.0031 * months**3 - 0.0038 * months**2 - 0.473 * months + 5.7388

    #interpolate daily data and stdev
    days = np.linspace(1, 12, 365) #spread monthly data across 365 days
    wind_capacity_daily = interp1d(months, wind_capacity, kind='cubic')(days)
    wind_std_dev_daily = interp1d(months, wind_std_dev, kind='cubic')(days)

    #gen 5 simulations
    simulations = {}
    for i in range(1, 6):
        simulations[f"Simulation_{i}"] = np.random.normal(wind_capacity_daily, wind_std_dev_daily)

    #replace neg values w/0...
    for key in simulations:
        simulations[key] = np.maximum(simulations[key], 0)

    #convert to dataframe, save to csv
    df = pd.DataFrame(simulations)
    df.to_csv("simulated_wind_production.csv", index=False)

    #plot 5sims
    plt.figure(figsize=(10, 5))
    for i in range(1, 6):
        plt.plot(range(1, 366), simulations[f"Simulation_{i}"], label=f"Simulation {i}")
    plt.xlabel("Day of the Year")
    plt.ylabel("Simulated Wind Production (Knots^2)")
    plt.title("Simulated Daily Wind Production (5 Simulations)")
    plt.legend()
    plt.show()

    print("Wind simulation completed. Dataset saved to 'simulated_wind_production.csv'.")