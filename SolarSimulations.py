import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from optimisation import days_to_months_exact


def run_solar_simulation():
    """
    Simulates daily solar radiation for a full year, generates visualizations for the 5 final simulations,
    and saves the simulated dataset to a CSV file.
    """
    print("Running solar simulation...")

    #define monthly solar rad (avg + stdev)
    months = np.array([0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    #polynomial equations are inputted directly
    avg_solar_radiation = 2.9668 * months**3 - 110.07 * months**2 + 966.25 * months - 848.84
    std_solar_radiation = 1.1211 * months**3 - 39.068 * months**2 + 331.93 * months - 236.89

    #interpolate vals across 365 d ays
    days = np.arange(1, 366) #generate days 1-366
    exact_months = days_to_months_exact(days) #convert days to exact fractional months
    exact_months = np.clip(exact_months, 0, 12) #clamp exact months to the interpolation range

    avg_solar_radiation_daily = interp1d(months, avg_solar_radiation, kind='linear')(exact_months)
    std_solar_radiation_daily = interp1d(months, std_solar_radiation, kind='linear')(exact_months)

    std_solar_radiation_daily = np.maximum(std_solar_radiation_daily, 0)

    #generate 5 simulations
    simulations = {}
    for i in range(1, 6):
        simulations[f"Simulation_{i}"] = np.random.normal(avg_solar_radiation_daily, std_solar_radiation_daily)

    #replace neg values w/0 because wind and solar can never be negative
    for key in simulations:
        simulations[key] = np.maximum(simulations[key], 0)

    #convert to dataframe and save to csv
    df = pd.DataFrame(simulations)
    df.to_csv("simulated_solar_radiation.csv", index=False)

    #plot 5 sims
    plt.figure(figsize=(10, 5))
    for i in range(1, 6):
        plt.plot(range(1, 366), simulations[f"Simulation_{i}"], label=f"Simulation {i}")
    plt.xlabel("Day of the Year")
    plt.ylabel("Simulated Solar Radiation (J/cm^2)")
    plt.title("Simulated Daily Solar Radiation (5 Simulations)")
    plt.legend()
    plt.show()

    print("Solar simulation completed. Dataset saved to 'simulated_solar_radiation.csv'.")