import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_wind_simulation():
    print("Running wind simulation...")

    days = np.arange(1, 366)

    #average wind speed squared equation
    wind_capacity_daily = (
            125.528148 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            18.344844 * np.cos(2 * np.pi * -0.002732 * days + 0.249556) +
            18.344844 * np.cos(2 * np.pi * 0.002732 * days + -0.249556) +
            3.719645 * np.cos(2 * np.pi * 0.117486 * days + -0.565647) +
            3.719645 * np.cos(2 * np.pi * -0.117486 * days + 0.565647)
    )

    #standard deviation of wind speed squared equation
    wind_std_dev_daily = (
            95.800535 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            17.261051 * np.cos(2 * np.pi * -0.002732 * days + 0.362807) +
            17.261051 * np.cos(2 * np.pi * 0.002732 * days + -0.362807) +
            3.840067 * np.cos(2 * np.pi * 0.117486 * days + -0.557912) +
            3.840067 * np.cos(2 * np.pi * -0.117486 * days + 0.557912)
    )

    #plotting average and standard deviation of wind speed squared
    plt.figure(figsize=(10, 6))
    plt.plot(days, wind_capacity_daily, label="Average Wind Speed Squared", color="blue")
    plt.plot(days, wind_std_dev_daily, label="Standard Deviation", color="orange")
    plt.xlabel("Day of the Year")
    plt.ylabel("Wind Speed Squared (Knots^2)")
    plt.title("Average and Standard Deviation of Wind Speed Squared")
    plt.legend()
    plt.grid()
    plt.show()

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