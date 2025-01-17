import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_solar_simulation():
    print("Running solar simulation...")

    #generate days 1-366
    days = np.arange(1, 366)

    #average solar radiation equation (fourier transform)
    avg_solar_radiation_daily = (
            974.988895 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            420.293533 * np.cos(2 * np.pi * -0.002732 * days + 2.903370) +
            420.293533 * np.cos(2 * np.pi * 0.002732 * days + -2.903370) +
            30.446476 * np.cos(2 * np.pi * -0.005464 * days + -1.495144) +
            30.446476 * np.cos(2 * np.pi * 0.005464 * days + 1.495144)
    )

    #standard deviation solar radiation equation (fourier transform)
    std_solar_radiation_daily = (
            356.043602 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            136.183315 * np.cos(2 * np.pi * -0.002732 * days + 2.865705) +
            136.183315 * np.cos(2 * np.pi * 0.002732 * days + -2.865705) +
            13.607234 * np.cos(2 * np.pi * -0.005464 * days + -1.953777) +
            13.607234 * np.cos(2 * np.pi * 0.005464 * days + 1.953777)
    )

    #ensure standard deviation is non-negative
    std_solar_radiation_daily = np.maximum(std_solar_radiation_daily, 0)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 366), avg_solar_radiation_daily, label="Average Solar Radiation", color="blue")
    plt.plot(range(1, 366), std_solar_radiation_daily, label="Standard Deviation", color="orange")
    plt.xlabel("Day of the Year")
    plt.ylabel("Radiation (J/cm^2)")
    plt.title("Average and Standard Deviation of Solar Radiation")
    plt.legend()
    plt.grid()
    plt.show()

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