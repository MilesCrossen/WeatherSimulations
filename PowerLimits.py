import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure the TkAgg backend is used
import matplotlib
matplotlib.use("TkAgg")

def analyze_power_limits_derivative():
    """
    Analyze the maximum power to add or remove from the grid for each simulation based on the derivative of the energy balance curve.
    """
    print("Analyzing power limits using derivatives for each simulation...")

    #load
    wind_data = pd.read_csv("simulated_wind_production.csv")
    solar_data = pd.read_csv("simulated_solar_radiation.csv")
    demand_data = pd.read_csv("simulated_demand.csv")

    #extract demand
    demand = demand_data["Average_Demand_MW"]

    #prep for multiple sims (5)
    results = []
    for i in range(1, 6): #naming sims
        #calc total production for each sim
        production = wind_data[f"Simulation_{i}"] + solar_data[f"Simulation_{i}"]
        power_difference = production - demand

        #find derivative (input/output difference)
        derivative = np.gradient(power_difference)

        #find maximum power to add (negative derivative) and remove (positive derivative)
        max_power_to_add = abs(derivative[derivative < 0].min()) #largest negative value
        max_power_to_remove = derivative[derivative > 0].max() #largest positive value

        results.append((f"Simulation {i}", max_power_to_add, max_power_to_remove))

        print(f"{f'Simulation {i}':<15} | Max Power to Add: {max_power_to_add:.2f} MW | Max Power to Remove: {max_power_to_remove:.2f} MW")

    #plot derivatives for visual analysis
    plt.figure(figsize=(10, 5))
    for i in range(1, 6):
        production = wind_data[f"Simulation_{i}"] + solar_data[f"Simulation_{i}"]
        power_difference = production - demand
        derivative = np.gradient(power_difference)
        plt.plot(range(1, 366), derivative, label=f"Simulation {i}")

    plt.axhline(0, color="black", linestyle="--", linewidth=1, label="No Power Difference")
    plt.xlabel("Day of the Year")
    plt.ylabel("Power Difference (MW)")
    plt.title("Instantaneous Power Difference (Derivative) for Each Simulation")
    plt.legend()
    plt.show()

    print("Power limits analysis completed.")

#ensure it runs when main is executed (Doesnt work though for some reason, as of 16/1)
def main():
    analyze_power_limits_derivative()

if __name__ == "__main__":
    main()
