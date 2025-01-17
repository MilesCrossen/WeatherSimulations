import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

def run_demand_simulation():
    print("Running demand simulation...")

    #define days of the year
    days = np.arange(1, 366)

    #fourier transform equation for average demand
    avg_demand_daily = (
        4454.708901 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
        202.979616 * np.cos(2 * np.pi * -0.002740 * days + 0.261259) +
        202.979616 * np.cos(2 * np.pi * 0.002740 * days + -0.261259) +
        133.571826 * np.cos(2 * np.pi * -0.142466 * days + -2.989836) +
        133.571826 * np.cos(2 * np.pi * 0.142466 * days + 2.989836)
    )

    #plot average demand over the year
    plt.figure(figsize=(10, 5))
    plt.plot(days, avg_demand_daily, label="Daily Average Demand", color="green")
    plt.xlabel("Day of the Year")
    plt.ylabel("Electricity Demand (MW)")
    plt.title("Daily Electricity Demand for a Full Year")
    plt.legend()
    plt.grid(True)
    plt.show()

    #save daily demand to simulated_demand.csv
    demand_data = pd.DataFrame({
        "Day_of_Year": days,
        "Average_Demand_MW": avg_demand_daily
    })
    demand_data.to_csv("simulated_demand.csv", index=False)

    print("Demand simulation completed. Dataset saved to 'simulated_demand.csv'.")

#run the simulation
if __name__ == "__main__":
    run_demand_simulation()