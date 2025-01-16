import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Fix for PyCharm compatibility
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
from optimisation import days_to_months_exact

def run_demand_simulation():
    """
    Plots and analyzes the electricity demand for a year based on the provided polynomial equation.
    """
    print("Running demand simulation...")

    #define monthly demand data in months
    months = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    avg_demand = 1.8062 * months**3 - 9.8403 * months**2 - 156.1 * months + 5084.4

    #interpolate data across 365 days
    #days = np.linspace(1, 12, 365) #spread data across 365 d ays using linear space
    #could be improved later on by specifying particular month lengths as is done in other py files of
    #this programme. This approximates months as identical lengths
    #avg_demand_daily = interp1d(months, avg_demand, kind='cubic')(days)

    days = np.arange(1, 366)
    exact_months = days_to_months_exact(days)
    exact_months = np.clip(exact_months, 0, 12)

    avg_demand_daily = interp1d(months, avg_demand, kind = 'cubic')(exact_months)

    # Plot average demand over the year
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 366), avg_demand_daily, label="Daily Average Demand", color="green")
    plt.xlabel("Day of the Year")
    plt.ylabel("Electricity Demand (MW)")
    plt.title("Daily Electricity Demand for a Full Year")
    plt.legend()
    plt.show()

    #save daily demand to simulated_demand.csv
    demand_data = pd.DataFrame({
        "Day_of_Year": days,
        "Average_Demand_MW": avg_demand_daily
    })
    demand_data.to_csv("simulated_demand.csv", index=False)

    print("Demand simulation completed. Dataset saved to 'simulated_demand.csv'.")