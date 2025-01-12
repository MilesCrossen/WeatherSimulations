from WindSimulations import run_wind_simulation
from SolarSimulations import run_solar_simulation
from DemandSimulations import run_demand_simulation
from optimisation import optimize_coefficients
import pandas as pd
import matplotlib.pyplot as plt

production_factor = 1.05

def compare_multiple_simulations(w_opt, s_opt):
    #compares running sum of 5 simulations of scaled optimized power production and plots
    print("Comparing multiple simulations of scaled optimized power production with demand...")

    #loads simulated data
    wind_data = pd.read_csv("simulated_wind_production.csv")
    solar_data = pd.read_csv("simulated_solar_radiation.csv")
    demand_data = pd.read_csv("simulated_demand.csv")

    #extract data
    demand = demand_data["Average_Demand_MW"]
    running_sum_demand = demand.cumsum()

    #prepare for multiple simulations
    simulations = []
    for i in range(1, 6): #names simulations Simulation_1 to Simulation_5
        #apply the scaled optimization equation (multiplying by production factor)
        optimized_production = (
            production_factor * (s_opt * solar_data[f"Simulation_{i}"] + w_opt * wind_data[f"Simulation_{i}"])
        )
        running_sum_production = optimized_production.cumsum() #running sum is cumulative sum so far
        simulations.append((f"Simulation {i}", running_sum_production))

    #plot all simulations and demand
    plt.figure(figsize=(10, 5))
    for label, running_sum_production in simulations:
        plt.plot(range(1, 366), running_sum_production, label=label)
    plt.plot(range(1, 366), running_sum_demand, label="Running Sum of Electricity Demand", color="black", linewidth=2)
    plt.xlabel("Day of the Year")
    plt.ylabel("Cumulative Power (MW)")
    plt.title("Comparison of Running Sums of 5 Scaled Simulations and Electricity Demand")
    plt.legend()
    plt.show()

def compare_multiple_simulations_mwh(w_opt, s_opt):
    #compares cumulative surplus/deficit for each simulation w/baseline demand
    print("Calculating energy surplus/deficit in MWh for multiple simulations...")

    #load simulated data
    wind_data = pd.read_csv("simulated_wind_production.csv")
    solar_data = pd.read_csv("simulated_solar_radiation.csv")
    demand_data = pd.read_csv("simulated_demand.csv")

    #extract demand data
    demand = demand_data["Average_Demand_MW"]

    #prepare for multiple simulations
    simulations = []
    for i in range(1, 6):
        #apply scaled optimization equation
        optimized_production = (
            production_factor * (s_opt * solar_data[f"Simulation_{i}"] + w_opt * wind_data[f"Simulation_{i}"])
        )
        daily_difference = optimized_production - demand
        daily_difference_mwh = daily_difference * 24 #converts to MWh by changing timeframe
        #from days to hours (i.e. multiplying by 24)
        cumulative_mwh = daily_difference_mwh.cumsum()
        simulations.append((f"Simulation {i}", cumulative_mwh))

    #plots all cumulative results
    plt.figure(figsize=(10, 5))
    for label, cumulative_mwh in simulations:
        plt.plot(range(1, 366), cumulative_mwh, label=label)
    plt.axhline(0, color="black", linestyle="--", linewidth=1) #reference line (surplus/deficit of 0)
    plt.xlabel("Day of the Year")
    plt.ylabel("Cumulative Energy Balance (MWh)")
    plt.title(f"Cumulative Energy Surplus/Deficit Over the Year for 5 Simulations ({production_factor}x Production)")
    plt.legend()
    plt.show()

    #print max surplus and min deficit for each simulation
    for label, cumulative_mwh in simulations:
        max_surplus = cumulative_mwh.max()
        max_deficit = cumulative_mwh.min()
        print(f"For {label}: Maximum surplus is {max_surplus:.2f} MWh, maximum deficit is {max_deficit:.2f} MWh.")

def main():
    #runs all simulations and compares power production w/demand
    print("Starting all simulations...")

    #run simulations
    run_demand_simulation()
    run_wind_simulation()
    run_solar_simulation()

    #get optimized coefficients
    w_opt, s_opt = optimize_coefficients()

    #compare multiple simulations
    compare_multiple_simulations(w_opt, s_opt)
    compare_multiple_simulations_mwh(w_opt, s_opt)

    print("All simulations completed.")

if __name__ == "__main__":
    main()