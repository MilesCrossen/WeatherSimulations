import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#define days
days = np.arange(1, 366)

#demand average (fourier transform eqn)
def adjusted_demand(days):
    return (
        4454.708901 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
        202.979616 * np.cos(2 * np.pi * -0.002740 * days + 0.261259) +
        202.979616 * np.cos(2 * np.pi * 0.002740 * days + -0.261259) +
        133.571826 * np.cos(2 * np.pi * -0.142466 * days + -2.989836) +
        133.571826 * np.cos(2 * np.pi * 0.142466 * days + 2.989836)
    )

#solar average
def solar_capacity(days):
    return (
        974.988895 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
        420.293533 * np.cos(2 * np.pi * -0.002732 * days + 2.903370) +
        420.293533 * np.cos(2 * np.pi * 0.002732 * days + -2.903370) +
        30.446476 * np.cos(2 * np.pi * -0.005464 * days + -1.495144) +
        30.446476 * np.cos(2 * np.pi * 0.005464 * days + 1.495144)
    )

#wind average
def wind_capacity(days):
    return (
        125.528148 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
        18.344844 * np.cos(2 * np.pi * -0.002732 * days + 0.249556) +
        18.344844 * np.cos(2 * np.pi * 0.002732 * days + -0.249556) +
        3.719645 * np.cos(2 * np.pi * 0.117486 * days + -0.565647) +
        3.719645 * np.cos(2 * np.pi * -0.117486 * days + 0.565647)
    )

def optimize_coefficients():
    #optimizes coefficients for wind and power gen capacity to minimize the absolute error
    #between predicted production and adjusted demand

    # Generate data
    demand = adjusted_demand(days)
    solar = solar_capacity(days)
    wind = wind_capacity(days)

    #define objective function to minimize absolute error
    def objective(params):
        a, b = params
        predicted = a * wind + b * solar
        error = np.abs(demand - predicted)
        return np.sum(error)

    #initial guesses
    initial_guess = [1, 1]

    #optimize using minimization
    result = minimize(objective, initial_guess, method='Nelder-Mead')

    #obtain/extract coefficients w_opt (wind) and s_opt (solar)
    w_opt, s_opt = result.x
    print(f"Optimized coefficients: wind = {w_opt:.4f}, solar = {s_opt:.4f}")

    #predicted values
    predicted = w_opt * wind + s_opt * solar

    #calculate total absolute error
    absolute_error = np.abs(demand - predicted)
    total_error = np.sum(absolute_error)
    print(f"Total Absolute Error: {total_error:.2f}")

    #calculate and print average wind and solar production
    avg_wind_production = np.mean(w_opt * wind)
    avg_solar_production = np.mean(s_opt * solar)
    wind_to_solar_ratio = avg_wind_production / avg_solar_production

    print(f"Average Wind Production: {avg_wind_production:.2f} MW")
    print(f"Average Solar Production: {avg_solar_production:.2f} MW")
    print(f"Wind-to-Solar Production Ratio: {wind_to_solar_ratio:.2f}")

    #plot results
    plt.figure(figsize=(12, 6))
    plt.plot(days, demand, label="Adjusted Demand", linestyle="-", color="blue")  # Demand line
    plt.plot(days, predicted, label="Predicted", linestyle="-", color="orange")  # Predicted line
    plt.xlabel("Day")
    plt.ylabel("Electricity Demand (MW)")
    plt.title("Daily Adjusted Demand vs Predicted Production")
    plt.legend()
    plt.grid(True)
    plt.show()

    #return optimized coefficients
    return w_opt, s_opt

#run optimization if executed directly
if __name__ == "__main__":
    w_opt, s_opt = optimize_coefficients()
    print(f"Script executed directly: Optimized coefficients - wind: {w_opt}, solar: {s_opt}")