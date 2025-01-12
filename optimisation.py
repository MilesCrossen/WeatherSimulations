import matplotlib
matplotlib.use("TkAgg") #tkagg is a compatible backend (tkinter)
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#define days
days = np.arange(1, 366)

#convert to EXACT months/month lengths
def days_to_months_exact(days):
    cumulative_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    months = []
    for day in days:
        for i in range(1, len(cumulative_days)):
            if cumulative_days[i-1] < day <= cumulative_days[i]:
                month = i
                fraction = (day - cumulative_days[i-1]) / (cumulative_days[i] - cumulative_days[i-1])
                months.append(month + fraction - 1)  #sub 1 to start at 0
                break
    return np.array(months)

#adjust demand eqn
def adjusted_demand(days):
    months = days_to_months_exact(days)
    return 1.8062 * months**3 - 9.8403 * months**2 - 156.1 * months + 5084.4

#solar capacity eqn
def solar_capacity(days):
    months = days_to_months_exact(days)
    return 2.9668 * months**3 - 110.07 * months**2 + 966.25 * months - 848.84

#wind capacity eqn
def wind_capacity(days):
    months = days_to_months_exact(days)
    return 0.0029 * months**3 + 0.0321 * months**2 - 0.9065 * months + 13.076

def optimize_coefficients():
    #optimizes coefficients for wind and power gen capacity to minimise the absolute error between
    #predicted production and adjusted demand
    #gen data
    demand = adjusted_demand(days)
    solar = solar_capacity(days)
    wind = wind_capacity(days)

    #define objective function to minimise absolute error
    def objective(params):
        a, b = params
        predicted = a * wind + b * solar
        error = np.abs(demand - predicted)
        return np.sum(error)

    #initial guesses
    initial_guess = [1, 1]

    #optimize using minimisation
    result = minimize(objective, initial_guess, method='Nelder-Mead')

    #obtain/extract coefficients w_opt (wind) and s_opt( solar)
    w_opt, s_opt = result.x
    print(f"Optimized coefficients: wind = {w_opt:.4f}, solar = {s_opt:.4f}")

    #predicted vals
    predicted = w_opt * wind + s_opt * solar
    absolute_error = np.abs(demand - predicted)
    total_error = np.sum(absolute_error)

    print(f"Total Absolute Error: {total_error:.2f}")

    #plot results
    plt.figure(figsize=(12, 6))
    plt.plot(days, demand, label="Adjusted Demand", marker="o", markersize=2, linestyle="None")
    plt.plot(days, predicted, label="Predicted", linestyle="-")
    plt.xlabel("Day")
    plt.ylabel("Electricity Demand (MW)")
    plt.title("Daily Adjusted Demand vs Predicted Production")
    plt.legend()
    plt.grid(True)
    plt.show()

    #return optimized coeffs
    return w_opt, s_opt

#run optimization if executed directly
if __name__ == "__main__":
    w_opt, s_opt = optimize_coefficients()
    print(f"Script executed directly: Optimized coefficients - wind: {w_opt}, solar: {s_opt}")