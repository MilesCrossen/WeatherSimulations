import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.fft import fft, ifft
from FourierAnalysis import top_n

#set matplotlib backend explicitly for PyCharm compatibility
matplotlib.use("TkAgg")

#define the file path
file_path = "22-23 Electricity Demand.csv"

#function to process AI demand data and perform Fourier Transform
def process_ai_demand_and_fourier(file_path, top_n):
    #load the data with appropriate settings
    try:
        data = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    #check for required columns
    if "DateTime" not in data.columns or "AI Demand" not in data.columns:
        print("Required columns ('DateTime', 'AI Demand') are missing from the file.")
        return

    #convert DateTime column to datetime format
    try:
        data['DateTime'] = pd.to_datetime(data['DateTime'], format='%d/%m/%Y %H:%M')
    except Exception as e:
        print(f"Error parsing DateTime column: {e}")
        return

    #filter for data within the year 2022
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31, 23, 59)  # End of 2022
    data = data[(data['DateTime'] >= start_date) & (data['DateTime'] <= end_date)]

    #ensure AI Demand is numeric and drop invalid rows
    data['AI Demand'] = pd.to_numeric(data['AI Demand'], errors='coerce')
    data = data.dropna(subset=['AI Demand'])

    #extract the date from DateTime and group by date to calculate daily average
    data['Date'] = data['DateTime'].dt.date
    daily_avg_demand = data.groupby('Date')['AI Demand'].mean()

    #save the daily average demand to a new CSV
    output_file = "daily_avg_ai_demand.csv"
    try:
        daily_avg_demand.to_csv(output_file, index=True, header=["Average AI Demand (MW)"])
        print(f"Daily average AI demand for 2022 saved to '{output_file}'.")
    except Exception as e:
        print(f"Error saving the CSV: {e}")
        return

    #plot the daily average AI demand
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(daily_avg_demand.index, daily_avg_demand.values, marker='o', color='green')
        plt.title("Average Daily AI Demand (2022)")
        plt.xlabel("Date")
        plt.ylabel("AI Demand (MW)")
        plt.grid()
        plt.xticks(rotation=45) #rotate x-axis labels for better readability
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error generating the plot: {e}")
        return

    #perform fourier transform on the daily averages
    print("\nPerforming Fourier Transform on daily average AI demand...")
    daily_values = daily_avg_demand.values
    num_days = len(daily_values)

    #compute Fourier coefficients
    fourier_coefficients = fft(daily_values)

    #reconstruct signal with top n components
    amplitudes = np.abs(fourier_coefficients) / num_days
    significant_indices = np.argsort(amplitudes)[-top_n:][::-1]

    truncated_coefficients = np.zeros_like(fourier_coefficients)
    truncated_coefficients[significant_indices] = fourier_coefficients[significant_indices]

    reconstructed_signal = ifft(truncated_coefficients).real

    #print Fourier Transform equation
    frequencies = np.fft.fftfreq(num_days)
    equation_terms = []
    for k in significant_indices:
        amplitude = amplitudes[k]
        phase = np.angle(fourier_coefficients[k])
        frequency = frequencies[k]
        term = f"{amplitude:.6f} * cos(2π * {frequency:.6f} * t + {phase:.6f})"
        equation_terms.append(term)

    equation = " + ".join(equation_terms)
    print("\nDerived Fourier Transform Equation (Top Components):")
    print(f"f(t) = {equation}\n")

    #plot original vs reconstructed signal
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, num_days + 1), daily_values, label="Original Signal", marker="o")
        plt.plot(range(1, num_days + 1), reconstructed_signal, label="Reconstructed Signal", linestyle="--")
        plt.xlabel("Day of the Year")
        plt.ylabel("AI Demand (MW)")
        plt.title("Original vs Reconstructed Signal (AI Demand)")
        plt.legend()
        plt.grid()
        plt.show()
    except Exception as e:
        print(f"Error generating the Fourier Transform plot: {e}")


#call the function to process and analyze AI demand
process_ai_demand_and_fourier(file_path, top_n)