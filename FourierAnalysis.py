import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg') #tkagg fixes rendering errors for me
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

top_n = 5

#load and preprocess data
def load_and_preprocess_data(file_path, column_name):
    #load data from csv
    data = pd.read_csv(file_path)

    #convert date -> datetime format. Date must be in Excel short form
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y') #adjust format according to CSV

    ##ensure specified column is numeric, handle unexpected non-numeric entries
    data[column_name] = pd.to_numeric(data[column_name], errors='coerce')

    #drop rows with missing/invalid values in column
    data = data.dropna(subset=[column_name])

    #extract day of year (ignoring leap years to simplify)
    data['Day_of_Year'] = data['Date'].dt.strftime('%m-%d') #extract MM-DD for grouping

    #average values for each day of the year across all years
    daily_avg = data.groupby('Day_of_Year')[column_name].mean()

    #compute standard deviation for each day of the year
    daily_std = data.groupby('Day_of_Year')[column_name].std()

    #convert to numpy arrays
    averaged_values = daily_avg.values
    std_values = daily_std.values

    return averaged_values, std_values, daily_avg.index


#find fourier transform
def perform_fourier_transform(signal):
    fourier_coefficients = fft(signal) #compute FFT
    return fourier_coefficients


#reconstruct signal with top n components
def reconstruct_signal_with_top_components(fourier_coefficients, num_days, top_n):
    #compute amplitudes of the fourier coefficients
    amplitudes = np.abs(fourier_coefficients) / num_days

    #get the indices of the top n amplitudes (ignoring DC component @ index 0)
    significant_indices = np.argsort(amplitudes)[-top_n:][::-1]

    #create a new array for truncated fourier coefficients
    truncated_coefficients = np.zeros_like(fourier_coefficients)
    truncated_coefficients[significant_indices] = fourier_coefficients[significant_indices]

    #reconstruct signal using the truncated coefficients
    reconstructed_signal = ifft(truncated_coefficients).real

    return reconstructed_signal, significant_indices


#print fourier transform eqn for top n components
def print_fourier_equation(fourier_coefficients, num_days, significant_indices):
    frequencies = np.fft.fftfreq(num_days) #frequency vals
    equation_terms = []

    #loop through the significant indices
    for k in significant_indices:
        amplitude = np.abs(fourier_coefficients[k]) / num_days
        phase = np.angle(fourier_coefficients[k])
        frequency = frequencies[k]
        term = f"{amplitude:.6f} * cos(2π * {frequency:.6f} * t + {phase:.6f})"
        equation_terms.append(term)

    #combine terms into an equation
    equation = " + ".join(equation_terms)
    print("\nDerived Fourier Transform Equation (Top Components):")
    print(f"f(t) = {equation}\n")


#plot results
def plot_results(original_signal, reconstructed_signal, fourier_coefficients, significant_indices, day_labels, column_name):
    num_days = len(original_signal)
    frequencies = np.fft.fftfreq(num_days) #frequency values

    #plot original vs reconstructed signal
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_days + 1), original_signal, label=f"Original Signal ({column_name})", marker="o")
    plt.plot(range(1, num_days + 1), reconstructed_signal, label="Reconstructed Signal (Top Components)", linestyle="--")
    plt.xlabel("Day of the Year")
    plt.ylabel(column_name)
    plt.title(f"Original vs Reconstructed Signal ({column_name})")
    plt.legend()
    plt.grid()
    plt.show()

    #plot fourier coefficients (magnitude)
    plt.figure(figsize=(12, 6))
    plt.stem(frequencies[:num_days // 2], np.abs(fourier_coefficients[:num_days // 2]), basefmt=" ")
    plt.scatter(
        frequencies[significant_indices],
        np.abs(fourier_coefficients[significant_indices]),
        color="red",
        label="Top Components",
    )
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title(f"Fourier Coefficients (Top Components Highlighted) - {column_name}")
    plt.legend()
    plt.grid()
    plt.show()


#main fourier analysis function
def analyze_fourier(file_path, column_name):
    #analyse specified column of dataset and compute fourier transform for avg + stdev
    #load and preprocess data
    averaged_values, std_values, day_labels = load_and_preprocess_data(file_path, column_name)

    #perform fourier transform for averages
    avg_fourier_coefficients = perform_fourier_transform(averaged_values)

    #perform fourier transform for standard deviations
    std_fourier_coefficients = perform_fourier_transform(std_values)

    num_days = len(averaged_values)

    #reconstruct average signal using top components
    avg_reconstructed_signal, avg_significant_indices = reconstruct_signal_with_top_components(
        avg_fourier_coefficients, num_days, top_n
    )

    #reconstruct standard deviation signal using top components
    std_reconstructed_signal, std_significant_indices = reconstruct_signal_with_top_components(
        std_fourier_coefficients, num_days, top_n
    )

    #print fourier transform equation for the average signal
    print("\nFourier Transform for Averages:")
    print_fourier_equation(avg_fourier_coefficients, num_days, avg_significant_indices)

    #print fourier transform equation for the standard deviation signal
    print("\nFourier Transform for Standard Deviations:")
    print_fourier_equation(std_fourier_coefficients, num_days, std_significant_indices)

    #plot average signal and its fourier components
    plot_results(
        averaged_values, avg_reconstructed_signal, avg_fourier_coefficients, avg_significant_indices, day_labels, column_name
    )

    #plot standard deviation signal and its fourier components
    plot_results(
        std_values, std_reconstructed_signal, std_fourier_coefficients, std_significant_indices, day_labels, f"Std Dev of {column_name}"
    )


#run fourier analysis when this file is executed
if __name__ == "__main__":
    file_path = "WeatherDunsanyProcessed.csv" #replace with the actual file path
    column_name = "wdsp^3" #replace with the column name to analyze
    analyze_fourier(file_path, column_name)