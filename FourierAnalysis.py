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
    #also ignore 0s
    data = data[data[column_name] != 0]

    #extract day of year (ignoring leap years to simplify)
    data['Day_of_Year'] = data['Date'].dt.strftime('%m-%d')  # extract MM-DD for grouping

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
def get_fourier_equation(fourier_coefficients, num_days, significant_indices):
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
    return f"f(t) = {equation}"


#convert fourier transform into requested cosine sum format
def get_cosine_form(fourier_coefficients, num_days, significant_indices):
    frequencies = np.fft.fftfreq(num_days) #frequency vals
    cosine_terms = []

    #loop through the significant indices
    for k in significant_indices:
        amplitude = np.abs(fourier_coefficients[k]) / num_days
        phase = np.angle(fourier_coefficients[k])
        frequency = frequencies[k]
        term = f"{amplitude:.6f} * np.cos(2 * np.pi * {frequency:.6f} * days + {phase:.6f})"
        cosine_terms.append(term)

    #combine terms into the requested format
    return " + ".join(cosine_terms)


#plot results
def plot_results(original_signal, reconstructed_signal, fourier_coefficients, significant_indices, day_labels,
                 column_name):
    num_days = len(original_signal)
    frequencies = np.fft.fftfreq(num_days) #frequency values

    #plot original vs reconstructed signal
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_days + 1), original_signal, label=f"Original Signal ({column_name})", marker="o")
    plt.plot(range(1, num_days + 1), reconstructed_signal, label="Reconstructed Signal (Top Components)",
             linestyle="--")
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


#function to write Fourier results to CSV
def write_fourier_to_csv(csv_filename, file_path, column_name, avg_equation, std_equation, avg_cosine_form, std_cosine_form, row_num):
    # read existing data if file exists
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        #if file doesn't exist, create dataframe w/column headers
        df = pd.DataFrame(columns=["File", "Column", "Fourier_Avg", "Fourier_Std", "Fourier_Avg_Cosine_Form", "Fourier_Std_Cosine_Form"])

    #ensure the dataframe has all required columns (handle missing ones)
    expected_columns = ["File", "Column", "Fourier_Avg", "Fourier_Std", "Fourier_Avg_Cosine_Form", "Fourier_Std_Cosine_Form"]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = ""  # add missing columns

    #ensure the dataframe has enough rows
    while len(df) <= row_num:
        df.loc[len(df)] = [""] * len(df.columns)  # add empty rows if needed

    #insert new data at specified row
    df.loc[row_num] = [file_path, column_name, avg_equation, std_equation, avg_cosine_form, std_cosine_form]

    #save back to CSV
    df.to_csv(csv_filename, index=False)

    print(f"Fourier series saved to {csv_filename} (Row {row_num})")


#main fourier analysis function
def analyze_fourier(file_path, column_name, csv_filename="FourierResults.csv", row_num=0):
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

    #get fourier transform equations
    avg_equation = get_fourier_equation(avg_fourier_coefficients, num_days, avg_significant_indices)
    std_equation = get_fourier_equation(std_fourier_coefficients, num_days, std_significant_indices)

    #get cosine sum forms
    avg_cosine_form = get_cosine_form(avg_fourier_coefficients, num_days, avg_significant_indices)
    std_cosine_form = get_cosine_form(std_fourier_coefficients, num_days, std_significant_indices)

    #save fourier series to CSV
    write_fourier_to_csv(csv_filename, file_path, column_name, avg_equation, std_equation, avg_cosine_form, std_cosine_form, row_num)


#run fourier analysis when this file is executed
if __name__ == "__main__":
    file_path = "WeatherDunsanyProcessed.csv" #replace with the actual file path
    column_name = "glorad" #replace with the column name to analyze
    csv_filename = "FourierResults.csv" #file to store results
    row_num = 13 #update manually per weather station

    analyze_fourier(file_path, column_name, csv_filename, row_num)