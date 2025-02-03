import pandas as pd

#efficiency values
wind_efficiency = 0.4
solar_efficiency = 0.2

#load csv files
fourier_df = pd.read_csv("FourierResults.csv", dtype={"Column": str})  # Ensure Column is read as string
coefficients_df = pd.read_csv("coefficients.csv")

#extract relevant columns
fourier_df = fourier_df[["File", "Column"]]
coefficients_df = coefficients_df[["Basis Function", "Coefficient", "Average Value of Wind/Solar"]]

#convert Basis Function labels (y1, y2, etc.) to match indices
coefficients_df["Basis Function"] = coefficients_df["Basis Function"].apply(lambda x: int(x[1:]) - 1)

#merge data on index alignment
merged_df = pd.concat([fourier_df, coefficients_df], axis=1)


#compute installation capacity
def compute_installation_capacity(row):
    coefficient = row["Coefficient"]
    avg_wind_solar = row["Average Value of Wind/Solar"]

    column_value = str(row["Column"]) #ensure it's a string

    if "wdsp^3" in column_value: #wind speed-based power
        efficiency = wind_efficiency
    elif "glorad" in column_value: #solar radiation-based power
        efficiency = solar_efficiency
    else:
        return 0 #unknown type

    return abs(coefficient * avg_wind_solar / efficiency) #ensure non-negative capacity


#apply function to each row
merged_df["Installation Capacity (MW)"] = merged_df.apply(compute_installation_capacity, axis=1)

#save
installation_capacity_file = "InstallationCapacities.csv"
merged_df.to_csv(installation_capacity_file, index=False)

print(f"Installation capacities saved to {installation_capacity_file}")