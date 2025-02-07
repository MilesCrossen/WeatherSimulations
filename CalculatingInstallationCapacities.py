import pandas as pd

solar_efficiency = 0.2 #user defined... usually .15-.22 in real situations
wind_efficiency = 0.4 #user defined... up to 59.3% (notes in our old thermofluids notes)

#used csv files
fourier_df = pd.read_csv("FourierResults.csv")
coefficients_df = pd.read_csv("coefficients.csv")

#skip rows where 'file' column is empty
fourier_df = fourier_df.dropna(subset=["File"])
fourier_df = fourier_df[fourier_df["File"].str.strip() != ""] #remove empty/whitespace vals

#only keep necessary columns
fourier_df = fourier_df[["File", "Column"]].copy()
coefficients_df = coefficients_df[["Basis Function", "Coefficient", "Average Value of Wind/Solar"]].copy()

#convert basis functions to indexes
coefficients_df["Basis Function"] = coefficients_df["Basis Function"].apply(lambda x: int(x[1:]) - 1)

#check both dataframes are same length
min_length = min(len(fourier_df), len(coefficients_df))
fourier_df = fourier_df.iloc[:min_length].reset_index(drop=True)
coefficients_df = coefficients_df.iloc[:min_length].reset_index(drop=True)

#merge data w/index
merged_df = pd.concat([fourier_df, coefficients_df], axis=1)

#extract and filter weather station names for cleanliness
merged_df["Weather Station"] = merged_df["File"].str.replace("Weather", "").str.replace("Processed.csv", "", regex=True)

#drop rows where weather station column is empty
merged_df = merged_df.dropna(subset=["Weather Station"])
merged_df = merged_df[merged_df["Weather Station"].str.strip() != ""]

#check column names are correctly referenced
merged_df = merged_df.rename(columns={"Column": "Type", "Average Value of Wind/Solar": "Average Value"})

#calc installation capacity
def compute_installation_capacity(row):
    coefficient = row["Coefficient"]
    avg_production = row["Average Value"]

    if pd.isna(avg_production): #avoid errors if val is nan
        return 0

    if "wdsp^3" in row["Type"]: #wind power
        efficiency = wind_efficiency
    elif "glorad" in row["Type"]: #solar power
        efficiency = solar_efficiency
    else:
        return 0 #unknown/undefined type

    return abs((coefficient * avg_production) / efficiency) #check non negativity (not super important
#because coefficients and averages are already nonegative)

#compute installation capacity, add to dataframe
merged_df["Installation Capacity (MW)"] = merged_df.apply(compute_installation_capacity, axis=1)

#select important columns
final_df = merged_df[["Weather Station", "Type", "Coefficient", "Average Value", "Installation Capacity (MW)"]]

#save to new csv
output_file = "CapacityRecommendations.csv"
final_df.to_csv(output_file, index=False)

print(f"✅ Data successfully saved to {output_file}")