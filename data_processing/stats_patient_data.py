import pandas as pd

# Patient number
patient_number = 1

# Load the specific .tsv file
file_path = r"G:\Mi unidad\COVID fNIRS\Long-COVID-fNIRS-Analysis\Long_COVID_fNIRS_Raw_Data\patient_1_data.tsv"

# Load the .tsv file
df = pd.read_csv(file_path, sep='\t')

# Remove the first column (time)
df = df.iloc[:, 1:]

# Calculate the statistics: mean, standard deviation, minimum, and maximum
mean_values = df.iloc[:, :-1].mean(axis=0)
std_values = df.iloc[:, :-1].std(axis=0)
min_values = df.iloc[:, :-1].min(axis=0)
max_values = df.iloc[:, :-1].max(axis=0)

# Create a summary row combining the calculated statistics with unique names
summary_data = {'Patient_Number': patient_number}  # Add the patient number as the first column
for col in df.columns[:-1]:
    summary_data[f'mean_{col}'] = mean_values[col]
    summary_data[f'std_{col}'] = std_values[col]
    summary_data[f'min_{col}'] = min_values[col]
    summary_data[f'max_{col}'] = max_values[col]

# Add the status column (0 or 1) to the summary dictionary
summary_data['Status'] = df.iloc[0, -1]

# Convert to DataFrame
summary_row_df = pd.DataFrame([summary_data])

# Save the summarized DataFrame to an Excel file
output_file = r"G:\Mi unidad\COVID fNIRS\Long-COVID-fNIRS-Analysis\patient_1_summary_stats.xlsx"
summary_row_df.to_excel(output_file, index=False)

print(f"Summary data saved to {output_file}")
