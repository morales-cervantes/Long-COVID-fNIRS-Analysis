# Long COVID fNIRS Raw Data

This folder contains the raw fNIRS data for 17 patients involved in the study on long COVID neural correlates. Each `.tsv` file corresponds to a specific patient, with the format `patient_X_data.tsv`, where `X` is the patient number.

## File Structure

- **Time Column:** The first column in each `.tsv` file represents the time points at which the measurements were taken.
- **Status Column:** The last column in each file indicates the health status of the patient. A value of `0` indicates a healthy control, and `1` indicates a long COVID patient.

## Dataset Overview

- Total Patients: 17
  - Healthy Controls: 12
  - Long COVID Patients: 5

## File Naming Convention

- `patient_1_data.tsv` - Data for Patient 1
- `patient_2_data.tsv` - Data for Patient 2
- ...
- `patient_N_data.tsv` - Data for Patient N

## Usage

These files can be used for further analysis, including preprocessing, feature extraction, and machine learning model training. Ensure that the appropriate preprocessing steps are applied before using the data in any analysis.
