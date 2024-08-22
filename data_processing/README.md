## Data Processing Scripts

### `stats_patient_data.py`

This script is used to process individual patient data files from the fNIRS dataset. Specifically, it:

- Removes the time column from the raw data.
- Calculates key statistical measures (mean, standard deviation, minimum, and maximum) for each biomedical variable across the patient's recording session.
- Assigns a unique patient number for reference.
- Appends the patient's health status (control or long COVID) to the summarized data.
- Saves the summarized data into a new Excel file for further analysis.

**Adaptability:**
- The script is currently set up for processing a single patient's data. However, it can be easily adapted to process multiple patient files by iterating over all the `.tsv` files in the directory.

**Next Steps:**
- After processing individual patient files, the next step is to compile all 17 patients' summarized data into a single file. This consolidated file will serve as the complete dataset for further analysis and model training.

**Location:**
- The script can be found in the `scripts` folder of this repository.
