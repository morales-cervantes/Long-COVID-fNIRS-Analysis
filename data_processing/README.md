# Data Processing Scripts for Long COVID fNIRS Analysis

This directory contains the scripts and associated data used for processing and analyzing the fNIRS data related to Long COVID patients. The analysis focuses on identifying significant patterns in the data using various machine learning models, evaluating their performance, and generating relevant metrics and visualizations.

## Scripts Included

### 1. [Long_COVID_Data_Preprocessing.py](Long_COVID_Data_Preprocessing.py)
This script is responsible for processing the raw fNIRS data for individual patients. It performs the following tasks:
- Loads individual `.tsv` files containing patient data.
- Removes unnecessary columns, such as the time column.
- Computes statistical summaries (mean, standard deviation, minimum, and maximum) for each measured variable across time.
- Outputs a summary `.csv` file (`completos.csv`) containing a single row per patient, summarizing their data.

**Note:** The script is currently set up to process a single patient at a time, but it can be easily adapted to handle multiple patients by iterating over all files in the directory.

### 2. [Long_COVID_Model_Analysis.py](Long_COVID_Model_Analysis.py)
This script performs machine learning analysis on the summarized data from `completos.csv`. The main steps include:
- Loading the summarized data for all 17 patients.
- Applying the SMOTE algorithm to balance the dataset between healthy and Long COVID patients.
- Splitting the data into training and test sets, ensuring that each test set contains one real Long COVID patient and three real healthy patients.
- Running a Random Forest classifier over multiple iterations and runs, calculating performance metrics (accuracy, precision, recall, F1 score, AUC, sensitivity, and specificity) for each iteration.
- **Selecting the central 3 iterations:** The script now eliminates the best and worst iterations, retaining only the central 3 iterations for a more balanced evaluation.
- Plotting the best, worst, and mean ROC curves for comparison.

### Results and Outputs
- **completos.csv:** Contains the processed data for all 17 patients.
- **Central 3 Metrics for Each Run:** Saved in individual Excel files (`central_3_metrics_run_{n}.xlsx`) for each run.
- **ROC Curves:** The script generates and saves the ROC curves (`ROC CURVES.png`) showing the best, worst, and mean performance across all runs.
- **Final Metrics:** The overall performance metrics averaged across the central iterations are saved in `final_metrics.xlsx`.

![ROC Curves](ROC%20CURVES.png)

## How to Use
1. Ensure that all required Python packages are installed, particularly `pandas`, `numpy`, `imblearn`, `sklearn`, `matplotlib`, and `seaborn`.
2. Start by running `Long_COVID_Data_Preprocessing.py` to generate the summarized data for each patient.
3. Once you have the summarized data (`completos.csv`), run `Long_COVID_Model_Analysis.py` to perform the analysis and generate the results.
4. Review the generated metrics and ROC curves to evaluate the performance of the models.

## Authors
- Antony Morales-Cervantes
- Victor Herrera
- Blanca Nohemí Zamora-Mendoza
- Rogelio Flores-Ramírez
- Edgar Guevara

## Contact
For any questions or issues regarding this repository, please contact [Antony Morales-Cervantes](mailto:antony.mc@morelia.tecnm.mx).


