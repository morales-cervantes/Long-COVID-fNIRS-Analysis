# Long COVID Neural Correlates Identification Using fNIRS Data

This repository contains the complete dataset and Python scripts used for analyzing functional Near-Infrared Spectroscopy (fNIRS) data to identify neural correlates of long COVID. The analysis integrates advanced machine learning techniques to classify patients as either long COVID or control based on fNIRS data. The repository includes implementations of four machine learning models: K-Nearest Neighbors (KNN), Random Forest, Support Vector Machine (SVM), and XGBoost.

## Contents

- **Dataset**: The fNIRS data used in this study, including preprocessed features extracted from time series data for each patient.
- **Scripts**: Python scripts that perform data preprocessing, apply the Synthetic Minority Over-sampling Technique (SMOTE) for class balancing, and train the machine learning models. Each script is designed to handle the cross-validation process, model training, evaluation, and generation of performance metrics.
- **Models**: Implementations of KNN, Random Forest, SVM, and XGBoost, optimized for classifying long COVID patients.
- **Results**: Detailed outputs including accuracy, sensitivity, specificity, PPV, and NPV for each model, across multiple iterations of cross-validation.

## Methodology

The data were preprocessed to remove noise and artifacts, and then statistical features were extracted from the fNIRS time series data. SMOTE was applied to address class imbalance. The dataset was split into training and testing sets, ensuring representativity of both classes in each iteration. The models were trained using a 5-fold cross-validation approach to ensure robust performance.

## Purpose

This project aims to enhance the understanding of long COVID’s neurological impacts by leveraging portable neuroimaging techniques combined with machine learning. The results provide insights into the most effective models for classifying long COVID patients, potentially aiding in better diagnosis and management of the condition.

## Authors

- Antony Morales-Cervantes
- Victor Herrera
- Blanca Nohemí Zamora-Mendoza
- Rogelio Flores-Ramírez
- Edgar Guevara

## How to Use

1. **Clone the Repository**: Download the repository to your local machine.
2. **Install Dependencies**: Ensure you have Python and necessary libraries installed as listed in `README` from data_processing file.
3. **Run the Scripts**: Use the provided scripts to preprocess the data, train the models, and evaluate their performance.
4. **Explore the Results**: Analyze the outputs to understand the performance of each model and the implications for long COVID classification.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
