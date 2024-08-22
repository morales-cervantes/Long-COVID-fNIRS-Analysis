

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the CSV file
file_path = r"C:\Users\anton\Downloads\completos.csv"
df = pd.read_csv(file_path)

# Identify sick and control patients
sick_patients = df[df['status'] == 1]['Paciente'].unique()
control_patients = df[df['status'] == 0]['Paciente'].unique()

# Directory to save the files
output_dir = r"C:\Users\anton\Downloads\output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Repeat the entire process 10 times
final_metrics = []
all_confusion_matrices = []
all_roc_data = []
patient_selections = []
confusion_values = []

for run in range(10):
    all_metrics = []
    roc_data = []
    
    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_smote, y_smote = smote.fit_resample(df.drop(['Paciente', 'status'], axis=1), df['status'])
    
    # Create balanced DataFrame
    df_smote = pd.DataFrame(X_smote, columns=df.drop(['Paciente', 'status'], axis=1).columns)
    df_smote['status'] = y_smote
    
    # Generate identifiers for SMOTE-generated data
    df_smote['Paciente'] = ['Generated_{}'.format(i) if i >= len(df) else df.iloc[i]['Paciente'] for i in range(len(df_smote))]

    for i in range(5):
        # Select one real sick patient and 3 real control patients for the test set
        test_sick_patient = sick_patients[i % len(sick_patients)]
        test_control_patients = np.random.choice(control_patients, 3, replace=False)
        
        # Create test set with 1 real sick patient and 3 real control patients
        df_test = df[(df['Paciente'].isin([test_sick_patient]) | df['Paciente'].isin(test_control_patients))]
        df_test_smote = df_smote[~df_smote.index.isin(df_test.index)]
        
        X_test = df_test.drop(['Paciente', 'status'], axis=1)
        y_test = df_test['status']
        
        # Create training set with the rest of the balanced data
        df_train = df_smote[df_smote.index.isin(df_test_smote.index)]
        
        X_train = df_train.drop(['status', 'Paciente'], axis=1)
        y_train = df_train['status']
        
        # Train the model
        clf = RandomForestClassifier(random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        # Calculate performance metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate sensitivity and specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Save metrics
        all_metrics.append([i+1, acc, prec, rec, f1, roc_auc, sensitivity, specificity, run+1, tn, fp, fn, tp])
        
        # Save ROC points
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_data.append(pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Iteration': i+1, 'Run': run+1, 'AUC': roc_auc}))
        
        # Save training and test datasets to Excel files
        train_file_path = os.path.join(output_dir, f'training_set_run_{run+1}_iteration_{i+1}.xlsx')
        test_file_path = os.path.join(output_dir, f'test_set_run_{run+1}_iteration_{i+1}.xlsx')
        
        df_train.to_excel(train_file_path, index=False)
        df_test.to_excel(test_file_path, index=False)
        
        # Save patient selections
        patient_selections.append({
            'Run': run+1,
            'Iteration': i+1,
            'Test_Sick_Patient': test_sick_patient,
            'Test_Control_Patients': test_control_patients.tolist()
        })
        
        print(f"Run {run+1}, Iteration {i+1} - Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1 Score: {f1}, AUC: {roc_auc}, Sensitivity: {sensitivity}, Specificity: {specificity}")

    # Save metrics to a separate Excel file
    metrics_df = pd.DataFrame(all_metrics, columns=['Iteration', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Sensitivity', 'Specificity', 'Run', 'TN', 'FP', 'FN', 'TP'])
    
    # Calculate composite metric and select the top 5 iterations
    metrics_df['Composite'] = metrics_df['AUC'] * 0.5 + metrics_df['Sensitivity'] * 0.25 + metrics_df['Specificity'] * 0.25
    top_5_metrics = metrics_df.nlargest(5, 'Composite')
    
    # Sort and select the 3 central iterations, removing the best and worst
    top_5_metrics_sorted = top_5_metrics.sort_values(by='Composite')
    central_3_metrics = top_5_metrics_sorted.iloc[1:4]  # Remove the first and last entries

    
    # Store the metrics for the 3 central iterations
    final_metrics.append(central_3_metrics)
    
    # Save ROC points for the central iterations
    all_roc_data.extend([roc_data[int(i)-1] for i in central_3_metrics['Iteration']])
    
    metrics_file_path = os.path.join(output_dir, f'central_3_metrics_run_{run+1}.xlsx')
    central_3_metrics.to_excel(metrics_file_path, index=False)
    
    # Save confusion matrix values for the central 3 iterations
    for i in central_3_metrics['Iteration']:
        tn, fp, fn, tp = confusion_matrix(y_test, clf.predict(X_test)).ravel()
        confusion_values.append({'Run': run+1, 'Iteration': i, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})
    
    roc_file_path = os.path.join(output_dir, f'central_3_roc_data_run_{run+1}.xlsx')
    pd.concat(all_roc_data, ignore_index=True).to_excel(roc_file_path, index=False)

# Concatenate all final metrics and save to an Excel file
final_metrics_df = pd.concat(final_metrics, ignore_index=True)
final_metrics_file_path = os.path.join(output_dir, 'final_metrics.xlsx')
final_metrics_df.to_excel(final_metrics_file_path, index=False)

# Calculate and display average metrics
mean_metrics = final_metrics_df.drop(columns=['Iteration', 'Run']).mean()
mean_metrics_df = pd.DataFrame(mean_metrics).transpose()
mean_metrics_file_path = os.path.join(output_dir, 'mean_metrics.xlsx')
mean_metrics_df.to_excel(mean_metrics_file_path, index=False)
print(mean_metrics)

# Select the best, worst, and average ROC curves
roc_aucs = final_metrics_df.groupby(['Run', 'Iteration']).apply(lambda x: x['AUC'].iloc[0])
best_roc = roc_aucs.idxmax()
worst_roc = roc_aucs.idxmin()
mean_roc = roc_aucs.mean()

best_roc_data = [df for df in all_roc_data if df['Run'].iloc[0] == best_roc[0] and df['Iteration'].iloc[0] == best_roc[1]][0]
worst_roc_data = [df for df in all_roc_data if df['Run'].iloc[0] == worst_roc[0] and df['Iteration'].iloc[0] == worst_roc[1]][0]

# Plot the best, worst, and average ROC curves
plt.figure()

for label, roc in [('Best', best_roc_data), ('Worst', worst_roc_data)]:
    plt.plot(roc['FPR'], roc['TPR'], lw=2, alpha=0.8, label=f'{label} ROC (AUC = {roc["AUC"].iloc[0]:.2f})')

# Generate the average ROC curve
mean_fpr = np.linspace(0, 1, 100)
tprs = []
for df in all_roc_data:
    tpr = np.interp(mean_fpr, df['FPR'], df['TPR'])
    tpr[0] = 0.0
    tprs.append(tpr)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=0.8, label=f'Mean ROC (AUC = {mean_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.show()

# Save the TN, FP, FN, and TP values in an Excel file
confusion_values_df = pd.DataFrame(confusion_values)
confusion_values_file_path = os.path.join(output_dir, 'confusion_values.xlsx')
confusion_values_df.to_excel(confusion_values_file_path, index=False)

# Calculate average TN, FP, FN, and TP values from the central iterations
confusion_values_central = confusion_values_df.groupby(['Run', 'Iteration']).mean()

# Create a global confusion matrix based on the average values
tn_mean = confusion_values_central['TN'].sum()
fp_mean = confusion_values_central['FP'].sum()
fn_mean = confusion_values_central['FN'].sum()
tp_mean = confusion_values_central['TP'].sum()

confusion_matrix_global = np.array([
    [tn_mean, fp_mean],
    [fn_mean, tp_mean]
]).astype(int)

print("Global Confusion Matrix:")
print(confusion_matrix_global)

# Calculate sensitivity and specificity from the global confusion matrix
tn, fp, fn, tp = confusion_matrix_global.ravel()
global_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
global_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"Global Sensitivity: {global_sensitivity}")
print(f"Global Specificity: {global_specificity}")

# Include TN, FP, FN, and TP values in the final metrics file
mean_metrics_df['TN'] = tn_mean
mean_metrics_df['FP'] = fp_mean
mean_metrics_df['FN'] = fn_mean
mean_metrics_df['TP'] = tp_mean
mean_metrics_df.to_excel(mean_metrics_file_path, index=False)

# Save patient selections in an Excel file
patient_selections_df = pd.DataFrame(patient_selections)
patient_selections_file_path = os.path.join(output_dir, 'patient_selections.xlsx')
patient_selections_df.to_excel(patient_selections_file_path, index=False)


                    