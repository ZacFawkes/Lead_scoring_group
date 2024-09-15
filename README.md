# Data Preprocessing, Feature Selection, and Model Optimization

## 1. Import Necessary Libraries
Before starting, make sure to import the required libraries for data exploration, feature selection, and model optimization:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from optbinning import BinningProcess
```

## 2. Data Exploration
Check for Duplicates: Ensure there are no duplicate rows in your dataset.
Missing Values (NaN): Detect and handle missing values.
Outliers: Identify and manage outliers that may skew model performance.
```python
# Checking for missing values
data.isnull().sum()

# Checking for duplicates
data.duplicated().sum()

# Visualizing outliers using boxplots
sns.boxplot(data=data)
```

## 3. Feature Selection Using Correlation and Information Value (IV)
The ft_select_corr_iv function helps in selecting features based on Pearson correlation and Information Value (IV). Features that have a high correlation (> 0.7) and a low IV will be removed.

Function Definition:
```python
def ft_select_corr_iv(data, var_list, iv_df):
    corr = data[var_list].corr()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Plot the heatmap
    f, ax = plt.subplots(figsize=(20, 20))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, mask=mask, vmax=1, center=0.5,
                square=True, linewidths=.5, cbar_kws={"shrink": .6}, annot=True)
    plt.title("Pearson Correlation", fontsize=10)
    plt.show()

    corr = corr.reset_index()
    remove_vars = []
    for col in corr.columns[1:]:
        for i in range(len(corr[col])):
            var_i = corr["index"][i]
            if (abs(corr[col][i]) > 0.7) and (var_i != col):
                iv_var_i = iv_df.loc[iv_df["variable"] == var_i, "iv"].item()
                iv_var_col = iv_df.loc[iv_df["variable"] == col, "iv"].item()
                remove = col if iv_var_col < iv_var_i else var_i
                if remove not in remove_vars:
                    remove_vars.append(remove)

    select_var = [v for v in var_list if v not in remove_vars]
    select_var = list(iv_df[(iv_df['variable'].isin(select_var)) & (iv_df['iv'] > 0.07)]['variable'])
    return select_var, remove_vars
```
This function:

Calculates Pearson correlation among variables.
Plots a heatmap of the correlation matrix.
Removes variables with a high correlation and low IV values.

## 4. Automatic Binning of Variables
The ft_auto_binning function performs optimal binning on both numeric and categorical variables. It also calculates IV (Information Value) for each variable.
Function Definition:
```python
def ft_auto_binning(data_, numeric_vars, char_vars):
    l = []
    l_binning_table = {}
    binning_process = BinningProcess(variable_names=numeric_vars + char_vars, 
                                     categorical_variables=char_vars, 
                                     max_n_bins=5, split_digits=4)
    binning_process.fit(data_, data_['Converted'])
    for var in data_.columns:
        optb = binning_process.get_binned_variable(var)
        optb_table = optb.binning_table.build()
        optb_table = optb_table[optb_table['Event'] > 0]
        l_binning_table[var] = optb_table
        l.append([var,
                  optb_table.loc['Totals', 'IV'],
                  optb_table.shape[0] - 1,
                  optb_table.loc[optb_table.drop('Totals')['Count'].idxmax(), 'Bin'],
                  optb_table.drop('Totals')['Count'].max()
                 ])
    iv = pd.DataFrame(l, columns=['variable', 'iv', 'unique_bin', 'top_bin', 'freq_bin'])
    df_binning_table = pd.concat(l_binning_table, axis=0)
    return binning_process, iv, df_binning_table
```
## 5. Model Performance Summary
Baseline Logistic Regression:
Accuracy: 0.81
AUC Score: 0.88
Optimized Logistic Regression:
Accuracy: 0.789
AUC Score: 0.940
Optimized XGBoost:
Accuracy: 0.904
AUC Score: 0.966

## 6. Threshold Optimization for XGBoost
You can adjust the decision threshold to optimize accuracy. The code below demonstrates how to select the best threshold using the predicted probabilities:
```python
import numpy as np
from sklearn.metrics import accuracy_score

thresholds = np.arange(0.4, 0.9, 0.05)

# Get the predicted probabilities
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Initialize variables to store the best threshold and corresponding accuracy
best_threshold = 0.0
best_accuracy = 0.0

# Iterate through each threshold
for threshold in thresholds:
    # Apply the threshold to get binary predictions
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    
    # Calculate the accuracy for the current threshold
    accuracy = accuracy_score(y_test, y_pred_threshold)
    
    # Update the best threshold if the current accuracy is higher
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"Best Threshold: {best_threshold}")
print(f"Best Accuracy: {best_accuracy}")
```
For the optimized XGBoost model, the optimal threshold is 0.45, yielding an accuracy of 0.906.

## This markdown file outlines the necessary steps, from data exploration to model evaluation and threshold tuning. Let me know if you'd like to make any adjustments or additions!
