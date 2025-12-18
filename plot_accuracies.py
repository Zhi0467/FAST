import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Define the results directory
    results_dir = 'Results'
    
    # Find all accuracies.csv files in subdirectories of Results
    file_pattern = os.path.join(results_dir, '*', 'accuracies.csv')
    files = glob.glob(file_pattern)
    
    if not files:
        print("No accuracies.csv files found in Results subdirectories.")
        return

    data = []

    # Read each file and store the data
    for file_path in files:
        # Get the parent directory name as the model/experiment name
        folder_name = os.path.basename(os.path.dirname(file_path))
        
        try:
            df = pd.read_csv(file_path)
            # Ensure required columns exist
            if 'Accuracy' in df.columns:
                # Add the folder name to the dataframe
                df['Model'] = folder_name
                data.append(df)
            else:
                print(f"Warning: 'Accuracy' column not found in {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not data:
        print("No valid data found to plot.")
        return

    # Concatenate all dataframes
    all_data = pd.concat(data, ignore_index=True)

    # Calculate summary statistics for printing
    summary = all_data.groupby('Model')['Accuracy'].agg(['mean', 'std', 'count']).reset_index()
    print("\nSummary Statistics:")
    print(summary)

    # Set up the plot style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Create a bar plot with error bars (representing the standard deviation or confidence interval)
    # ci='sd' shows standard deviation. Change to ci=95 for 95% confidence interval
    ax = sns.barplot(x='Model', y='Accuracy', data=all_data, ci='sd', capsize=.1)

    # Rotate x-axis labels for better readability if they are long
    plt.xticks(rotation=45, ha='right')
    plt.title('Comparison of Model Accuracies')
    plt.tight_layout()

    # Save the plot
    output_path = 'Results/model_accuracies_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()

