import os
import pandas as pd
import argparse
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_basic_correlations(csv_file_path, class_labels):
    """Calculates simple correlations between behaviors."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Error: CSV file not found or empty: {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    bout_starts = defaultdict(lambda: [])

    for class_label in class_labels.values():
        df['bout_start'] = ((df['Class Label'] == class_label) &
                            ((df['Frame Number'] == 1) |
                             (df['Class Label'].shift(1) != class_label)))
        bout_starts[class_label] = df[df['bout_start'] == True].index.tolist()

    correlations = {}
    for (class1, class2) in itertools.combinations(class_labels.values(), 2):
        indices1 = set(bout_starts[class1])
        indices2 = set(bout_starts[class2])
        serie1 = pd.Series([i in indices1 for i in range(len(df))])
        serie2 = pd.Series([i in indices2 for i in range(len(df))])
        corr_coeff = serie1.corr(serie2)
        correlations[(class1, class2)] = corr_coeff

    return correlations

def save_correlations_to_excel(correlations, output_folder, video_name):
    """Saves correlation results to Excel."""
    if correlations:
        excel_path = os.path.join(output_folder, f"{video_name}_basic_correlations.xlsx")
        try:
            rows = []
            for (class1, class2), corr in correlations.items():
                rows.append({"Behavior 1": class1, "Behavior 2": class2, "Correlation": corr})
            df_correlations = pd.DataFrame(rows)
            df_correlations.to_excel(excel_path, sheet_name="Basic Correlations", index=False)
            print(f"Basic correlations saved to: {excel_path}")
        except Exception as e:
             print(f"Error writing to excel file: {e}")

def plot_correlations(correlations, output_folder, video_name):
    """Generates a heatmap of the correlation matrix."""
    if not correlations:
        print("No correlation data to plot.")
        return

    # Convert the correlations dictionary to a DataFrame for plotting
    df_corr = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
    df_corr.index.names = ['Behavior 1', 'Behavior 2']
    df_corr = df_corr.reset_index()

    # Create a pivot table for the heatmap
    df_pivot = df_corr.pivot(index='Behavior 1', columns='Behavior 2', values='Correlation')

    # Get the list of behaviors to ensure the matrix is square and includes all
    behaviors = sorted(list(set([b for pair in correlations.keys() for b in pair])))
    
    # Reindex the pivot table to include all behaviors, filling NaNs with 0
    df_pivot = df_pivot.reindex(index=behaviors, columns=behaviors, fill_value=0)
    
    # Ensure symmetry:  The correlation of (A, B) should be the same as (B, A)
    for i in range(len(behaviors)):
        for j in range(i, len(behaviors)):
            b1 = behaviors[i]
            b2 = behaviors[j]
            # If (b1, b2) is in correlations, use that value; otherwise, it's 0.
            if (b1,b2) in correlations:
                val = correlations[(b1, b2)]
            elif (b2, b1) in correlations:
                val = correlations[(b2, b1)]
            else:
                val = 0  # Default if neither is present

            df_pivot.loc[b1, b2] = val
            df_pivot.loc[b2, b1] = val  # Ensure symmetry


    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title(f"Correlation Matrix of Behaviors - {video_name}")
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_folder, f"{video_name}_correlation_heatmap.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Correlation heatmap saved to: {output_path}")


def main():
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(description="Calculate basic correlations between behaviors.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder.")
    parser.add_argument("--class_labels", required=True, help="Class labels dictionary (as a string).")
    parser.add_argument("--frame_rate", required = True, type = int, help = "Frame Rate of the video")
    parser.add_argument("--video_name", required=True, help="Video name.")
    args = parser.parse_args()

    csv_output_folder = os.path.join(args.output_folder, "csv_output")
    os.makedirs(csv_output_folder, exist_ok=True)

    try:
        class_labels_dict = eval(args.class_labels)
        if not isinstance(class_labels_dict, dict):
            raise ValueError("Class labels must be a dictionary.")
    except (ValueError, SyntaxError) as e:
        print(f"Error: Invalid class labels: {e}")
        return

    csv_file_path = os.path.join(csv_output_folder, f"{args.video_name}_analysis.csv")
    if os.path.exists(csv_file_path):
       correlation_results = calculate_basic_correlations(csv_file_path, class_labels_dict)
       save_correlations_to_excel(correlation_results, args.output_folder, args.video_name)
       if correlation_results:  # Only plot if there are results
            plot_correlations(correlation_results, args.output_folder, args.video_name)
    else:
        print(f"Error: {csv_file_path} not found.  Run general_analysis.py first.")

if __name__ == "__main__":
    main()