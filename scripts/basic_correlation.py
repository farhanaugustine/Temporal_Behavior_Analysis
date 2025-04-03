import os
import pandas as pd
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import ast 
from scipy import stats

def calculate_basic_correlations(csv_file_path, class_labels):
    """Calculates Spearman rank correlations between behaviors."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return f"Error: CSV file not found or empty: {csv_file_path}" # Changed to return error string
    except Exception as e:
        return f"Error reading CSV: {e}" # Changed to return error string

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
        serie1 = pd.Series([i in indices1 for i in range(len(df))]).astype(int)
        serie2 = pd.Series([i in indices2 for i in range(len(df))]).astype(int)
        corr_coeff, p_value =  stats.spearmanr(serie1, serie2)
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
            return f"Basic correlations saved to: {excel_path}" # Changed to return success string
        except Exception as e:
             return f"Error writing to excel file: {e}" # Changed to return error string
    return "No correlations to save." # Return string if no correlations


def plot_correlations(correlations, output_folder, video_name):
    """Generates a heatmap of the correlation matrix."""
    if not correlations:
        return "No correlation data to plot." # Changed to return string

    df_corr = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
    df_corr.index.name = 'Behavior Pair'
    df_corr = df_corr.reset_index()

    behaviors = sorted(list(set(b for pair in correlations for b in pair)))
    data = {}
    for b1 in behaviors:
        data[b1] = {}
        for b2 in behaviors:
            if (b1, b2) in correlations:
                data[b1][b2] = correlations[(b1, b2)]
            elif (b2, b1) in correlations:
                data[b1][b2] = correlations[(b2, b1)]
            else:
                data[b1][b2] = 0

    df_pivot = pd.DataFrame(data).loc[behaviors, behaviors]

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title(f"Correlation Matrix of Behaviors - {video_name}")
    plt.tight_layout()

    output_path = os.path.join(output_folder, f"{video_name}_correlation_heatmap.png")
    plt.savefig(output_path)
    plt.close()
    return f"Correlation heatmap saved to: {output_path}" # Changed to return success string


def main_analysis(output_folder, class_labels, frame_rate, video_name): # Modified to main_analysis and keyword args
    """Main analysis function to calculate and save basic correlations."""

    csv_output_folder = os.path.join(output_folder, "csv_output")
    os.makedirs(csv_output_folder, exist_ok=True)

    if not isinstance(class_labels, dict): # Class labels should already be dict
        return "Error: Class labels must be a dictionary." # Return error string

    csv_file_path = os.path.join(csv_output_folder, f"{video_name}_analysis.csv")
    if not os.path.exists(csv_file_path):
        return f"Error: {csv_file_path} not found. Run general_analysis.py first." # Return error string

    correlation_results = calculate_basic_correlations(csv_file_path, class_labels)
    if isinstance(correlation_results, str): # Check if calculate_basic_correlations returned an error string
        return correlation_results # Return the error string

    excel_output_msg = save_correlations_to_excel(correlation_results, output_folder, video_name)
    heatmap_output_msg = plot_correlations(correlation_results, output_folder, video_name)

    output_messages = [msg for msg in [excel_output_msg, heatmap_output_msg] if msg is not None] # Collect non-None messages
    return "\n".join(output_messages) if output_messages else "Basic correlation analysis completed. No specific output messages."


if __name__ == "__main__":
    from scipy import stats 
    # For direct execution (if needed for testing, though GUI will use main_analysis)
    # Example of how to run it directly with hardcoded arguments for testing:
    output_folder = "path/to/your/output_folder" # Replace with a real path for testing
    class_labels_dict = {0: "Exploration", 1: "Grooming", 2: "Jump", 3: "Wall-Rearing", 4: "Rear"}
    frame_rate = 30
    video_name = "your_video_name" # Replace with a real video name
    
    # Create dummy arguments (similar to argparse args for testing)
    class Args:
        def __init__(self, output_folder, class_labels, frame_rate, video_name):
            self.output_folder = output_folder
            self.class_labels = str(class_labels) #Pass as string for direct test
            self.frame_rate = frame_rate
            self.video_name = video_name

    test_args = Args(output_folder, class_labels_dict, frame_rate, video_name)

    # Simulate command-line execution for direct testing of main (now main_analysis)
    # You'd parse args like this if still using command line, but GUI will call main_analysis directly
    # main(test_args) # Original main used command line args
    
    # Instead, for direct testing of main_analysis, call it with keyword arguments:
    output_message = main_analysis(output_folder=test_args.output_folder, 
                                  class_labels=class_labels_dict, # Pass dict directly
                                  frame_rate=test_args.frame_rate, 
                                  video_name=test_args.video_name)
    print(output_message) # Print output message for direct test