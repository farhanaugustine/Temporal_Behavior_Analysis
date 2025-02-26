import os
import pandas as pd
import argparse
import itertools
import pyinform
import ast
import numpy as np
from sklearn.preprocessing import LabelEncoder

def calculate_transfer_entropy(csv_file_path, class_labels, max_lag_frames=150, k=3): #Added k
    """Calculates (conditional) transfer entropy using pyinform directly."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Error: CSV file not found or empty: {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    # 1. Prepare the data: Convert to integer time series.
    all_series = {}
    for class_label in class_labels.values():
        # Create the binary series (1 if behavior is present, 0 otherwise)
        all_series[class_label] = pd.Series([1 if label == class_label else 0 for label in df['Class Label']])

    # 2. Label Encode: This is necessary if the values are string labels.
    combined_series = pd.concat(all_series.values()).astype('category')
    encoder = LabelEncoder().fit(combined_series)
    for label, series in all_series.items():
        all_series[label] = encoder.transform(series)  # NOW it's a NumPy array

    results = {}

    # 3. Iterate through all PAIRS of behaviors.
    for (class1, class2) in itertools.product(class_labels.values(), repeat=2):
        if class1 == class2:
            continue  # Skip if source and target are the same

        # 4. Calculate Conditional Transfer Entropy.
        source = all_series[class1].astype(np.int32)  # Explicitly cast to int32
        target = all_series[class2].astype(np.int32)  # Explicitly cast to int32

        # Create the conditioning variables array.
        conditioning_vars = []
        for label in class_labels.values():
            if label != class1 and label != class2:
                conditioning_vars.append(all_series[label])  # Append the NumPy array

        # Stack the conditioning variables into a 2D array, or set to None if empty
        if conditioning_vars:
            conditioning_vars = np.column_stack(conditioning_vars).astype(np.int32)  # Explicit cast
        else:
             conditioning_vars = None
        #Debugging lines (uncomment to use):
        #print(f"Source: {class1}, shape: {source.shape}")
        #print(f"Target: {class2}, shape: {target.shape}")

        #if conditioning_vars is not None:
        #    print(f"Conditioning variables shape: {conditioning_vars.shape}")
        #else:
        #    print("No conditioning variables")
        try:

            te = pyinform.transfer_entropy(source, target, k=k, local=False, condition=conditioning_vars) #k is the embedding length
            #print(f"TE from {class1} to {class2}: {te:.4f}")  # Debugging print

        except Exception as e:
            print(f"Error calculating TE from {class1} to {class2}: {e}")
            te = None

        results[(class1, class2)] = te

    return results

def save_te_to_excel(te_results, output_folder, video_name):
    """Saves transfer entropy results to an Excel file."""
    if te_results:
        excel_path = os.path.join(output_folder, f"{video_name}_transfer_entropy.xlsx")
        try:
            df_te = pd.DataFrame([{"Source Behavior": c1, "Target Behavior": c2, "Transfer Entropy": te}
                                  for (c1, c2), te in te_results.items()])
            df_te.to_excel(excel_path, sheet_name="Transfer Entropy", index=False)
            print(f"Transfer entropy results saved to: {excel_path}")
        except Exception as e:
            print(f"Error writing to Excel file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Perform Transfer Entropy analysis.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder.")
    parser.add_argument("--class_labels", required=True, help="Class labels dictionary (as a string).")
    parser.add_argument("--frame_rate", required=True, type=int, help="Frame rate.")  # Keep, but not used
    parser.add_argument("--video_name", required=True, help="Video name.")
    parser.add_argument("--max_lag", type=int, default=150, help="Maximum lag in frames (default: 150).") #Keep, but not used
    parser.add_argument("--k", type=int, default=3, help="Embedding dimension k (default: 3).") # Add k
    args = parser.parse_args()

    csv_output_folder = os.path.join(args.output_folder, "csv_output")
    os.makedirs(csv_output_folder, exist_ok=True)

    try:
        class_labels_dict = ast.literal_eval(args.class_labels)
        if not isinstance(class_labels_dict, dict):
            raise ValueError("Class labels must be a dictionary.")
    except (ValueError, SyntaxError) as e:
        print(f"Error: Invalid class labels format: {e}")
        return

    csv_file_path = os.path.join(csv_output_folder, f"{args.video_name}_analysis.csv")
    if not os.path.exists(csv_file_path):
        print(f"Error: {csv_file_path} not found. Run general_analysis.py first.")
        return

    te_results = calculate_transfer_entropy(csv_file_path, class_labels_dict, args.max_lag, args.k) # Pass k
    save_te_to_excel(te_results, args.output_folder, args.video_name)

if __name__ == "__main__":
    main()