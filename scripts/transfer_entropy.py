import os
import pandas as pd
import itertools
import ast 
import argparse
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt  
from sklearn.preprocessing import LabelEncoder  
import io # Import io for capturing print output


def calculate_transfer_entropy_no_lib(source, target, k=1, conditioning=None, laplace_smoothing=0.0):
    """
    Calculates transfer entropy from source to target WITHOUT using pyinform.
    """
    n = len(source)
    if len(target) != n:
        raise ValueError("Source and target series must have the same length.")

    if n <= k:
        return 0.0  # Not enough data

    joint_counts = defaultdict(int)  
    target_counts = defaultdict(int) 

    for t in range(k, n - 1):  # Start at k; end at n-1

        source_past = tuple(source[t-k:t])
        target_past = tuple(target[t-k:t])

        if conditioning is not None:
            conditioning_past = tuple(conditioning[t-k:t, :].flatten()) 
        else:
            conditioning_past = ()  # Empty tuple if no conditioning

        joint_state = (target[t+1], target_past, source_past, conditioning_past)
        target_state = (target[t+1], target_past, conditioning_past) 

        joint_counts[joint_state] += 1
        target_counts[target_state] += 1

    te = 0.0
    for joint_state, joint_count in joint_counts.items():
        target_future, target_past, source_past, conditioning_past = joint_state

        target_state = (target_future, target_past, conditioning_past)
        target_count = target_counts[target_state]

        past_state = (target_past, conditioning_past)
        num_seen_joint = sum(1 for js in joint_counts if js[1:] == (target_past, source_past, conditioning_past))
        num_seen_target = sum(1 for ts in target_counts if ts[1:] == (target_past, conditioning_past))

        if num_seen_joint > 0 and num_seen_target > 0:
          p_joint = (joint_count + laplace_smoothing) / (num_seen_joint + laplace_smoothing * 2)
          p_target = (target_count + laplace_smoothing) / (num_seen_target + laplace_smoothing * 2)
          te += (joint_count / n) * np.log(p_joint / p_target)
    return te


def calculate_transfer_entropy(csv_file_path, class_labels, max_lag_frames=150, k=3, laplace_smoothing=0.0):
    """Calculates (conditional) transfer entropy."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return f"Error: CSV file not found or empty: {csv_file_path}", None # Return error string
    except Exception as e:
        return f"Error reading CSV: {e}", None # Return error string

    all_series = {}
    for class_label in class_labels.values():
        all_series[class_label] = pd.Series([1 if label == class_label else 0 for label in df['Class Label']])

    combined_series = pd.concat(all_series.values()).astype('category')
    encoder = LabelEncoder().fit(combined_series)
    for label, series in all_series.items():
        all_series[label] = encoder.transform(series).astype(np.int32) # NumPy array, integers

    results = {}

    for (class1, class2) in itertools.product(class_labels.values(), repeat=2):
        if class1 == class2:
            continue  # Skip same source and target

        source = all_series[class1]
        target = all_series[class2]

        conditioning_vars = [] # Conditioning variables array
        for label in class_labels.values():
            if label != class1 and label != class2:
                conditioning_vars.append(all_series[label])  

        if conditioning_vars:
            conditioning_vars = np.column_stack(conditioning_vars)
        else:
             conditioning_vars = None

        try:
            te = calculate_transfer_entropy_no_lib(source, target, k, conditioning_vars, laplace_smoothing)
        except Exception as e:
            error_message = f"Error calculating TE from {class1} to {class2}: {e}"
            print(error_message)
            return error_message, None # Return error string, and None for results
        results[(class1, class2)] = te

    return None, results # Return None for error, and results


def save_te_to_excel(te_results, output_folder, video_name, k):
    """Saves transfer entropy results to Excel file."""
    if te_results:
        excel_path = os.path.join(output_folder, f"{video_name}_transfer_entropy_k{k}.xlsx")  # k in filename
        try:
            df_te = pd.DataFrame([{"Source Behavior": c1, "Target Behavior": c2, "Transfer Entropy": te}
                                  for (c1, c2), te in te_results.items()])
            df_te.to_excel(excel_path, sheet_name="Transfer Entropy", index=False)
            return f"Transfer entropy results saved to: {excel_path}" # Return success message
        except Exception as e:
             return f"Error writing to Excel file: {e}" # Return error message
    return "No transfer entropy data to save." # Return if no data


def main_analysis(output_folder, class_labels, video_name, k_values="1,5,10", laplace_smoothing=0.1, transition_threshold=5): # Keyword args
    """Main function to run Transfer Entropy analysis."""

    csv_output_folder = os.path.join(output_folder, "csv_output") # Corrected: output_folder instead of args.output_folder
    os.makedirs(csv_output_folder, exist_ok=True)

    if not isinstance(class_labels, dict):
        return "Error: Class labels must be a dictionary." # Error string

    csv_file_path = os.path.join(csv_output_folder, f"{video_name}_analysis.csv") # Corrected: video_name instead of args.video_name
    if not os.path.exists(csv_file_path):
        return f"Error: {csv_file_path} not found. Run general_analysis.py first." # Error string

    # Parse k_values from command line argument
    try:
        k_values_list = [int(k) for k in k_values.split(",")] # Corrected: k_values instead of args.k_values
    except ValueError:
        return "Error: Invalid k values. Must be comma-separated integers." # Error string

    all_te_results = {} # Store all TE results
    output_messages = [] # List to collect messages

    # ------ Data Analysis (Before TE Calculation) ------
    df = pd.read_csv(csv_file_path)

    # --- Behavior Frequencies ---
    behavior_counts = df['Class Label'].value_counts()
    behavior_freq_output = io.StringIO() # Capture output
    behavior_freq_output.write("\n--- Behavior Frequencies ---\n")
    behavior_freq_output.write(behavior_counts.to_string())
    behavior_freq_str = behavior_freq_output.getvalue()
    behavior_freq_output.close()
    print(behavior_freq_str)
    output_messages.append(behavior_freq_str)


    # --- Transition Counts ---
    transition_counts = {}
    transition_warning_messages = [] # To collect warning messages
    transition_counts_output = io.StringIO() # Capture output
    transition_counts_output.write("\n--- Transition Counts ---\n")

    for behavior in class_labels.values():
        transitions = 0
        series = pd.Series([1 if label == behavior else 0 for label in df['Class Label']])
        for i in range(1, len(series)):
            if series[i] != series[i-1]:
                transitions += 1
        transition_counts[behavior] = transitions
        transition_counts_output.write(f"{behavior}: {transitions} transitions\n")
        if transitions < transition_threshold:
            warning_message = f"Warning: {behavior} has very few transitions. TE results may be unreliable."
            transition_warning_messages.append(warning_message) # Collect warning
            transition_counts_output.write(warning_message + "\n") # Log warning too

    transition_counts_str = transition_counts_output.getvalue()
    transition_counts_output.close()
    print(transition_counts_str)
    output_messages.append(transition_counts_str)
    output_messages.extend(transition_warning_messages) # Add warnings to output messages


    # --- Pairwise Co-occurrence ---
    cooccurrence_matrix = pd.DataFrame(index=class_labels.values(), columns=class_labels.values())
    for class1, class2 in itertools.product(class_labels.values(), repeat=2):
        cooccurrence = len(df[(df['Class Label'] == class1) & (df['Class Label'] == class2)]) #Co-occurrence
        cooccurrence_matrix.loc[class1, class2] = cooccurrence

    cooccurrence_output = io.StringIO() # Capture output
    cooccurrence_output.write("\n--- Pairwise Co-occurrence Matrix ---\n")
    cooccurrence_output.write(cooccurrence_matrix.to_string())
    cooccurrence_str = cooccurrence_output.getvalue()
    cooccurrence_output.close()
    print(cooccurrence_str)
    output_messages.append(cooccurrence_str)
    # --------------------------------------------------

    plot_paths = [] # List to collect plot paths

    for k in k_values_list:
        message_calculating_te = f"Calculating TE for k = {k}"
        print(message_calculating_te)
        output_messages.append(message_calculating_te)

        error_te, te_results = calculate_transfer_entropy(csv_file_path, class_labels_dict, 150, k, laplace_smoothing) # Corrected: Removed args.max_lag, used 150 directly, and corrected laplace_smoothing to keyword arg
        if error_te: # If calculate_transfer_entropy returned an error string
            return error_te # Return error string to GUI
        all_te_results[k] = te_results
        excel_output_msg = save_te_to_excel(te_results, output_folder, video_name, k) # Corrected: output_folder, video_name instead of args.output_folder, args.video_name
        if excel_output_msg:
            output_messages.append(excel_output_msg) # Add excel message


    behavior_pairs = list(itertools.combinations(class_labels_dict.values(), 2)) # Unique behavior pairs

    for class1, class2 in behavior_pairs: # Iterate through all pairs
        te_values_class1_to_class2 = []
        te_values_class2_to_class1 = []

        for k in k_values_list: # Collect TE values
            te_values_class1_to_class2.append(all_te_results[k].get((class1, class2), None))
            te_values_class2_to_class1.append(all_te_results[k].get((class2, class1), None))

        # Remove None values for plotting if any error occurred
        te_values_class1_to_class2_plot = [te for te in te_values_class1_to_class2 if te is not None] #Plotting data
        te_values_class2_to_class1_plot = [te for te in te_values_class2_to_class1 if te is not None] #Plotting data


        # Plot Transfer Entropy vs k
        plt.figure(figsize=(10, 6)) # Adjust figure size if needed
        plt.plot(k_values_list[:len(te_values_class1_to_class2_plot)], te_values_class1_to_class2_plot, marker='o', linestyle='-', label=f'{class1} -> {class2}') #Plot
        plt.plot(k_values_list[:len(te_values_class2_to_class1_plot)], te_values_class2_to_class1_plot, marker='o', linestyle='-', label=f'{class2} -> {class1}') #Plot
        plt.xlabel("Embedding Dimension (k)")
        plt.ylabel("Transfer Entropy")
        plt.title(f"Transfer Entropy vs. k for {class1} and {class2}")
        plt.legend()

        plot_path = os.path.join(output_folder, f"TE_vs_k_{class1}_{class2}.png") # Corrected: output_folder instead of args.output_folder
        plt.savefig(plot_path) # Save plot
        plt.close()
        message_plot_saved = f"Plot of TE vs k saved to {plot_path}"
        print(message_plot_saved)
        output_messages.append(message_plot_saved)
        plot_paths.append(plot_path)


    return "\n".join(output_messages) # Return combined messages

if __name__ == "__main__":
    # Example for direct testing:
    output_folder_path = "path/to/your/output_folder" # Replace with real path
    class_labels_dict = {0: "Exploration", 1: "Grooming", 2: "Jump", 3: "Wall-Rearing", 4: "Rear"}
    frame_rate_val = 30
    video_name_val = "your_video_name" # Replace with real video name
    k_values_test = "1,3" # Test k values

    class Args: # Dummy Args class for testing (not needed for GUI)
        def __init__(self, output_folder, class_labels, frame_rate, video_name, max_lag, k_values, laplace_smoothing, transition_threshold):
            self.output_folder = output_folder
            self.class_labels = str(class_labels) # Pass as string for direct test
            self.frame_rate = frame_rate
            self.video_name = video_name
            self.max_lag = max_lag # Not used in TE, but kept for consistency
            self.k_values = k_values
            self.laplace_smoothing = laplace_smoothing
            self.transition_threshold = transition_threshold

    test_args = Args(output_folder_path, class_labels_dict, frame_rate_val, video_name_val, 150, k_values_test, 0.1, 5)

    # Simulate command-line execution (original main) - not needed for GUI
    # main(test_args)

    # Direct call to main_analysis for testing:
    output_message = main_analysis(output_folder=test_args.output_folder,
                                  class_labels=class_labels_dict, # Pass dict directly
                                  video_name=test_args.video_name,
                                  k_values=test_args.k_values,
                                  laplace_smoothing=test_args.laplace_smoothing,
                                  transition_threshold=test_args.transition_threshold)
    print(output_message) # Print output for direct test