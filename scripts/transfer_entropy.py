import os
import pandas as pd
import argparse
import itertools
import ast
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt  # Import for plotting (install if needed)
from sklearn.preprocessing import LabelEncoder  # Keep for the LabelEncoder

def calculate_transfer_entropy_no_lib(source, target, k=1, conditioning=None, laplace_smoothing=0.0):
    """
    Calculates transfer entropy from source to target WITHOUT using pyinform.

    Args:
        source (np.ndarray): Source time series (integer encoded).
        target (np.ndarray): Target time series (integer encoded).
        k (int): Embedding dimension (length of past history).
        conditioning (np.ndarray, optional): Conditioning variables (2D array). Defaults to None.
        laplace_smoothing (float): Laplace smoothing parameter to avoid zero probabilities.

    Returns:
        float: Transfer entropy value.
    """

    n = len(source)
    if len(target) != n:
        raise ValueError("Source and target series must have the same length.")

    if n <= k:
        return 0.0  # Not enough data

    # Create dictionaries to store counts of state transitions.  We use defaultdict
    # because it automatically creates entries if they don't exist.
    joint_counts = defaultdict(int)  # Counts P(target(t+1) | target(t:t-k+1), source(t:t-k+1), conditioning...)
    target_counts = defaultdict(int) # Counts P(target(t+1) | target(t:t-k+1), conditioning...)

    # Iterate through the time series to count state transitions
    for t in range(k, n - 1):  # Start at k to have enough history; end at n-1 to have t+1

        # Create the state vectors (histories)
        source_past = tuple(source[t-k:t])
        target_past = tuple(target[t-k:t])

        # Handle conditioning variables
        if conditioning is not None:
            conditioning_past = tuple(conditioning[t-k:t, :].flatten()) #Flatten the conditioning variables
        else:
            conditioning_past = ()  # Empty tuple if no conditioning

        # Construct the joint state (past and future)
        joint_state = (target[t+1], target_past, source_past, conditioning_past)
        target_state = (target[t+1], target_past, conditioning_past) # State without source
        #PRINT
        #print(f"Joint state {joint_state}")

        # Increment the counts
        joint_counts[joint_state] += 1
        target_counts[target_state] += 1

    # Calculate probabilities and KL divergence
    te = 0.0
    for joint_state, joint_count in joint_counts.items():
        # Extract components of the joint state
        target_future, target_past, source_past, conditioning_past = joint_state

        # Calculate probabilities
        #p_joint = joint_count / n # Normalize counts by the number of steps
        #Find the number of times the target state was seen
        target_state = (target_future, target_past, conditioning_past)
        target_count = target_counts[target_state]

        #p_target = target_count / n

        #KL divergence is only calculated if p_target is non-zero
        #if p_target > 0:
            # Find the number of times the past history was seen to normalize
        past_state = (target_past, conditioning_past)
        num_seen_joint = sum(1 for js in joint_counts if js[1:] == (target_past, source_past, conditioning_past))
        num_seen_target = sum(1 for ts in target_counts if ts[1:] == (target_past, conditioning_past))
        #PRINT
        #print(f"Target future, target past, source past and conditioning past: {target_future}, {target_past}, {source_past} and {conditioning_past}")
        #print(f"NUm Seen target {num_seen_target}")

        if num_seen_joint > 0 and num_seen_target > 0:
          p_joint = (joint_count + laplace_smoothing) / (num_seen_joint + laplace_smoothing * 2)
          p_target = (target_count + laplace_smoothing) / (num_seen_target + laplace_smoothing * 2)
          #print(f"Pjoint: {p_joint} P target: {p_target}")
          te += (joint_count / n) * np.log(p_joint / p_target)
    return te

def calculate_transfer_entropy(csv_file_path, class_labels, max_lag_frames=150, k=3, laplace_smoothing=0.0):
    """Calculates (conditional) transfer entropy."""
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
        all_series[label] = encoder.transform(series).astype(np.int32) # NOW it's a NumPy array and it's integers!

    results = {}

    # 3. Iterate through all PAIRS of behaviors.
    for (class1, class2) in itertools.product(class_labels.values(), repeat=2):
        if class1 == class2:
            continue  # Skip if source and target are the same

        # 4. Calculate Conditional Transfer Entropy.
        source = all_series[class1]
        target = all_series[class2]

        # Create the conditioning variables array.
        conditioning_vars = []
        for label in class_labels.values():
            if label != class1 and label != class2:
                conditioning_vars.append(all_series[label])  # Append the NumPy array

        # Stack the conditioning variables into a 2D array, or set to None if empty
        if conditioning_vars:
            conditioning_vars = np.column_stack(conditioning_vars)
        else:
             conditioning_vars = None
        #PRINT
        #print("Source shape and data:", source.shape, source)
        #print("Target shape and data:", target.shape, target)
        #print("K and Laplace smoothing", k, laplace_smoothing)
        #print("Conditioning shape and data", conditioning_vars.shape if conditioning_vars is not None else None, conditioning_vars)

        try:

            te = calculate_transfer_entropy_no_lib(source, target, k, conditioning_vars, laplace_smoothing)
            #print(f"TE from {class1} to {class2}: {te:.4f}")  # Debugging print

        except Exception as e:
            print(f"Error calculating TE from {class1} to {class2}: {e}")
            te = None

        results[(class1, class2)] = te

    return results

def save_te_to_excel(te_results, output_folder, video_name, k):
    """Saves transfer entropy results to an Excel file."""
    if te_results:
        excel_path = os.path.join(output_folder, f"{video_name}_transfer_entropy_k{k}.xlsx")  # Include k in filename
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
    parser.add_argument("--k_values", type=str, default="1,5,10", help="Comma-separated list of k values (embedding dimensions).")
    parser.add_argument("--laplace_smoothing", type=float, default=0.1, help="Laplace smoothing parameter (default: 0.1).")
    parser.add_argument("--transition_threshold", type=int, default=5, help="Minimum number of transitions required for analysis (default: 5)")

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

    # Parse k_values from command line argument
    k_values = [int(k) for k in args.k_values.split(",")]

    all_te_results = {} # Store all results

    # ------ Data Analysis (Before TE Calculation) ------
    df = pd.read_csv(csv_file_path)

    # --- Behavior Frequencies ---
    behavior_counts = df['Class Label'].value_counts()
    print("\n--- Behavior Frequencies ---")
    print(behavior_counts)

    # --- Transition Counts ---
    transition_counts = {}
    for behavior in class_labels_dict.values():
        transitions = 0
        series = pd.Series([1 if label == behavior else 0 for label in df['Class Label']])
        for i in range(1, len(series)):
            if series[i] != series[i-1]:
                transitions += 1
        transition_counts[behavior] = transitions

    print("\n--- Transition Counts ---")
    for behavior, count in transition_counts.items():
        print(f"{behavior}: {count} transitions")
        if count < args.transition_threshold:
            print(f"Warning: {behavior} has very few transitions. TE results may be unreliable.")

    # --- Pairwise Co-occurrence ---
    # (Implementation depends on what co-occurrence means in your context. Simplest is just count joint occurrences in same frame)
    cooccurrence_matrix = pd.DataFrame(index=class_labels_dict.values(), columns=class_labels_dict.values())
    for class1, class2 in itertools.product(class_labels_dict.values(), repeat=2):
        cooccurrence = len(df[(df['Class Label'] == class1) & (df['Class Label'] == class2)]) #Number of times these appear together. If they can't be in the same frame, find an alternative calculation.
        cooccurrence_matrix.loc[class1, class2] = cooccurrence
    print("\n--- Pairwise Co-occurrence Matrix ---")
    print(cooccurrence_matrix)
    # --------------------------------------------------

    for k in k_values:
        print(f"Calculating TE for k = {k}")
        te_results = calculate_transfer_entropy(csv_file_path, class_labels_dict, args.max_lag, k, args.laplace_smoothing)
        all_te_results[k] = te_results
        save_te_to_excel(te_results, args.output_folder, args.video_name, k) # Save for each k

    # Plotting TE vs k
    behavior_pairs = list(itertools.combinations(class_labels_dict.values(), 2)) # Unique behavior pairs

    for class1, class2 in behavior_pairs: # Iterate through all pairs
        te_values_class1_to_class2 = []
        te_values_class2_to_class1 = []

        for k in k_values: # Collect TE values
            te_values_class1_to_class2.append(all_te_results[k].get((class1, class2), None))
            te_values_class2_to_class1.append(all_te_results[k].get((class2, class1), None))

        # Remove None values for plotting if any error occurred
        te_values_class1_to_class2 = [te for te in te_values_class1_to_class2 if te is not None]
        te_values_class2_to_class1 = [te for te in te_values_class2_to_class1 if te is not None]

        # Plot Transfer Entropy vs k
        plt.figure(figsize=(10, 6)) # Adjust figure size if needed
        plt.plot(k_values[:len(te_values_class1_to_class2)], te_values_class1_to_class2, marker='o', linestyle='-', label=f'{class1} -> {class2}') #Plot the data
        plt.plot(k_values[:len(te_values_class2_to_class1)], te_values_class2_to_class1, marker='o', linestyle='-', label=f'{class2} -> {class1}') #Plot the data
        plt.xlabel("Embedding Dimension (k)") # Give x axis a label
        plt.ylabel("Transfer Entropy") #Give y axis a label
        plt.title(f"Transfer Entropy vs. k for {class1} and {class2}") #Give the plot a title
        plt.legend() # add a legend so we know which is which

        plot_path = os.path.join(args.output_folder, f"TE_vs_k_{class1}_{class2}.png") #Create the plot path
        plt.savefig(plot_path) # Save the plot to a path
        plt.close() # close it or you might run out of memory

        print(f"Plot of TE vs k saved to {plot_path}") # give the user the save path

if __name__ == "__main__":
    main()