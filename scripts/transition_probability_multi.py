import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys  

def create_combined_transition_heatmap(all_data, output_folder):
    """Generates and saves a combined transition heatmap."""
    combined_df = pd.concat(all_data, ignore_index=True)

    output_messages = [] # List to collect messages

    if combined_df.empty:
        error_message = "Error: Combined DataFrame is empty. Cannot create heatmap."
        logging.error(error_message)
        output_messages.append(error_message)
        return output_messages # Return error messages, no paths

    current_behaviors = combined_df['Current Behavior'].unique()
    next_behaviors = combined_df['Next Behavior'].unique()
    behaviors = sorted(set(current_behaviors) | set(next_behaviors))

    if not behaviors:
        error_message = "Error: No behaviors found in the data. Cannot create heatmap."
        logging.error(error_message)
        output_messages.append(error_message)
        return output_messages # Return error messages, no paths


    behavior_to_index = {behavior: index for index, behavior in enumerate(behaviors)}
    transition_counts = np.zeros((len(behaviors), len(behaviors)))

    for _, row in combined_df.iterrows():
        current_behavior = row['Current Behavior']
        next_behavior = row['Next Behavior']

        if current_behavior not in behavior_to_index or next_behavior not in behavior_to_index:
            warning_message = f"Warning: Skipping row with unknown behavior(s): '{current_behavior}', '{next_behavior}'"
            logging.warning(warning_message)
            output_messages.append(warning_message)
            continue

        row_index = behavior_to_index[current_behavior]
        col_index = behavior_to_index[next_behavior]
        transition_counts[row_index][col_index] += 1  # Count transitions

    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transition_probabilities = transition_counts / row_sums

    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_probabilities, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=behaviors, yticklabels=behaviors,
                cbar_kws={'label': 'Combined Transition Probability'})
    plt.xlabel("Next Behavior")
    plt.ylabel("Current Behavior")
    plt.title("Combined Transition Probabilities Across All Videos")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_path = os.path.join(output_folder, "combined_transition_heatmap.png")  # Simplified filename
    plt.savefig(output_path)
    plt.close()
    message_heatmap_saved = f"Combined heatmap saved to {output_path}"
    logging.info(message_heatmap_saved)
    print(message_heatmap_saved)
    output_messages.append(message_heatmap_saved)

    return output_messages # Return messages, including path to heatmap


def analyze_transition_files(transition_files_path, output_folder):
    """Analyzes multiple transition files, generates heatmaps."""
    os.makedirs(output_folder, exist_ok=True)
    all_dataframes = []
    output_messages = [] # List to collect messages

    for filename in os.listdir(transition_files_path):
        if filename.endswith("_transition_probabilities.xlsx"):
            filepath = os.path.join(transition_files_path, filename)

            logging.info("FILEPATH " + filepath)

            try:
                df = pd.read_excel(filepath)

                if df.empty:
                    message_df_empty = f"Skipping file {filename}: DataFrame is empty."
                    logging.warning(message_df_empty)
                    output_messages.append(message_df_empty)
                    continue

                if not all(col in df.columns for col in ['Current Behavior', 'Next Behavior', 'Probability']):
                    message_missing_cols = f"Skipping file {filename}: Missing required columns."
                    logging.warning(message_missing_cols)
                    output_messages.append(message_missing_cols)
                    continue
                if df.isnull().values.any():
                    message_missing_values = f"Skipping file {filename}: Contains missing values."
                    logging.warning(message_missing_values)
                    output_messages.append(message_missing_values)
                    continue
                if not pd.api.types.is_numeric_dtype(df['Probability']):
                    message_prob_not_numeric = f"Skipping file {filename}: 'Probability' column must be numeric."
                    logging.warning(message_prob_not_numeric)
                    output_messages.append(message_prob_not_numeric)
                    continue
                if not all(0 <= p <= 1 for p in df['Probability']):
                    message_prob_out_of_range = f"Skipping file {filename}: 'Probability' values must be between 0 and 1."
                    logging.warning(message_prob_out_of_range)
                    output_messages.append(message_prob_out_of_range)
                    continue

                all_dataframes.append(df)

            except FileNotFoundError as e:
                error_message_file_not_found = f"File not found: {filepath}, caused by {e}"
                logging.error(error_message_file_not_found)
                output_messages.append(error_message_file_not_found)
            except Exception as e:
                error_message_processing = f"Error processing {filename}: {e}"
                logging.error(error_message_processing)
                output_messages.append(error_message_processing)

    if all_dataframes:
        heatmap_messages = create_combined_transition_heatmap(all_dataframes, output_folder) # Get heatmap messages
        if heatmap_messages:
            output_messages.extend(heatmap_messages) #Extend main messages list
    else:
        message_no_valid_data = "No valid data found to create a combined heatmap."
        logging.warning(message_no_valid_data)
        output_messages.append(message_no_valid_data)
        print("No valid data found to create a combined heatmap.")

    return output_messages # Return combined messages


def main_analysis(transition_folder, output_folder): # Keyword args
    """Main function to run multi-video transition probability analysis."""

    if not os.path.isdir(transition_folder):
        return f"Error: Transition folder not found: {transition_folder}" # Error string

    analysis_messages = analyze_transition_files(transition_folder, output_folder) # Get analysis messages
    return "\n".join(analysis_messages) # Return combined messages


if __name__ == "__main__":
    # Example for direct testing:
    transition_folder_path = "path/to/your/transition_probabilities_excel_folder" # Replace
    output_folder_path = "path/to/your/output_folder" # Replace

    class Args: # Dummy Args class for testing (not needed for GUI)
        def __init__(self, transition_folder, output_folder):
            self.transition_folder = transition_folder
            self.output_folder = output_folder

    test_args = Args(transition_folder_path, output_folder_path)

    # Simulate command-line execution (original main) - not needed for GUI
    # main(test_args)

    # Direct call to main_analysis for testing:
    output_message = main_analysis(transition_folder=test_args.transition_folder, 
                                  output_folder=test_args.output_folder)
    print(output_message) # Print output for direct test