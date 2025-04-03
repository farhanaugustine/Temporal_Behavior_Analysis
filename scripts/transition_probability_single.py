import os
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import io # Import io for capturing print output


def calculate_transition_probabilities(csv_file_path, class_labels):
    """Calculates transition probabilities."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return f"Error: CSV file not found or empty: {csv_file_path}", None # Return error string
    except Exception as e:
        return f"Error reading CSV: {e}", None # Return error string

    transitions = defaultdict(lambda: defaultdict(int))
    for i in range(len(df) - 1):
        current_label = df['Class Label'][i]
        next_label = df['Class Label'][i + 1]
        transitions[current_label][next_label] += 1

    probabilities = defaultdict(lambda: defaultdict(float))
    for current, next_counts in transitions.items():
        total = sum(next_counts.values())
        if total >0: #Ensure total is greater than zero
          for next_label, count in next_counts.items():
              probabilities[current][next_label] = count / total

    return None, probabilities # Return None for error, and results

def save_transitions_to_excel(transition_probs, output_folder, video_name):
    """Saves transition probabilities to an Excel file."""
    if transition_probs:
        excel_path = os.path.join(output_folder, f"{video_name}_transition_probabilities.xlsx")
        try:
            rows = []
            for current_label, next_probs in transition_probs.items():
                for next_label, prob in next_probs.items():
                    rows.append({"Current Behavior": current_label, "Next Behavior": next_label, "Probability": prob})
            df_transitions = pd.DataFrame(rows)
            df_transitions.to_excel(excel_path, sheet_name='Transition Probabilities', index=False)
            return f"Transition probabilities saved to: {excel_path}" # Return success message
        except Exception as e:
             return f"Error writing to excel file: {e}" # Return error message
    return "No transition probability data to save." # Return if no data


def create_transition_heatmap(df, video_name, output_folder):
    """Generates and saves a transition heatmap for a single video."""
    behaviors = sorted(set(df['Current Behavior'].unique()) | set(df['Next Behavior'].unique()))
    heatmap_data = np.zeros((len(behaviors), len(behaviors)))
    behavior_to_index = {behavior: index for index, behavior in enumerate(behaviors)}

    plt_output_buffer = io.StringIO() # Capture plot save output

    for _, row in df.iterrows():
        current_behavior = row['Current Behavior']
        next_behavior = row['Next Behavior']
        probability = row['Probability']
        row_index = behavior_to_index[current_behavior]
        col_index = behavior_to_index[next_behavior]
        heatmap_data[row_index][col_index] = probability

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=behaviors, yticklabels=behaviors,
                cbar_kws={'label': 'Transition Probability'})
    plt.xlabel("Next Behavior")
    plt.ylabel("Current Behavior")
    plt.title(f"Transition Probabilities for {video_name}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_path = os.path.join(output_folder, f"{video_name}_transition_heatmap.png")
    plt.savefig(output_path)
    plt.close()
    message_heatmap_saved = f"Heatmap saved for {video_name} to: {output_path}"
    print(message_heatmap_saved)
    plt_output_buffer.write(message_heatmap_saved) # Capture output message
    plot_output = plt_output_buffer.getvalue()
    plt_output_buffer.close()
    return plot_output # Return captured plot message


def main_analysis(output_folder, class_labels, frame_rate, video_name): # Keyword args
    """Main function to parse arguments and run analysis."""

    csv_output_folder = os.path.join(output_folder, "csv_output")
    os.makedirs(csv_output_folder, exist_ok=True)

    if not isinstance(class_labels, dict):
        return "Error: Class labels must be a dictionary." # Return error string

    csv_file_path = os.path.join(csv_output_folder, f"{video_name}_analysis.csv")
    if not os.path.exists(csv_file_path):
        return f"Error: {csv_file_path} not found.  Run general_analysis.py first." # Return error string

    error_prob_calc, transition_probabilities = calculate_transition_probabilities(csv_file_path, class_labels) # Get potential error and results
    if error_prob_calc: # If calculate_transition_probabilities returned an error string
        return error_prob_calc # Return error string to GUI

    excel_output_msg = save_transitions_to_excel(transition_probabilities, output_folder, video_name) # Save to excel and get message
    heatmap_output_msg = "" # Initialize heatmap message
    if transition_probabilities: # Only create heatmap if probabilities were calculated
        # Create a DataFrame from the calculated probabilities for heatmap function
        rows = []
        for current_label, next_probs in transition_probabilities.items():
            for next_label, prob in next_probs.items():
                rows.append({"Current Behavior": current_label, "Next Behavior": next_label, "Probability": prob})
        df_transitions = pd.DataFrame(rows)
        heatmap_output_msg = create_transition_heatmap(df_transitions, video_name, output_folder) # Create heatmap and get message, pass df


    output_messages = [msg for msg in [excel_output_msg, heatmap_output_msg] if msg is not None] # Collect non-None messages
    return "\n".join(output_messages) if output_messages else "Transition probability analysis completed. No specific output messages." # Return combined messages


if __name__ == "__main__":
    # Example for direct testing:
    output_folder_path = "path/to/your/output_folder" # Replace with real path
    class_labels_dict = {0: "Exploration", 1: "Grooming", 2: "Jump", 3: "Wall-Rearing", 4: "Rear"}
    frame_rate_val = 30
    video_name_val = "your_video_name" # Replace with real video name

    class Args: # Dummy Args class for testing (not needed for GUI)
        def __init__(self, output_folder, class_labels, frame_rate, video_name):
            self.output_folder = output_folder
            self.class_labels = str(class_labels) # Pass as string for direct test
            self.frame_rate = frame_rate
            self.video_name = video_name

    test_args = Args(output_folder_path, class_labels_dict, frame_rate_val, video_name_val)

    # Simulate command-line execution (original main) - not needed for GUI
    # main(test_args)

    # Direct call to main_analysis for testing:
    output_message = main_analysis(output_folder=test_args.output_folder,
                                  class_labels=test_args.class_labels, # Pass dict directly
                                  frame_rate=test_args.frame_rate,
                                  video_name=test_args.video_name)
    print(output_message) # Print output for direct test