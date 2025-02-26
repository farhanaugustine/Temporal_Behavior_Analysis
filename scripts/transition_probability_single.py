import os
import pandas as pd
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast

def calculate_transition_probabilities(csv_file_path, class_labels):
    """Calculates transition probabilities."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Error: CSV file not found or empty: {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

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

    return probabilities

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
            print(f"Transition probabilities saved to: {excel_path}")
        except Exception as e:
             print(f"Error writing to excel file: {e}")

def create_transition_heatmap(df, video_name, output_folder):
    """Generates and saves a transition heatmap for a single video."""
    behaviors = sorted(set(df['Current Behavior'].unique()) | set(df['Next Behavior'].unique()))
    heatmap_data = np.zeros((len(behaviors), len(behaviors)))
    behavior_to_index = {behavior: index for index, behavior in enumerate(behaviors)}

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
    print(f"Heatmap saved for {video_name}")

def main():
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(description="Calculate transition probabilities and generate a heatmap for a single video.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder.")
    parser.add_argument("--class_labels", required=True, help="Class labels dictionary (as a string).")
    parser.add_argument("--frame_rate", required=True, type=int, help="Frame rate of the video.")
    parser.add_argument("--video_name", required=True, help="Video name.")
    args = parser.parse_args()

    csv_output_folder = os.path.join(args.output_folder, "csv_output")
    os.makedirs(csv_output_folder, exist_ok=True)

    try:
        class_labels_dict = ast.literal_eval(args.class_labels)  # Use ast.literal_eval
        if not isinstance(class_labels_dict, dict):
            raise ValueError("Class labels must be a dictionary.")
    except (ValueError, SyntaxError) as e:
        print(f"Error: Invalid class labels: {e}")
        return

    csv_file_path = os.path.join(csv_output_folder, f"{args.video_name}_analysis.csv")
    if not os.path.exists(csv_file_path):
        print(f"Error: {csv_file_path} not found.  Run general_analysis.py first.")
        return

    transition_probabilities = calculate_transition_probabilities(csv_file_path, class_labels_dict)

    if transition_probabilities:
        save_transitions_to_excel(transition_probabilities, args.output_folder, args.video_name)

        # Create a DataFrame from the calculated probabilities for the single video
        rows = []
        for current_label, next_probs in transition_probabilities.items():
            for next_label, prob in next_probs.items():
                rows.append({"Current Behavior": current_label, "Next Behavior": next_label, "Probability": prob})
        df_transitions = pd.DataFrame(rows)

        # Generate individual heatmap for the current video
        create_transition_heatmap(df_transitions, args.video_name, args.output_folder)

if __name__ == "__main__":
    main()