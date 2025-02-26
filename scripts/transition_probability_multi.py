import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def create_combined_transition_heatmap(all_data, output_folder):
    """Generates and saves a combined transition heatmap."""
    combined_df = pd.concat(all_data, ignore_index=True)

    # --- Check for empty combined_df ---
    if combined_df.empty:
        print("Error: Combined DataFrame is empty. Cannot create heatmap.")
        return

    # --- Find Unique Behaviors (and handle potential empty sets) ---
    current_behaviors = combined_df['Current Behavior'].unique()
    next_behaviors = combined_df['Next Behavior'].unique()
    behaviors = sorted(set(current_behaviors) | set(next_behaviors))

    if not behaviors:
        print("Error: No behaviors found in the data. Cannot create heatmap.")
        return

    behavior_to_index = {behavior: index for index, behavior in enumerate(behaviors)}
    transition_counts = np.zeros((len(behaviors), len(behaviors)))

    for _, row in combined_df.iterrows():
        current_behavior = row['Current Behavior']
        next_behavior = row['Next Behavior']

        # --- Check if behaviors are in the mapping ---
        if current_behavior not in behavior_to_index or next_behavior not in behavior_to_index:
            print(f"Warning: Skipping row with unknown behavior(s): '{current_behavior}', '{next_behavior}'")
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
    print(f"Combined heatmap saved to {output_path}")

def analyze_transition_files(transition_files_path, output_folder):
    """Analyzes multiple transition files, generates heatmaps."""
    os.makedirs(output_folder, exist_ok=True)
    all_dataframes = []

    for filename in os.listdir(transition_files_path):
        if filename.endswith("_transition_probabilities.xlsx"):
            filepath = os.path.join(transition_files_path, filename)
            try:
                df = pd.read_excel(filepath)

                # --- Check for empty DataFrame ---
                if df.empty:
                    print(f"Skipping file {filename}: DataFrame is empty.")
                    continue

                if not all(col in df.columns for col in ['Current Behavior', 'Next Behavior', 'Probability']):
                    print(f"Skipping file {filename}: Missing required columns.")
                    continue
                if df.isnull().values.any():
                    print(f"Skipping file {filename}: Contains missing values.")
                    continue
                if not pd.api.types.is_numeric_dtype(df['Probability']):
                    print(f"Skipping file {filename}: 'Probability' column must be numeric.")
                    continue
                if not all(0 <= p <= 1 for p in df['Probability']):
                    print(f"Skipping file {filename}: 'Probability' values must be between 0 and 1.")
                    continue

                all_dataframes.append(df)

            except FileNotFoundError:
                print(f"File not found: {filepath}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    if all_dataframes:
        create_combined_transition_heatmap(all_dataframes, output_folder) #creates heatmap from dataframes
    else:
        print("No valid data found to create a combined heatmap.")


def main():
    """Main function to parse arguments and run multi-video analysis."""
    parser = argparse.ArgumentParser(description="Combine transition probabilities from multiple videos and generate a combined heatmap.")
    parser.add_argument("--transition_folder", required=True, help="Path to the folder containing transition probability Excel files.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for combined results.")
    args = parser.parse_args()

    analyze_transition_files(args.transition_folder, args.output_folder)

if __name__ == "__main__":
    main()