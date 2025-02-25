import os
import pandas as pd
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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


def create_combined_transition_heatmap(all_data, output_folder, file_prefix=""):
    """Generates and saves a combined transition heatmap."""
    combined_df = pd.concat(all_data, ignore_index=True)
    behaviors = sorted(set(combined_df['Current Behavior'].unique()) | set(combined_df['Next Behavior'].unique()))
    behavior_to_index = {behavior: index for index, behavior in enumerate(behaviors)}
    transition_counts = np.zeros((len(behaviors), len(behaviors)))

    for _, row in combined_df.iterrows():
        current_behavior = row['Current Behavior']
        next_behavior = row['Next Behavior']
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
    output_path = os.path.join(output_folder, f"{file_prefix}combined_transition_heatmap.png")
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
                video_name = os.path.basename(filename).replace("_transition_probabilities.xlsx", "")
                create_transition_heatmap(df, video_name, output_folder)

            except FileNotFoundError:
                print(f"File not found: {filepath}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    if all_dataframes:
        create_combined_transition_heatmap(all_dataframes, output_folder)
    else:
        print("No valid data found to create a combined heatmap.")


def main():
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(description="Calculate transition probabilities and generate heatmaps.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder.")
    parser.add_argument("--class_labels", required=True, help="Class labels dictionary (as a string).")
    parser.add_argument("--frame_rate", required=True, type=int, help="Frame rate of the video.")
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
    
    #Create a folder for combined analysis:
    excel_output_folder = os.path.join(args.output_folder, "transition_probabilities_excel")
    if not os.path.exists(excel_output_folder):
        os.makedirs(excel_output_folder)

     # Copy the transition_probabilities excels to that folder
    source_excel_path = os.path.join(args.output_folder, f"{args.video_name}_transition_probabilities.xlsx") #Original file
    destination_excel_path = os.path.join(excel_output_folder, f"{args.video_name}_transition_probabilities.xlsx") #New destination
    if os.path.exists(source_excel_path):
        try:
            import shutil
            shutil.copy(source_excel_path, destination_excel_path)
            print(f"Copied transition_probabilities Excel file to: {destination_excel_path}")
        except Exception as e:
            print(f"Error copying file: {e}")
    
    # Perform the combined analysis (multiple videos)
    analyze_transition_files(excel_output_folder, args.output_folder)

if __name__ == "__main__":
    main()