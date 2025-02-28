import pandas as pd
import numpy as np
from hmmlearn import hmm
import argparse
import os
import matplotlib.pyplot as plt  # Import matplotlib for visualization
import networkx as nx  # For the state transition diagram

def analyze_hmm_multi(input_folder, output_folder, n_components=2, random_state=42):
    """
    Analyzes behavior sequences from multiple CSV files using HMMs.

    Args:
        input_folder (str): Path to the folder containing CSV files (one per video).
        output_folder (str): Path to the folder to save the results.
        n_components (int): Number of hidden states in the HMM.
        random_state (int): Random seed for reproducibility.
    """

    all_transition_counts = {}
    all_emission_probs = {}
    all_transition_matrices = {}  # Store transition matrices

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            video_name = os.path.splitext(filename)[0]
            csv_file_path = os.path.join(input_folder, filename)

            print(f"\nAnalyzing video: {video_name}")

            try:
                df = pd.read_csv(csv_file_path)
            except FileNotFoundError:
                print(f"Error: CSV file not found: {csv_file_path}")
                continue  # Skip to the next file
            except Exception as e:
                print(f"Error reading CSV {filename}: {e}")
                continue

            # Convert behavior labels to numerical values
            behavior_labels = df['Class Label'].unique()
            label_map = {label: i for i, label in enumerate(behavior_labels)}
            numerical_data = df['Class Label'].map(label_map).values.reshape(-1, 1)
            reverse_label_map = {i: label for label, i in label_map.items()}  #For reverse look up, not used in this part yet but will be helpful for future

            # Build and fit the HMM
            model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", random_state=random_state)
            try:
                model.fit(numerical_data)
            except Exception as e:
                print(f"Error fitting HMM for {filename}: {e}")
                continue

            # Predict hidden states
            hidden_states = model.predict(numerical_data)

            # Analyze state transitions
            transitions = []
            for i in range(1, len(hidden_states)):
                transitions.append((hidden_states[i - 1], hidden_states[i]))

            transition_counts = pd.Series(transitions).value_counts()
            transition_counts.index = transition_counts.index.map(lambda x: (f"State {x[0]}", f"State {x[1]}"))

            all_transition_counts[video_name] = transition_counts  # Store transition counts

            # Analyze emissions
            emission_probs = {}
            for state in range(n_components):
                state_indices = np.where(hidden_states == state)[0]
                emitted_behaviors = df['Class Label'].iloc[state_indices].tolist()

                behavior_counts = pd.Series(emitted_behaviors).value_counts(normalize=True)
                emission_probs[state] = behavior_counts

            all_emission_probs[video_name] = emission_probs  # Store emission probabilities

            state_labels = assign_state_labels(emission_probs)  # Assign labels to states

             # Calculate transition matrix
            transition_matrix = np.zeros((n_components, n_components))
            for i in range(len(hidden_states) - 1):
                current_state = hidden_states[i]
                next_state = hidden_states[i + 1]
                transition_matrix[current_state, next_state] += 1

            # Normalize rows to get transition probabilities
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            transition_matrix = transition_matrix / row_sums
            all_transition_matrices[video_name] = transition_matrix #Save

            print(f"\n--- HMM Transition Counts (Hidden States) - {video_name} ---")
            print(transition_counts)

            print(f"\n--- Emission Probabilities (Behavior | Hidden State) - {video_name} ---")
            for state, probs in emission_probs.items():
                if not probs.empty:  # Handle states with no emissions
                    print(f"State {state} ({state_labels[state]}):")  # State Label
                    print(probs)
                else:
                    print(f"State {state} ({state_labels[state]}): No emissions found in this state.")

            # --- Visualization ---
            visualize_hmm_results(video_name, emission_probs, transition_matrix, behavior_labels, state_labels, output_folder)

    # Combine and save results (example: CSV)
    combined_transitions_df = pd.DataFrame(all_transition_counts).fillna(0)
    combined_emissions_df = {} # Nested dictionary, first key is video name, second is the state

    # Process emissions data: Fill any missing state column with all the behavior labels with 0
    all_behaviors = set()
    for video_name, emission_probs in all_emission_probs.items():
        for state, probs in emission_probs.items():
            all_behaviors.update(probs.index)

    all_behaviors = sorted(list(all_behaviors))  # Sorted all behaviors to maintain order

    # Process emissions data: Build a DataFrame
    combined_emissions_data = {}
    for video_name, emission_probs in all_emission_probs.items():
        combined_emissions_data[video_name] = {}
        for state in range(n_components):  # Ensure all states are present
            if state in emission_probs:
                # Insert the known probabilites
                state_data = emission_probs[state].reindex(all_behaviors, fill_value=0) #Add missing
            else:
                state_data = pd.Series(index=all_behaviors, data=0)  # Zero if the state doesn't exist

            combined_emissions_data[video_name][state] = state_data

    # Convert to DataFrame, multilevel index
    index = pd.MultiIndex.from_product([combined_emissions_data.keys(), range(n_components)], names=['Video', 'State'])
    all_emissions_df = pd.DataFrame(index=index, columns=all_behaviors)  # create empty dataframe
    for video_name, state_data in combined_emissions_data.items():
        for state, probs in state_data.items():
            all_emissions_df.loc[(video_name, state)] = probs.values #Add to Dataframe

    transitions_output_path = os.path.join(output_folder, "combined_hmm_transitions.csv")
    emissions_output_path = os.path.join(output_folder, "combined_hmm_emissions.csv")
    combined_transitions_df.to_csv(transitions_output_path)
    all_emissions_df.to_csv(emissions_output_path)

    # Save transition matrices
    transition_matrices_output_path = os.path.join(output_folder, "hmm_transition_matrices.csv")
    # Prepare data for CSV export
    data = []
    for video, matrix in all_transition_matrices.items():
        for i in range(n_components):
            for j in range(n_components):
                data.append([video, f"State {i}", f"State {j}", matrix[i, j]])

    # Create DataFrame and save to CSV
    transition_df = pd.DataFrame(data, columns=['Video', 'From State', 'To State', 'Probability'])
    transition_df.to_csv(transition_matrices_output_path, index=False)

    print(f"\nCombined HMM transition counts saved to: {transitions_output_path}")
    print(f"Combined HMM emission probabilities saved to: {emissions_output_path}")
    print(f"HMM transition matrices (probabilities) saved to: {transition_matrices_output_path}")

def assign_state_labels(emission_probs): #No change
    """Assigns descriptive labels to hidden states based on emission probabilities."""
    state_labels = {}
    for state, probs in emission_probs.items():
        if not probs.empty:
            most_likely_behavior = probs.idxmax()
            state_labels[state] = f"Dominated by {most_likely_behavior}"
        else:
            state_labels[state] = "Unused State"  # Handle states with no emissions
    return state_labels

def visualize_hmm_results(video_name, emission_probs, transition_matrix, behavior_labels, state_labels, output_folder):
    """Generates and saves visualizations for HMM results."""

    # --- 1. Emission Probability Bar Charts ---
    num_states = len(emission_probs)
    fig, axes = plt.subplots(1, num_states, figsize=(15, 5), sharey=True)  # Adjusted figure size

    for i in range(num_states):
        ax = axes[i]
        state_label = state_labels[i]
        probs = emission_probs[i]
        if not probs.empty:
            ax.bar(probs.index, probs.values)
            ax.set_title(f"State {i} ({state_label})", fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel("Behavior", fontsize=10)
            if i == 0:
                ax.set_ylabel("Probability", fontsize=10)  # Only label the y-axis once
        else:
            ax.text(0.5, 0.5, "No emissions", ha='center', va='center', fontsize=10)

    fig.suptitle(f"Emission Probabilities - {video_name}", fontsize=14)  # Added overall title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
    emission_chart_path = os.path.join(output_folder, f"{video_name}_emission_probs.png")
    plt.savefig(emission_chart_path)
    plt.close(fig)  # Close figure to free memory
    print(f"Emission probabilities chart saved: {emission_chart_path}")

    # --- 2. State Transition Diagram ---
    G = nx.DiGraph()

    # Add nodes with labels
    for i in range(num_states):
        state_label = state_labels[i]
        G.add_node(f"State {i}: {state_label}")  # Use State i: Label as node name

    for i in range(num_states):
        for j in range(num_states):
            prob = transition_matrix[i, j]
            from_node = f"State {i}: {state_labels[i]}" #Set up label
            to_node = f"State {j}: {state_labels[j]}" #Set up label
            G.add_edge(from_node, to_node, weight=prob) #Update new nodes

    pos = nx.circular_layout(G) #Layout
    plt.figure(figsize=(16, 12))  # Make the figure bigger

    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=8, font_weight="bold", alpha=0.7)
    edge_labels = {(i, j): f"{data['weight']:.2f}" for i, j, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f"State Transition Diagram - {video_name}", fontsize=14) #Chart Title
    plt.tight_layout()
    transition_diagram_path = os.path.join(output_folder, f"{video_name}_transition_diagram.png")
    plt.savefig(transition_diagram_path)
    plt.close()
    print(f"Transition diagram saved: {transition_diagram_path}")

def main():
    parser = argparse.ArgumentParser(description="Run HMM analysis on multiple behavioral data files.")
    parser.add_argument("--input_folder", required=True, help="Path to the folder containing the CSV files.")
    parser.add_argument("--output_folder", required=True, help="Path to the folder to save the combined results.")
    parser.add_argument("--n_components", type=int, default=2, help="Number of hidden states in the HMM (default: 2).")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility (default: 42).")

    args = parser.parse_args()

    analyze_hmm_multi(args.input_folder, args.output_folder, args.n_components, args.random_state)

if __name__ == "__main__":
    main()