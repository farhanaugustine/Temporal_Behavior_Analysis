import pandas as pd
import numpy as np
from hmmlearn import hmm
import os
import matplotlib.pyplot as plt  
import networkx as nx  

def analyze_hmm_multi(input_folder, output_folder, n_components=5, random_state=42):
    """
    Analyzes behavior sequences from multiple CSV files using HMMs.
    """

    all_transition_counts = {}
    all_emission_probs = {}
    all_transition_matrices = {}  # Store transition matrices
    output_messages = [] # List to collect messages

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            video_name = os.path.splitext(filename)[0]
            csv_file_path = os.path.join(input_folder, filename)

            message_video_analyzing = f"\nAnalyzing video: {video_name}"
            print(message_video_analyzing)
            output_messages.append(message_video_analyzing)

            try:
                df = pd.read_csv(csv_file_path)
            except FileNotFoundError:
                message_file_not_found = f"Error: CSV file not found: {csv_file_path}"
                print(message_file_not_found)
                output_messages.append(message_file_not_found)
                continue  # Skip to next file
            except Exception as e:
                message_csv_read_error = f"Error reading CSV {filename}: {e}"
                print(message_csv_read_error)
                output_messages.append(message_csv_read_error)
                continue

            # Convert behavior labels to numerical values
            behavior_labels = df['Class Label'].unique()
            label_map = {label: i for i, label in enumerate(behavior_labels)}
            numerical_data = df['Class Label'].map(label_map).values.reshape(-1, 1)

            # Build and fit the HMM
            model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", random_state=random_state)
            try:
                model.fit(numerical_data)
            except Exception as e:
                message_hmm_fit_error = f"Error fitting HMM for {filename}: {e}"
                print(message_hmm_fit_error)
                output_messages.append(message_hmm_fit_error)
                continue

            # Predict hidden states
            hidden_states = model.predict(numerical_data)

            # Analyze state transitions
            transitions = []
            for i in range(1, len(hidden_states)):
                transitions.append((hidden_states[i - 1], hidden_states[i]))

            transition_counts = pd.Series(transitions).value_counts()
            transition_counts.index = transition_counts.index.map(lambda x: (f"State {x[0]}", f"State {x[1]}"))
            all_transition_counts[video_name] = transition_counts  # Store

            # Analyze emissions
            emission_probs = {}
            for state in range(n_components):
                state_indices = np.where(hidden_states == state)[0]
                emitted_behaviors = df['Class Label'].iloc[state_indices].tolist()
                behavior_counts = pd.Series(emitted_behaviors).value_counts(normalize=True)
                emission_probs[state] = behavior_counts
            all_emission_probs[video_name] = emission_probs  # Store

            state_labels = assign_state_labels(emission_probs)  # Assign state labels

            # Calculate transition matrix
            transition_matrix = np.zeros((n_components, n_components))
            for i in range(len(hidden_states) - 1):
                current_state = hidden_states[i]
                next_state = hidden_states[i + 1]
                transition_matrix[current_state, next_state] += 1
            transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True) # Normalize
            all_transition_matrices[video_name] = transition_matrix # Save

            message_transition_counts = f"\n--- HMM Transition Counts (Hidden States) - {video_name} ---\n{transition_counts.to_string()}"
            print(message_transition_counts)
            output_messages.append(message_transition_counts)

            message_emission_probs_header = f"\n--- Emission Probabilities (Behavior | Hidden State) - {video_name} ---"
            print(message_emission_probs_header)
            output_messages.append(message_emission_probs_header)
            for state, probs in emission_probs.items():
                state_emission_output = f"State {state} ({state_labels[state]}):\n" # State Label
                if not probs.empty:
                    state_emission_output += probs.to_string()
                else:
                    state_emission_output += "No emissions found in this state."
                print(state_emission_output)
                output_messages.append(state_emission_output)

            # --- Visualization ---
            plot_messages = visualize_hmm_results(video_name, emission_probs, transition_matrix, behavior_labels, state_labels, output_folder) # Get plot messages
            if plot_messages:
                output_messages.extend(plot_messages) # Add plot messages

    # Combine and save results (CSV)
    combined_transitions_df = pd.DataFrame(all_transition_counts).fillna(0)
    combined_emissions_df = {} # Nested dict, first key video name, second is state

    # Process emissions data: DataFrame
    all_behaviors = set() # Collect all behaviors
    for video_name, emission_probs in all_emission_probs.items():
        for state, probs in emission_probs.items():
            all_behaviors.update(probs.index)
    all_behaviors = sorted(list(all_behaviors))  # Sorted behaviors

    combined_emissions_data = {} # Process emissions data: Build DataFrame
    for video_name, emission_probs in all_emission_probs.items():
        combined_emissions_data[video_name] = {}
        for state in range(n_components):  # Ensure all states present
            if state in emission_probs:
                state_data = emission_probs[state].reindex(all_behaviors, fill_value=0) # Add missing
            else:
                state_data = pd.Series(index=all_behaviors, data=0)  # Zero if state DNE
            combined_emissions_data[video_name][state] = state_data

    index = pd.MultiIndex.from_product([combined_emissions_data.keys(), range(n_components)], names=['Video', 'State']) # Convert to DataFrame, multilevel index
    all_emissions_df = pd.DataFrame(index=index, columns=all_behaviors)  # create empty dataframe
    for video_name, state_data in combined_emissions_data.items():
        for state, probs in state_data.items():
            all_emissions_df.loc[(video_name, state)] = probs.values # Add to Dataframe

    transitions_output_path = os.path.join(output_folder, "combined_hmm_transitions.csv")
    emissions_output_path = os.path.join(output_folder, "combined_hmm_emissions.csv")
    combined_transitions_df.to_csv(transitions_output_path)
    all_emissions_df.to_csv(emissions_output_path)
    message_transitions_saved = f"\nCombined HMM transition counts saved to: {transitions_output_path}"
    print(message_transitions_saved)
    output_messages.append(message_transitions_saved)
    message_emissions_saved = f"Combined HMM emission probabilities saved to: {emissions_output_path}"
    print(message_emissions_saved)
    output_messages.append(message_emissions_saved)

    # Save transition matrices
    transition_matrices_output_path = os.path.join(output_folder, "hmm_transition_matrices.csv")
    data = [] # Prepare data for CSV export
    for video, matrix in all_transition_matrices.items():
        for i in range(n_components):
            for j in range(n_components):
                data.append([video, f"State {i}", f"State {j}", matrix[i, j]])
    transition_df = pd.DataFrame(data, columns=['Video', 'From State', 'To State', 'Probability']) # Create DataFrame
    transition_df.to_csv(transition_matrices_output_path, index=False) #Save
    message_matrices_saved = f"HMM transition matrices (probabilities) saved to: {transition_matrices_output_path}"
    print(message_matrices_saved)
    output_messages.append(message_matrices_saved)

    return "\n".join(output_messages) # Return combined messages


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

    plot_messages = [] # List to collect plot messages

    # --- 1. Emission Probability Bar Charts ---
    num_states = len(emission_probs)
    fig, axes = plt.subplots(1, num_states, figsize=(15, 5), sharey=True)  

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
                ax.set_ylabel("Probability", fontsize=10)  
        else:
            ax.text(0.5, 0.5, "No emissions", ha='center', va='center', fontsize=10)

    fig.suptitle(f"Emission Probabilities - {video_name}", fontsize=14)  
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
    emission_chart_path = os.path.join(output_folder, f"{video_name}_emission_probs.png")
    plt.savefig(emission_chart_path)
    plt.close(fig)  
    message_emission_chart_saved = f"Emission probabilities chart saved: {emission_chart_path}"
    print(message_emission_chart_saved)
    plot_messages.append(message_emission_chart_saved)

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
    message_transition_diagram_saved = f"Transition diagram saved: {transition_diagram_path}"
    print(message_transition_diagram_saved)
    plot_messages.append(message_transition_diagram_saved)

    return plot_messages # Return plot messages


def main_analysis(input_folder, output_folder, n_components=2, random_state=42): # Keyword args with defaults
    """Main function to run multi-video HMM analysis."""

    if not os.path.isdir(input_folder):
        return f"Error: Input folder not found: {input_folder}" # Error string
    os.makedirs(output_folder, exist_ok=True)

    analysis_messages = analyze_hmm_multi(input_folder, output_folder, n_components, random_state) # Get analysis messages
    return "\n".join(analysis_messages) # Return combined messages


if __name__ == "__main__":
    # Example for direct testing:
    input_folder_path = r"C:\Users\Aegis-MSI\Documents\DeerMice_Yolov11_Re-analysis\Videos\XLmodel_30-min_videoAnalysis\labels\HMMs_multi" # Replace
    output_folder_path = r"C:\Users\Aegis-MSI\Documents\DeerMice_Yolov11_Re-analysis\Videos\XLmodel_30-min_videoAnalysis\labels\HMMs_multi" # Replace
    n_components_test = 5
    random_state_test = 42

    class Args: # Dummy Args class for testing (not needed for GUI)
        def __init__(self, input_folder, output_folder, n_components, random_state):
            self.input_folder = input_folder
            self.output_folder = output_folder
            self.n_components = n_components
            self.random_state = random_state

    test_args = Args(input_folder_path, output_folder_path, n_components_test, random_state_test)

    # Simulate command-line execution (original main) - not needed for GUI
    # main(test_args)

    # Direct call to main_analysis for testing:
    output_message = main_analysis(input_folder=test_args.input_folder, 
                                  output_folder=test_args.output_folder,
                                  n_components=test_args.n_components, 
                                  random_state=test_args.random_state)
    print(output_message) # Print output for direct test