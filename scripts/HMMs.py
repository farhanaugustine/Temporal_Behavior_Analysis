import pandas as pd
import numpy as np
from hmmlearn import hmm  # Requires: pip install hmmlearn
import io 
import os
def analyze_hmm(csv_file_path, n_components=5, random_state=42):
    """
    Analyzes behavior sequences using a Hidden Markov Model (HMM).
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        return f"Error: CSV file not found: {csv_file_path}" # Return error string
    except Exception as e:
        return f"Error reading CSV: {e}" # Return error string

    # Convert behavior labels to numerical values
    behavior_labels = df['Class Label'].unique()
    label_map = {label: i for i, label in enumerate(behavior_labels)}
    numerical_data = df['Class Label'].map(label_map).values.reshape(-1, 1)  # Reshape for HMM

    # Build and fit the HMM
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", random_state=random_state)
    model.fit(numerical_data)

    # Predict hidden states
    hidden_states = model.predict(numerical_data)

    # Analyze state transitions
    transitions = []
    for i in range(1, len(hidden_states)):
        transitions.append((hidden_states[i - 1], hidden_states[i]))

    transition_counts = pd.Series(transitions).value_counts()
    transition_counts.index = transition_counts.index.map(lambda x: (f"State {x[0]}", f"State {x[1]}"))

    # Analyze emissions
    emission_probs = {}
    for state in range(n_components):
        state_indices = np.where(hidden_states == state)[0]
        emitted_behaviors = df['Class Label'].iloc[state_indices].tolist()
        behavior_counts = pd.Series(emitted_behaviors).value_counts(normalize=True)
        emission_probs[state] = behavior_counts

    state_labels = assign_state_labels(emission_probs)  # Assign labels

    # Capture print output
    output_buffer = io.StringIO()
    
    output_buffer.write("\n--- HMM Transition Counts (Hidden States) ---\n")
    output_buffer.write(transition_counts.to_string())
    output_buffer.write("\n\n--- Emission Probabilities (Behavior | Hidden State) ---\n")
    for state, probs in emission_probs.items():
        output_buffer.write(f"State {state} ({state_labels[state]}):\n") # State Label
        if not probs.empty:
            output_buffer.write(probs.to_string() + "\n")
        else:
            output_buffer.write("No emissions found in this state.\n")

    output_string = output_buffer.getvalue()
    output_buffer.close()
    return None, output_string # Return None for error, and output string


def assign_state_labels(emission_probs):
    """Assigns descriptive labels to hidden states based on emission probabilities."""
    state_labels = {}
    for state, probs in emission_probs.items():
        if not probs.empty:
            most_likely_behavior = probs.idxmax()
            state_labels[state] = f"Dominated by {most_likely_behavior}"
        else:
            state_labels[state] = "Unused State"  # Handle states with no emissions
    return state_labels


def main_analysis(csv_file, n_components=5, random_state=42): # Keyword args with defaults
    """Main function to run HMM analysis."""

    if not os.path.exists(csv_file):
        return f"Error: CSV file not found: {csv_file}" # Error string

    error_hmm, analysis_output = analyze_hmm(csv_file, n_components, random_state) # Get potential error and output
    if error_hmm: # If analyze_hmm returned an error string
        return error_hmm # Return error string to GUI

    return analysis_output # Return analysis output string


if __name__ == "__main__":
    # Example for direct testing:
    csv_file_path_test = "path/to/your/csv_output/your_video_name_analysis.csv" # Replace
    n_components_test = 3
    random_state_test = 0

    class Args: # Dummy Args class for testing (not needed for GUI)
        def __init__(self, csv_file, n_components, random_state):
            self.csv_file = csv_file
            self.n_components = n_components
            self.random_state = random_state

    test_args = Args(csv_file_path_test, n_components_test, random_state_test)

    # Simulate command-line execution (original main) - not needed for GUI
    # main(test_args)

    # Direct call to main_analysis for testing:
    output_message = main_analysis(csv_file=test_args.csv_file, 
                                  n_components=test_args.n_components,
                                  random_state=test_args.random_state)
    print(output_message) # Print output for direct test