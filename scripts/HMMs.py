import pandas as pd
import numpy as np
from hmmlearn import hmm  # Requires: pip install hmmlearn
import argparse

def analyze_hmm(csv_file_path, n_components=2, random_state=42):
    """
    Analyzes behavior sequences using a Hidden Markov Model (HMM).

    Args:
        csv_file_path (str): Path to the CSV file containing the behavior data.
                             CSV should have 'Class Label' column.
        n_components (int): Number of hidden states in the HMM.
        random_state (int): Random seed for reproducibility.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Convert behavior labels to numerical values
    behavior_labels = df['Class Label'].unique()
    label_map = {label: i for i, label in enumerate(behavior_labels)}
    numerical_data = df['Class Label'].map(label_map).values.reshape(-1, 1)  # Reshape for HMM
    reverse_label_map = {i: label for label, i in label_map.items()} #For reverse look up, not used in this part yet but will be helpful for future

    # Build and fit the HMM
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", random_state=random_state)
    model.fit(numerical_data)

    # Predict hidden states
    hidden_states = model.predict(numerical_data)

    # Analyze state transitions, using reverse mapping to be more descriptive
    transitions = []
    for i in range(1, len(hidden_states)):
        transitions.append((hidden_states[i - 1], hidden_states[i]))

    #Count transitions and map the transitions
    transition_counts = pd.Series(transitions).value_counts()
    transition_counts.index = transition_counts.index.map(lambda x: (f"State {x[0]}", f"State {x[1]}")) #Changed
    print("\n--- HMM Transition Counts (Hidden States) ---")
    print(transition_counts)

    # Analyze emissions
    emission_probs = {}
    for state in range(n_components):
        state_indices = np.where(hidden_states == state)[0]
        emitted_behaviors = df['Class Label'].iloc[state_indices].tolist()

        behavior_counts = pd.Series(emitted_behaviors).value_counts(normalize=True)
        emission_probs[state] = behavior_counts

    print("\n--- Emission Probabilities (Behavior | Hidden State) ---")
    state_labels = assign_state_labels(emission_probs)  # Assign labels

    for state, probs in emission_probs.items():
      if not probs.empty:  # Handle states with no emissions
        print(f"State {state} ({state_labels[state]}):")  # State Label
        print(probs)
      else:
        print(f"State {state} ({state_labels[state]}): No emissions found in this state.")

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

def main():
    parser = argparse.ArgumentParser(description="Run HMM analysis on behavioral data.")
    parser.add_argument("--csv_file", required=True, help="Path to the CSV file containing the behavior data.")
    parser.add_argument("--n_components", type=int, default=2, help="Number of hidden states in the HMM (default: 2).")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility (default: 42).")

    args = parser.parse_args()

    analyze_hmm(args.csv_file, n_components=args.n_components, random_state=args.random_state)

if __name__ == "__main__":
    main()