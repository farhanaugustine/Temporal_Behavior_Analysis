import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx  # Requires: pip install networkx
import matplotlib.pyplot as plt
import io  # Import io for capturing print output
import os # Import os for path manipulation

def analyze_sequence_mining(csv_file_path, min_support=0.05, min_confidence=0.7):
    """
    Performs sequence mining and generates a directed graph of association rules.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        return f"Error: CSV file not found: {csv_file_path}", None # Return error string
    except Exception as e:
        return f"Error reading CSV file: {e}", None # Return error string

    # Convert behavior data into transaction data
    transactions = df['Class Label'].apply(lambda x: [x]).groupby(df['Bout ID']).sum().tolist()

    # Encode transaction data
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Find frequent itemsets using Apriori
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # Capture print output
    output_buffer = io.StringIO()
    output_buffer.write("\n--- Association Rules ---\n")
    output_buffer.write(rules.to_string())
    rule_output = output_buffer.getvalue()
    output_buffer.close()
    print(rule_output) # Keep print for direct execution

    # Create a directed graph from association rules
    G = nx.DiGraph()
    for index, row in rules.iterrows():
         if row['confidence'] >= 0.9:  # Adjust threshold - keep as is for now
            antecedents = ", ".join(row['antecedents'])
            consequents = ", ".join(row['consequents'])
            G.add_edge(antecedents, consequents, confidence=row['confidence'])

    pos = nx.spring_layout(G)  # Layout algorithm
    plt.figure(figsize=(14, 10)) # Adjusted size
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", alpha=0.7)

    edge_labels = nx.get_edge_attributes(G, 'confidence')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels) # Add edge labels

    plt.title("Association Rules Graph", fontsize=14) # Chart title
    plt.tight_layout()

    # Save graph
    graph_path = os.path.join(os.getcwd(), "association_rules_graph.png") # Save to current directory
    plt.savefig(graph_path)
    plt.close()
    message_graph_saved = f"Association rules graph saved as: {graph_path}"
    print(message_graph_saved) # Keep print for direct execution

    output_string = rule_output + "\n" + message_graph_saved # Combine rule output and graph message

    return None, output_string # Return None for error, and output string

def main_analysis(csv_file, min_support=0.05, min_confidence=0.7): # Keyword args with defaults
    """Main function to run sequence mining analysis."""

    if not os.path.exists(csv_file):
        return f"Error: CSV file not found: {csv_file}" # Error string

    error_seq_mining, analysis_output = analyze_sequence_mining(csv_file, min_support, min_confidence) # Get potential error and output
    if error_seq_mining: # If analyze_sequence_mining returned an error string
        return error_seq_mining # Return error string to GUI

    return analysis_output # Return analysis output string


if __name__ == "__main__":
    # Example for direct testing:
    csv_file_path_test = "path/to/your/csv_output/your_video_name_analysis.csv" # Replace
    min_support_test = 0.06
    min_confidence_test = 0.8

    class Args: # Dummy Args class for testing (not needed for GUI)
        def __init__(self, csv_file, min_support, min_confidence):
            self.csv_file = csv_file
            self.min_support = min_support
            self.min_confidence = min_confidence

    test_args = Args(csv_file_path_test, min_support_test, min_confidence_test)

    # Simulate command-line execution (original main) - not needed for GUI
    # main(test_args)

    # Direct call to main_analysis for testing:
    output_message = main_analysis(csv_file=test_args.csv_file, 
                                  min_support=test_args.min_support,
                                  min_confidence=test_args.min_confidence)
    print(output_message) # Print output for direct test