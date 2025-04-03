import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import argparse
import os
import networkx as nx
import matplotlib.pyplot as plt
import io # Import io for capturing print output

def analyze_sequence_mining_multi(input_folder, output_folder, min_support=0.5, min_confidence=0.8, top_n=1000):
    """
    Performs sequence mining on multiple CSV files and combines/summarizes results.
    """

    all_rules = []
    output_messages = [] # List to collect messages

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            video_name = os.path.splitext(filename)[0]
            csv_file_path = os.path.join(input_folder, filename)

            message_analyzing_video = f"\nAnalyzing video: {video_name}"
            print(message_analyzing_video)
            output_messages.append(message_analyzing_video)

            try:
                df = pd.read_csv(csv_file_path)
            except FileNotFoundError:
                message_file_not_found = f"Error: CSV file not found: {csv_file_path}"
                print(message_file_not_found)
                output_messages.append(message_file_not_found)
                continue  # Skip to next file
            except Exception as e:
                message_csv_read_error = f"Error reading CSV file: {e}"
                print(message_csv_read_error)
                output_messages.append(message_csv_read_error)
                continue

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
            rules['video_name'] = video_name  # Add video name to rules
            all_rules.append(rules)

            # Capture and log rules for each video
            output_buffer = io.StringIO()
            output_buffer.write(f"\n--- Association Rules - {video_name} ---\n")
            output_buffer.write(rules.to_string())
            video_rules_output = output_buffer.getvalue()
            output_buffer.close()
            print(video_rules_output)
            output_messages.append(video_rules_output)


    # Combine association rules from all videos
    combined_rules = pd.concat(all_rules, ignore_index=True)

    # Capture and log combined rules
    output_buffer_combined_rules = io.StringIO()
    output_buffer_combined_rules.write("\n--- Combined Association Rules ---\n")
    output_buffer_combined_rules.write(combined_rules.to_string())
    combined_rules_output = output_buffer_combined_rules.getvalue()
    output_buffer_combined_rules.close()
    print(combined_rules_output)
    output_messages.append(combined_rules_output)


    # Select top N rules for visualization
    top_rules = combined_rules.nlargest(top_n, 'confidence')

    # Create directed graph from top association rules
    G = nx.DiGraph()
    for index, row in top_rules.iterrows():
        antecedents = ", ".join(row['antecedents'])
        consequents = ", ".join(row['consequents'])
        G.add_edge(antecedents, consequents, confidence=row['confidence'])

    # Improve graph visualization
    pos = nx.spring_layout(G, k=0.5, iterations=50)  
    plt.figure(figsize=(20, 12))  # Increased figure size

    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue",
            font_size=12, font_weight="bold", alpha=0.7) 

    edge_labels = nx.get_edge_attributes(G, 'confidence')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10) 

    plt.title(f"Top {top_n} Behavioral Association Rules (Confidence >= {min_confidence}) Visualized as a Network Graph Across Multiple Videos", fontsize=16) # f-string
    plt.tight_layout()
    output_chart_path = os.path.join(output_folder, "combined_association_rules_graph.png")
    plt.savefig(output_chart_path)
    plt.close()
    message_graph_saved = f"Combined association rules graph saved as: {output_chart_path}"
    print(message_graph_saved)
    output_messages.append(message_graph_saved)

    # Save combined association rules to CSV
    output_csv_path = os.path.join(output_folder, "combined_association_rules.csv")
    combined_rules.to_csv(output_csv_path, index=False)
    message_csv_saved = f"Combined association rules saved to: {output_csv_path}"
    print(message_csv_saved)
    output_messages.append(message_csv_saved)

    message_analysis_complete = f"Combined sequence mining analysis complete. Results saved to: {output_folder}"
    print(message_analysis_complete)
    output_messages.append(message_analysis_complete)

    return None, output_messages # Return None for error, and output messages


def main_analysis(input_folder, output_folder, min_support=0.5, min_confidence=0.8, top_n=1000): # Keyword args with defaults
    """Main function to run multi-video sequence mining analysis."""

    if not os.path.isdir(input_folder):
        return f"Error: Input folder not found: {input_folder}" # Return error string
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True) # Create if doesn't exist

    error_seq_mining_multi, analysis_output = analyze_sequence_mining_multi(input_folder, output_folder, min_support, min_confidence, top_n) # Get potential error and output
    if error_seq_mining_multi: # If analyze_sequence_mining_multi returned error string
        return error_seq_mining_multi # Return error string to GUI

    return "\n".join(analysis_output) # Return analysis output string


if __name__ == "__main__":
    # Example for direct testing:
    input_folder_path = "path/to/your/input_folder" # Replace
    output_folder_path = "path/to/your/output_folder" # Replace
    min_support_test = 0.06
    min_confidence_test = 0.8
    top_n_test = 12

    class Args: # Dummy Args class for testing (not needed for GUI)
        def __init__(self, input_folder, output_folder, min_support, min_confidence, top_n):
            self.input_folder = input_folder
            self.output_folder = output_folder
            self.min_support = min_support
            self.min_confidence = min_confidence
            self.top_n = top_n

    test_args = Args(input_folder_path, output_folder_path, min_support_test, min_confidence_test, top_n_test)

    # Simulate command-line execution (original main) - not needed for GUI
    # main(test_args)

    # Direct call to main_analysis for testing:
    output_message = main_analysis(input_folder=test_args.input_folder, 
                                  output_folder=test_args.output_folder,
                                  min_support=test_args.min_support,
                                  min_confidence=test_args.min_confidence,
                                  top_n=test_args.top_n)
    print(output_message) # Print output for direct test