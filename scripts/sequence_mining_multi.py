import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import argparse
import os
import networkx as nx
import matplotlib.pyplot as plt

def analyze_sequence_mining_multi(input_folder, output_folder, min_support=0.05, min_confidence=0.7):
    """
    Performs sequence mining on multiple CSV files and combines/summarizes the results.

    Args:
        input_folder (str): Path to the folder containing the CSV files.
        output_folder (str): Path to the folder to save the combined results.
        min_support (float): Minimum support threshold.
        min_confidence (float): Minimum confidence threshold.
    """

    all_rules = []

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
                print(f"Error reading CSV file: {e}")
                continue

            # Convert behavior data into transaction data
            transactions = df['Class Label'].apply(lambda x: [x]).groupby(df['Bout ID']).sum().tolist()

            # Encode the transaction data using TransactionEncoder
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

            # Find frequent itemsets using the Apriori algorithm
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

            # Generate association rules
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            rules['video_name'] = video_name  # Add video name to the rules
            all_rules.append(rules)

            print(f"\n--- Association Rules - {video_name} ---")
            print(rules)

    # Combine association rules from all videos
    combined_rules = pd.concat(all_rules, ignore_index=True)

    print("\n--- Combined Association Rules ---")
    print(combined_rules)

    # Create a directed graph from the combined association rules
    G = nx.DiGraph()
    for index, row in combined_rules.iterrows():
        antecedents = ", ".join(row['antecedents'])
        consequents = ", ".join(row['consequents'])
        G.add_edge(antecedents, consequents, confidence=row['confidence'])

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold")

    edge_labels = nx.get_edge_attributes(G, 'confidence')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Combined Association Rules Graph")
    plt.tight_layout()
    output_chart_path = os.path.join(output_folder, "combined_association_rules_graph.png")
    plt.savefig(output_chart_path)
    plt.show() #Show graph too.

    print(f"Combined association rules graph saved as: {output_chart_path}")

    # Save the combined association rules to a CSV file
    output_csv_path = os.path.join(output_folder, "combined_association_rules.csv")
    combined_rules.to_csv(output_csv_path, index=False)
    print(f"Combined association rules saved to: {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Perform sequence mining on multiple CSV files and combine the results.")
    parser.add_argument("--input_folder", required=True, help="Path to the folder containing the CSV files.")
    parser.add_argument("--output_folder", required=True, help="Path to the folder to save the combined results.")
    parser.add_argument("--min_support", type=float, default=0.05, help="Minimum support threshold (default: 0.05).")
    parser.add_argument("--min_confidence", type=float, default=0.7, help="Minimum confidence threshold (default: 0.7).")

    args = parser.parse_args()

    analyze_sequence_mining_multi(args.input_folder, args.output_folder, args.min_support, args.min_confidence)

if __name__ == "__main__":
    main()