import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import argparse
import networkx as nx  # Requires: pip install networkx
import matplotlib.pyplot as plt

def analyze_sequence_mining(csv_file_path, min_support=0.05, min_confidence=0.7):
    """
    Performs sequence mining and generates a directed graph of association rules.

    Args:
        csv_file_path (str): Path to the CSV file.
        min_support (float): Minimum support threshold.
        min_confidence (float): Minimum confidence threshold.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

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

    print("\n--- Association Rules ---")
    print(rules)

    # Create a directed graph from the association rules
    G = nx.DiGraph()
    for index, row in rules.iterrows():
         if row['confidence'] >= 0.9:  # Adjust threshold
            antecedents = ", ".join(row['antecedents'])
            consequents = ", ".join(row['consequents'])
            G.add_edge(antecedents, consequents, confidence=row['confidence'])

    pos = nx.spring_layout(G)  # Layout algorithm for node positioning
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold")

    edge_labels = nx.get_edge_attributes(G, 'confidence')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels) #Add edge labels.

    plt.title("Association Rules Graph")
    plt.tight_layout()
    plt.savefig("association_rules_graph.png") #Save graph
    plt.show()
    print("Association rules graph saved as association_rules_graph.png")

def main():
    parser = argparse.ArgumentParser(description="Perform sequence mining and generate an association rules graph.")
    parser.add_argument("--csv_file", required=True, help="Path to the CSV file containing the behavior data.")
    parser.add_argument("--min_support", type=float, default=0.05, help="Minimum support threshold (default: 0.05).")
    parser.add_argument("--min_confidence", type=float, default=0.7, help="Minimum confidence threshold (default: 0.7).")

    args = parser.parse_args()

    analyze_sequence_mining(args.csv_file, min_support=args.min_support, min_confidence=args.min_confidence)

if __name__ == "__main__":
    main()