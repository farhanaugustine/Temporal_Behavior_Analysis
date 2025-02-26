import pandas as pd
from collections import Counter
import argparse
import matplotlib.pyplot as plt  # Import matplotlib

def analyze_ngrams(csv_file_path, n=2, top_n=10):  #Added top_n argument
    """
    Performs N-gram analysis and generates a bar chart of the top N-grams.

    Args:
        csv_file_path (str): Path to the CSV file.
        n (int): The length of the N-grams.
        top_n (int): Number of top N-grams to display in the chart.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Extract behavior labels
    behaviors = df['Class Label'].tolist()

    # Generate N-grams
    ngrams = zip(*[behaviors[i:] for i in range(n)])
    ngram_list = [" ".join(ngram) for ngram in ngrams]

    # Count N-gram frequencies
    ngram_counts = Counter(ngram_list)

    print(f"\n--- Top {n}-grams ---")
    most_common_ngrams = ngram_counts.most_common(top_n) #Changed most_common to use top_n
    for ngram, count in most_common_ngrams:
        print(f"{ngram}: {count}")

    # Create a bar chart
    ngrams, counts = zip(*most_common_ngrams)
    plt.figure(figsize=(12, 6))  # Adjust figure size as needed
    plt.bar(ngrams, counts)
    plt.xlabel("N-gram")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} {n}-grams")
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
    plt.tight_layout()

    # Save the plot
    plt.savefig("ngram_frequencies.png")  # Save the plot to a file
    plt.show() #Show graph too.
    print("N-gram frequency chart saved as ngram_frequencies.png")

def main():
    parser = argparse.ArgumentParser(description="Perform N-gram analysis and generate a frequency chart.")
    parser.add_argument("--csv_file", required=True, help="Path to the CSV file containing the behavior data.")
    parser.add_argument("--n", type=int, default=2, help="The length of the N-grams (default: 2).")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top N-grams to display (default: 10).") #Added argument

    args = parser.parse_args()

    analyze_ngrams(args.csv_file, n=args.n, top_n=args.top_n)

if __name__ == "__main__":
    main()