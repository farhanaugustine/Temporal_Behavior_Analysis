import pandas as pd
from collections import Counter
import argparse
import os
import matplotlib.pyplot as plt
import logging

def analyze_ngrams_multi(input_folder, output_folder, n=2, top_n=10):
    """
    Performs N-gram analysis on multiple CSV files and combines/summarizes the results.

    Args:
        input_folder (str): Path to the folder containing the CSV files.
        output_folder (str): Path to the folder to save the combined results.
        n (int): The length of the N-grams.
        top_n (int): Number of top N-grams to display in the chart.
    """

    logging.info(f"Starting N-gram analysis on folder: {input_folder}")

    input_folder = os.path.abspath(input_folder) #Absolute paths for clarity
    output_folder = os.path.abspath(output_folder)

    #Check for empty input folder
    if not os.listdir(input_folder):
        logging.warning(f"Input folder is empty: {input_folder}")
        print(f"Warning: Input folder is empty: {input_folder}")
        return

    csv_files = [filename for filename in os.listdir(input_folder) if filename.endswith(".csv")]

    if not csv_files: #Check for existance of at least 1 csv file
         logging.warning(f"No CSV files found in input folder: {input_folder}")
         print(f"Warning: No CSV files found in input folder: {input_folder}")
         return

    all_ngram_counts = {}

    for filename in csv_files:
        video_name = os.path.splitext(filename)[0]
        csv_file_path = os.path.join(input_folder, filename)

        logging.info(f"Analyzing video: {video_name}")
        print(f"\nAnalyzing video: {video_name}")

        try:
            df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            logging.error(f"CSV file not found: {csv_file_path}")
            print(f"Error: CSV file not found: {csv_file_path}")
            continue  # Skip to the next file
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            print(f"Error reading CSV file: {e}")
            continue

        # Extract behavior labels
        behaviors = df['Class Label'].tolist()

        # Generate N-grams
        ngrams = zip(*[behaviors[i:] for i in range(n)])
        ngram_list = [" ".join(ngram) for ngram in ngrams]

        # Count N-gram frequencies
        ngram_counts = Counter(ngram_list)
        all_ngram_counts[video_name] = ngram_counts  # Store the N-gram counts

        print(f"\n--- Top {n}-grams - {video_name} ---")
        most_common_ngrams = ngram_counts.most_common(top_n)
        for ngram, count in most_common_ngrams:
            print(f"{ngram}: {count}")

    # Combine N-gram counts from all videos
    combined_ngram_counts = Counter()
    for ngram_counts in all_ngram_counts.values():
        combined_ngram_counts.update(ngram_counts)

    # Get the top N-grams across all videos
    most_common_combined_ngrams = combined_ngram_counts.most_common(top_n)

    print(f"\n--- Top {n}-grams (Combined) ---")
    for ngram, count in most_common_combined_ngrams:
        print(f"{ngram}: {count}")

    # Create a bar chart of combined N-gram frequencies
    ngrams, counts = zip(*most_common_combined_ngrams)
    plt.figure(figsize=(12, 6))
    plt.bar(ngrams, counts)
    plt.xlabel("N-gram")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} Combined {n}-gram Animal Behavior Sequences", fontsize=16)  # Improved Title
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save the plot
    output_chart_path = os.path.join(output_folder, "combined_ngram_frequencies.png")
    plt.savefig(output_chart_path)
    plt.show() #Show graph too.

    print(f"Combined N-gram frequency chart saved as: {output_chart_path}")

    # Save the full combined N-gram counts to a CSV file
    output_csv_path = os.path.join(output_folder, "combined_ngram_counts.csv")
    df = pd.DataFrame.from_dict(combined_ngram_counts, orient='index', columns=['Count'])
    df.index.name = 'N-gram'
    df.sort_values(by='Count', ascending=False, inplace=True)
    df.to_csv(output_csv_path)
    print(f"Full combined N-gram counts saved to: {output_csv_path}")
    logging.info(f"N-gram analysis complete, results saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description="Perform N-gram analysis on multiple CSV files and combine the results.")
    parser.add_argument("--input_folder", required=True, help="Path to the folder containing the CSV files.")
    parser.add_argument("--output_folder", required=True, help="Path to the folder to save the combined results.")
    parser.add_argument("--n", type=int, default=2, help="The length of the N-grams (default: 2).")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top N-grams to display (default: 10).")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') #Added for debugging, show log file

    analyze_ngrams_multi(args.input_folder, args.output_folder, args.n, args.top_n)

if __name__ == "__main__":
    main()