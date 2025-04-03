import pandas as pd
from collections import Counter
import argparse
import matplotlib.pyplot as plt  # Import matplotlib
import io #Import io for capturing print output
import os #Import os for path manipulation

def analyze_ngrams(csv_file_path, n=2, top_n=10):  #Added top_n argument
    """
    Performs N-gram analysis and generates a bar chart of the top N-grams.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        return f"Error: CSV file not found: {csv_file_path}", None # Return error string
    except Exception as e:
        return f"Error reading CSV file: {e}", None # Return error string

    # Extract behavior labels
    behaviors = df['Class Label'].tolist()

    # Generate N-grams
    ngrams = zip(*[behaviors[i:] for i in range(n)])
    ngram_list = [" ".join(ngram) for ngram in ngrams]

    # Count N-gram frequencies
    ngram_counts = Counter(ngram_list)

    # Capture print output
    output_buffer = io.StringIO()

    output_buffer.write(f"\n--- Top {n}-grams ---\n")
    most_common_ngrams = ngram_counts.most_common(top_n) #Changed most_common to use top_n
    for ngram, count in most_common_ngrams:
        output_buffer.write(f"{ngram}: {count}\n")

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
    plot_path = os.path.join(os.getcwd(), "ngram_frequencies.png") # Save in current directory
    plt.savefig(plot_path)
    plt.close()
    message_plot_saved = f"N-gram frequency chart saved as {plot_path}"
    print(message_plot_saved) # Still print for console output if running directly

    output_string = output_buffer.getvalue() + message_plot_saved # Combine print output and plot message
    output_buffer.close()
    return None, output_string # Return None for error, and output string


def main_analysis(csv_file, n=2, top_n=10): # Keyword args with defaults
    """Main function to run N-gram analysis."""

    if not os.path.exists(csv_file):
        return f"Error: CSV file not found: {csv_file}" # Error string

    error_ngram, analysis_output = analyze_ngrams(csv_file, n, top_n) # Get potential error and output
    if error_ngram: # If analyze_ngrams returned an error string
        return error_ngram # Return error string to GUI

    return analysis_output # Return analysis output string

if __name__ == "__main__":
    # Example for direct testing:
    csv_file_path_test = "path/to/your/csv_output/your_video_name_analysis.csv" # Replace with real path
    n_test = 3
    top_n_test = 15

    class Args: # Dummy Args class for testing (not needed for GUI)
        def __init__(self, csv_file, n, top_n):
            self.csv_file = csv_file
            self.n = n
            self.top_n = top_n

    test_args = Args(csv_file_path_test, n_test, top_n_test)

    # Simulate command-line execution (original main) - not needed for GUI
    # main(test_args)

    # Direct call to main_analysis for testing:
    output_message = main_analysis(csv_file=test_args.csv_file,
                                  n=test_args.n,
                                  top_n=test_args.top_n)
    print(output_message) # Print output for direct test