import pandas as pd
from collections import Counter
import os
import matplotlib.pyplot as plt
import io #Import io for capturing print output

def analyze_ngrams_multi(input_folder, output_folder, n=2, top_n=10):
    """
    Performs N-gram analysis on multiple CSV files and combines/summarizes results.
    """
    input_folder = os.path.abspath(input_folder) #Absolute paths for clarity
    output_folder = os.path.abspath(output_folder)

    output_messages = [] # List to collect messages

    #Check for empty input folder
    if not os.listdir(input_folder):
        message_empty_input = f"Warning: Input folder is empty: {input_folder}"
        print(message_empty_input)
        output_messages.append(message_empty_input)
        return None, output_messages # Return None and messages

    csv_files = [filename for filename in os.listdir(input_folder) if filename.endswith(".csv")]

    if not csv_files: #Check for existance of at least 1 csv file
         message_no_csv = f"Warning: No CSV files found in input folder: {input_folder}"
         print(message_no_csv)
         output_messages.append(message_no_csv)
         return None, output_messages # Return None and messages

    all_ngram_counts = {}

    for filename in csv_files:
        video_name = os.path.splitext(filename)[0]
        csv_file_path = os.path.join(input_folder, filename)

        message_analyzing_video = f"\nAnalyzing video: {video_name}"
        print(message_analyzing_video)
        output_messages.append(message_analyzing_video)

        try:
            df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            message_file_not_found = f"CSV file not found: {csv_file_path}"
            print(message_file_not_found)
            output_messages.append(message_file_not_found)
            continue  # Skip to the next file
        except Exception as e:
            message_csv_read_error = f"Error reading CSV file: {e}"
            print(message_csv_read_error)
            output_messages.append(message_csv_read_error)
            continue

        # Extract behavior labels
        behaviors = df['Class Label'].tolist()

        # Generate N-grams
        ngrams = zip(*[behaviors[i:] for i in range(n)])
        ngram_list = [" ".join(ngram) for ngram in ngrams]

        # Count N-gram frequencies
        ngram_counts = Counter(ngram_list)
        all_ngram_counts[video_name] = ngram_counts  # Store the N-gram counts

        # Capture print output for each video
        output_buffer = io.StringIO()
        output_buffer.write(f"\n--- Top {n}-grams - {video_name} ---\n")
        most_common_ngrams = ngram_counts.most_common(top_n)
        for ngram, count in most_common_ngrams:
            output_buffer.write(f"{ngram}: {count}\n")
        video_ngram_output = output_buffer.getvalue()
        output_buffer.close()
        print(video_ngram_output)
        output_messages.append(video_ngram_output)


    # Combine N-gram counts from all videos
    combined_ngram_counts = Counter()
    for ngram_counts in all_ngram_counts.values():
        combined_ngram_counts.update(ngram_counts)

    # Get top N-grams combined
    most_common_combined_ngrams = combined_ngram_counts.most_common(top_n)

    # Capture combined n-gram output
    output_buffer_combined = io.StringIO()
    output_buffer_combined.write(f"\n--- Top {n}-grams (Combined) ---\n")
    for ngram, count in most_common_combined_ngrams:
        output_buffer_combined.write(f"{ngram}: {count}\n")
    combined_ngram_output = output_buffer_combined.getvalue()
    output_buffer_combined.close()
    print(combined_ngram_output)
    output_messages.append(combined_ngram_output)


    # Create bar chart of combined N-gram frequencies
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
    plt.close()
    message_chart_saved = f"Combined N-gram frequency chart saved as: {output_chart_path}"
    print(message_chart_saved)
    output_messages.append(message_chart_saved)

    # Save the full combined N-gram counts to a CSV file
    output_csv_path = os.path.join(output_folder, "combined_ngram_counts.csv")
    df = pd.DataFrame.from_dict(combined_ngram_counts, orient='index', columns=['Count'])
    df.index.name = 'N-gram'
    df.sort_values(by='Count', ascending=False, inplace=True)
    df.to_csv(output_csv_path)
    message_csv_saved = f"Full combined N-gram counts saved to: {output_csv_path}"
    print(message_csv_saved)
    output_messages.append(message_csv_saved)

    message_analysis_complete = f"N-gram analysis complete, results saved to: {output_folder}"
    print(message_analysis_complete)
    output_messages.append(message_analysis_complete)

    return None, output_messages # Return None for error, and output messages


def main_analysis(input_folder, output_folder, n=2, top_n=10): # Keyword args with defaults
    """Main function to run multi-video N-gram analysis."""

    if not os.path.isdir(input_folder):
        return f"Error: Input folder not found: {input_folder}" # Error string
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True) # Create if doesn't exist

    error_ngram_multi, analysis_output = analyze_ngrams_multi(input_folder, output_folder, n, top_n) # Get potential error and output
    if error_ngram_multi: #If analyze_ngrams_multi returned an error string
        return error_ngram_multi # Return error string to GUI

    return "\n".join(analysis_output) # Return analysis output string


if __name__ == "__main__":
    # Example for direct testing:
    input_folder_path = "path/to/your/input_folder" # Replace
    output_folder_path = "path/to/your/output_folder" # Replace
    n_test = 3
    top_n_test = 15

    class Args: # Dummy Args class for testing (not needed for GUI)
        def __init__(self, input_folder, output_folder, n, top_n):
            self.input_folder = input_folder
            self.output_folder = output_folder
            self.n = n
            self.top_n = top_n

    test_args = Args(input_folder_path, output_folder_path, n_test, top_n_test)

    # Simulate command-line execution (original main) - not needed for GUI
    # main(test_args)

    # Direct call to main_analysis for testing:
    output_message = main_analysis(input_folder=test_args.input_folder, 
                                  output_folder=test_args.output_folder,
                                  n=test_args.n,
                                  top_n=test_args.top_n)
    print(output_message) # Print output for direct test