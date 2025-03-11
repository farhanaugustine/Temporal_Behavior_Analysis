# Behavioral Analysis Toolkit

This project provides a user-friendly graphical interface (GUI) for analyzing behavioral data extracted from video using YOLOv8 or YOLOv11 object detection. It allows you to process raw YOLOv8 output, define "bouts" of behavior, and perform various analyses, including transition probabilities, cross-correlations, FFT analysis, rhythm detection, basic correlations, and total time spent analysis.

* **Author:** Farhan Augustine
* **Affiliation:** University of Maryland Baltimore County (UMBC)

# Getting Started (GUI)
<img width="725" alt="image" src="https://github.com/user-attachments/assets/54f7596b-2b20-4c93-8380-eb9e5bc2dc19" />



1.  **Installation:**
    * **Option 1: Using Anaconda to create a virtual enviroment (Recommended for most users):**
    	*   Using the provided `environment.yml` file create a virtual enviroment within Anaconda.
	```
 	  git clone https://github.com/farhanaugustine/Temporal_Behavior_Analysis.git
   	  cd Temporal_Behavior_Analysis
 	  conda env create -f environment.yml
 	  conda activate YOLO
 	  python GUI_v2.py
	```      
    *   **Option 2: Using the Executable**:
        *   If you received a pre-built executable (e.g., `BehavioralAnalysisTool.exe` on Windows, or an application bundle on macOS):
            1.  Simply double-click the executable to launch the application.  No further installation is required.
            2. If you're on macOS, you may need to go to System Preferences->Security & Privacy->General. In some cases, you must manually allow applications to run in this section of the settings.
        *	Note: Some features of macOS and Windows restrict unknown applications. It is important to check the security settings and allowlist this application.
	*  **Option 3: Running from Source** (For developers or users comfortable with Python):
        *   Install an anaconda environment with dependencies (see list of dependencies below)
        *   Make sure you have Python 3.10 or later installed.
        *   Install the required libraries:
            ```bash
            pip install tkinter pandas numpy matplotlib seaborn scipy statsmodels openpyxl scikit-learn
            ```
        *   Download the project files (including `GUI.py` and the `scripts` folder).
        *   Open a terminal or command prompt, navigate to the directory containing `GUI.py`, and run:
            ```bash
            python GUI_v2.py
            ```

2.  **Launch the Application:** Run the `GUI.py` script (or the standalone executable, if you created one).

3.  **Prepare Your Data (Optional but Recommended):**
    *   If you have YOLOv8 output files (`.txt`) from *multiple* videos in the *same* folder, click the **"Organize TXT Files"** button. This will create subfolders for each video, making your analysis easier.  Do this *before* selecting the output folder in the next step.

4.  **Input:**
    *   **YOLO Output Folder:** Click the "Browse" button next to "YOLO Output Folder" and select the folder containing the YOLOv8 `.txt` output files *for a single video*.  If you organized your files in step 3, select the subfolder for the video you want to analyze.
    *   **Class Labels:** Click the "Edit" button next to "Class Labels."  A dialog box will appear.  Enter a Python dictionary that maps the *numeric* class IDs used by YOLOv8 to the *names* of your behaviors.  For example:
        ```python
        {0: "Exploration", 1: "Grooming", 2: "Rearing", 3: "Jump"}
        ```
        Make sure the keys (0, 1, 2, 3) match the class IDs in your YOLOv8 output files.
    *   **Video Name:** Type the base name of your video file (e.g., `myvideo`). This is used for naming output files.
    *   **Frame Rate:** Enter the frame rate of your video (e.g., 30 frames per second).
    *   **Min Bout Duration (frames):**  *Important!* This setting helps filter out short, likely spurious detections.  Enter the *minimum* number of *consecutive* frames a behavior must be detected to be considered a real "bout" of that behavior.  For example, if you set this to 3, and a behavior is only detected for 1 or 2 frames in a row, those detections will be ignored.  A good starting value is often 3-5 frames, but you may need to adjust this based on your video's frame rate and the nature of the behaviors.
    *   **Max Gap Duration (frames):**  *Important!* This setting allows for short *interruptions* within a bout.  For example, if your animal is grooming, and the detector briefly misclassifies it as "exploring" for a frame or two, and *then* it's detected as grooming again, you probably want to consider that a *single* bout of grooming.  This setting controls how long that interruption can be.  A good starting value is often 3-5 frames.

    **Example (Bout Filtering):** Let's say you have these raw YOLO detections (G=Grooming, E=Exploration, X=Other):

    ```
    Frame:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
    Label:  G  G  G  E  G  G  G  G  X  X  G  G  G  G  G
    ```

    *   If `Min Bout Duration = 3` and `Max Gap Duration = 2`, the script would identify *one* long bout of grooming, spanning all 15 frames. The single 'E' frame and the two 'X' frames are ignored because they are shorter than the minimum bout duration *and* they fall within the allowed gap duration between grooming detections.
    *    If 'Min Bout Duration = 5' and 'Max Gap Duration = 1', the script would identify one short bout of grooming from frames 5 to 8.
    *   If `Min Bout Duration = 3` and `Max Gap Duration = 0`, the script would identify *two* bouts of grooming: frames 1-3 and frames 5-8, and another one for frames 11-15. The 'E' and 'X' detections would completely separate the bouts.

    *You'll likely need to experiment with these values to find what works best for your specific videos and behaviors.*

5.  **Run General Analysis:** Click the red button labeled **"general_analysis (RUN THIS FIRST)"**.  This is *essential* and must be done before running any other analyses.  This step processes the raw YOLO output and creates a CSV file that the other analyses use.  Progress and any errors will be shown in the "Log" area at the bottom of the window.

6.  **Run Other Analyses:** Once the general analysis is complete, you can click any of the other buttons in the "Analysis Scripts" section to perform specific analyses for single-video analysis only:
    *   **transition_probabilities:** Calculates how often one behavior follows another.
    *   **time_lagged_cross_correlation:** Looks for relationships between behaviors at different time lags.
    *   **fft_analysis:**  Identifies rhythmic patterns in the behaviors.
    *   **behavioral_rhythms:** Detects peaks in the occurrence of behaviors.
    *   **basic_correlations:** Calculates simple correlations between the occurrences of behaviors.
    *   **total_time_spent:** Calculates the total time spent on each behavior.
     *  **granger_causality:** Performs Granger Causality Test.

7.  **View Results:** The output files (Excel files, plots, etc.) will be saved within the main "YOLO Output Folder" you selected.  The log window will also show where the files are saved.

# Advanced Usage

**Running from the Command Line:**

You can run the individual analysis scripts directly from the command line. This is useful for testing, scripting, or if you prefer not to use the GUI.  Each script uses `argparse` to handle command-line arguments. To see the available options for each script, run it with the `--help` flag.  For example:

```bash
python scripts/general_analysis.py --help
python scripts/transition_probabilities.py --help
```
## Example (general_analysis.py) [**From command line**]:
```bash
python scripts/general_analysis.py --output_folder /path/to/yolo/output --class_labels "{0: 'Exploration', 1: 'Grooming'}" --frame_rate 30 --video_name myvideo --min_bout_duration 3 --max_gap_duration 5
```

# Script Descriptions (Brief):
* general_analysis.py: Processes raw YOLO output, defines bouts, creates the main CSV file. Run this first.

* transition_probabilities.py: Calculates and visualizes transition probabilities for EACH video and make sure to move and store them in the transition folder.

* time_lagged_cross_correlation.py: Calculates time-lagged cross-correlations for each video.

* fft_analysis.py: Performs FFT analysis to find dominant frequencies for each video.

* behavioral_rhythms.py: Detects behavioral rhythms in a single video.

* basic_correlations.py: Calculates Pearson correlations.

* total_time_spent.py: Calculates total time spent on each behavior.

* granger_causality.py: Performs Granger Causality Test.

* HMMs.py: Runs the HMMs.py script for sequence analysis.

* n_gram_analysis.py: Runs the n_gram_analysis.py script for sequence analysis.

* sequence_mining.py: Runs the sequence_mining.py script for sequence analysis.

* HMMs_multi.py: Runs the HMMs_multi.py script for multi-video sequence analysis. [**Requires a folder named "HMMs_multi" with all `general_analysis.py` output CSV files be placed in this folder**]

* n_gram_analysis_multi.py: Runs the n_gram_analysis_multi.py script for multi-video sequence analysis. [**Requires a folder named "n-gram_analysis" with all `general_analysis.py` output CSV files be placed in this folder**]

* sequence_mining_multi.py: Runs the sequence_mining_multi.py script for multi-video sequence analysis. [**Requires a folder named "sequence_mining" with all `general_analysis.py` output CSV files be placed in this folder**]

## Scripts and their Biological Significance

# Behavioral Analysis Suite

This suite of Python scripts provides a comprehensive framework for analyzing animal behavior from video recordings. The core workflow involves processing raw object detection data (e.g., from YOLO), defining behavioral bouts, and then applying a variety of statistical and machine learning techniques to extract meaningful insights into behavioral patterns and dynamics.

## Workflow Overview

1.  **Object Detection & Bout Definition:** Use an object detection model (like YOLO) to track the animal and relevant objects in the video. Then, use `general_analysis.py` to process the raw output, define behavioral bouts (e.g., grooming, locomotion), and create a structured CSV dataset.  This is the most critical step, as the accuracy of downstream analyses depends on the quality of the object detection and the ethological relevance of the defined behaviors.
2.  **Exploratory Analysis:**  Use scripts like `total_time_spent.py` and `basic_correlations.py` to get a high-level overview of the behavioral data.
3.  **Advanced Analysis:** Explore more sophisticated techniques, such as transition probabilities, time-lagged correlations, frequency analysis (FFT), and sequence mining to uncover complex behavioral patterns and relationships.
4.  **Group Comparisons:** Utilize the "multi" versions of sequence analysis and other scripts to compare behavioral measures and sequences across different individuals or experimental conditions.

## Scripts and their Biological Significance

| Script Name | Function | Biological Significance |
|---|---|---|
| `general_analysis.py` | Processes YOLO output, defines behavioral bouts (e.g., grooming, locomotion), creates a structured CSV dataset. | Provides the *sine qua non* for quantitative behavioral analysis. Accurate bout definitions are paramount: If studying social behavior, accurately distinguishing 'approach' from 'avoid' is crucial; for anxiety, distinguishing 'center exploration' from 'thigmotaxis' is key. This script ensures ethologically valid, quantifiable input for downstream analysis. |
| `transition_probability_single.py` | Calculates transition probabilities between behavioral states for a *single* video. | Unveils individual behavioral strategies. For instance, a high probability of transitioning from 'exploration' to 'foraging' in a novel environment suggests adaptive resource seeking. In contrast, a high probability of 'exploration' to 'immobility' might indicate maladaptive stress responses. Drug effects can be assessed by shifts in transition probabilities. |
| `transition_probability_multi.py` | Calculates transition probabilities between behavioral states across *multiple* videos. | Enables group-level comparisons of behavioral organization. For example, comparing transition matrices in control vs. knockout mice can reveal disruptions in typical behavioral sequencing caused by gene deletion. Alternatively, comparing transition probabilities across developmental stages highlights shifts in behavioral priorities. |
| `time_lagged_cross_correlation_single.py` | Calculates cross-correlations between behavioral time series at varying time lags for a *single* video. | Reveals temporally coupled behaviors within an individual. A strong positive correlation between 'approach' and 'following' at a short time lag in a social interaction paradigm indicates rapid responsiveness. A delayed negative correlation between 'exploration' and 'immobility' might suggest that periods of exploration lead to subsequent fatigue or anxiety. |
| `cross_correlation_combine.py` | Combines cross-correlation results from multiple videos. | Allows for robust statistical assessment of inter-behavioral dependencies across a population. Provides a means to identify consistently predictive or inversely related behaviors at a group level, informing our understanding of behavioral syndromes and shared neural substrates. |
| `cross_correlation_single.py` | Calculates cross-correlations between behavioral time series at varying time lags for a *single* video | Identifies predictive relationships between behaviors, including temporal delays. Useful for understanding complex, multi-stage patterns. Duplicate of time_lagged_cross_correlation_single.py? Review and potentially remove/merge. |
| `cross_correlation_stats.py` | Performs statistical analysis on the results of cross-correlation calculations. | Quantifies the significance of correlated behavioral patterns, reducing the risk of spurious interpretations. Facilitates the identification of robust, repeatable inter-behavioral associations, critical for translational research. |
| `fft_analysis_single.py` | Performs FFT on behavioral time series data for a *single* video. | Reveals individual-level rhythmic patterns. For instance, identifying a ~4-hour periodicity in grooming behavior suggests an ultradian rhythm potentially linked to hormonal cycles or internal clock mechanisms. Abnormalities in these rhythms may indicate neurological dysfunction. |
| `fft_analysis_multi.py` | Performs FFT on behavioral time series data across *multiple* videos. | Enables group comparisons of behavioral rhythms. For example, comparing the dominant frequencies of locomotor activity in wild-type vs. circadian clock mutant mice elucidates the role of specific genes in regulating rhythmic behavior. This approach also facilitates the study of chronotype differences. |
| `behavioral_rhythms_single.py` |  Detects and characterizes behavioral rhythms within a *single* video recording. | Allows characterization of complex rhythms beyond simple periodicities, using algorithms tailored to specific rhythm shapes. Helps to understand how individual variations in such rhythms can underpin behavioral diversity. |
| `behavioral_rhythms_multi.py` | Detects and characterizes behavioral rhythms across *multiple* video recordings. | Enables statistical comparison of rhythmic parameters, like amplitude and phase, between experimental groups. This facilitates the study of how environmental manipulation, disease models, or genetic variations influence such rhythms. |
| `basic_correlation.py` | Calculates Pearson correlation coefficients between different behavioral variables. | Offers a rapid assessment of behavioral relationships. Observing a strong positive correlation between 'time spent exploring' and 'novel object preference' in a cognitive assay provides evidence that exploration is linked to learning and memory. A negative correlation between 'social interaction' and 'self-grooming' might indicate social avoidance. |
| `granger_causality.py` | Performs Granger causality tests to determine if one behavior predicts another. | Suggests potential directional influences between behaviors. If past 'aggression' strongly predicts future 'avoidance' in a conspecific, but not vice versa, it implies that aggression causally drives avoidance, not merely that they co-occur. However, causality must always be interpreted cautiously. |
| `HMMs.py` | Applies Hidden Markov Models (HMMs) to model behavioral sequences from a *single* video. | Uncovers hidden states that govern behavioral transitions. For example, in a foraging task, HMMs might reveal a 'searching' state followed by a 'handling' state, even if these states aren't directly observable. Shifts in state occupancy can reveal the impact of reward structure or pharmacological interventions. |
| `HMMs_multi.py` | Applies Hidden Markov Models (HMMs) to model behavioral sequences across *multiple* videos. | Enables group comparisons of hidden state dynamics. For example, comparing state transition probabilities in a fear conditioning paradigm can reveal differences in how control and lesioned animals learn and express fear. |
| `n_gram_analysis.py` | Analyzes the frequency of n-gram (n-behavior sequences) patterns in a *single* video. | Identifies frequently occurring behavioral motifs or "syllables." Observing a specific sequence of 'approach-sniff-withdraw' in a social encounter could represent a stereotyped investigative pattern, which might be altered in neurodevelopmental disorders. |
| `n_gram_analysis_multi.py` | Analyzes the frequency of n-gram patterns across *multiple* videos. | Facilitates the identification of conserved behavioral sequences across individuals or groups. For example, demonstrating that a particular foraging sequence is significantly more frequent in experienced animals indicates a learned behavioral strategy. |
| `sequence_mining.py` | Discovers frequent and significant behavioral sequences in a *single* video. | Identifies statistically over-represented behavioral patterns that may not be apparent through simple observation. Revealing a sequence of 'grooming-pause-scratch' as highly significant suggests an underlying, potentially functional, behavioral routine. |
| `sequence_mining_multi.py` | Allows for comparison of dominant sequence patterns between groups. |
| `total_time_spent.py` | Calculates the total duration (or frequency) of each behavior. | Provides a foundational metric for characterizing behavioral phenotypes. A significant increase in 'self-grooming' in a stress paradigm, or decreased 'social interaction' in a model of autism spectrum disorder, constitutes a readily interpretable and quantifiable behavioral change. |
| `transfer_entropy.py` | Calculates transfer entropy between behavioral time series. | Quantifies information flow and directional influence between behaviors. A high transfer entropy from 'exploration' to 'object investigation' suggests that exploratory behavior drives subsequent investigation. A lack of transfer entropy in the reverse direction might indicate that object investigation is not a strong driver of further exploration. |


# Dependencies:
Dependencies
This project requires Python 3.10+ and the following libraries:

* tkinter
* pandas
* numpy
* matplotlib
* seaborn
* scipy
* statsmodels
* openpyxl
* scikit-learn (sklearn)
* re
* shutil
* hmmlearn
* mlxtend
* networkx

Install them using:
```bash
pip install tkinter pandas numpy matplotlib seaborn scipy statsmodels openpyxl scikit-learn hmmlearn mlxtend networkx
```
