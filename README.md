# Atrial Fibrillation Detection

**This repository contains the original code for the research paper: "Detection of Atrial Fibrillation with a Hybrid Deep Learning Model and Time-Frequency Representations".**

This repository contains code for Atrial Fibrillation (AF) detection using time-frequency analysis and deep learning.

## Repository Structure

The repository is organized into the following folders and files:

```
DL/
  classified_by_proportion.py
  lack_pack.py
  three_categoriesz_in_proportion.py

TF/
  time-fre.py

data/
  # Example dataset (structure described below)

README.md
```

*   **`DL/` (Deep Learning):** This folder contains Python scripts for deep learning models used for AF detection.
*   **`TF/` (Time-Frequency):** This folder contains `time-fre.py`, a Python script for performing time-frequency analysis on ECG data.
*   **`data/`:** This folder is intended to store datasets. An example dataset structure is provided within this folder to illustrate the expected data format.
*   **`README.md`:** This file (you are currently reading it) provides an overview of the repository and instructions for usage.

## Python Scripts

### `TF/time-fre.py` (Time-Frequency Analysis)

This script performs time-frequency analysis on ECG data.  It is a prerequisite step before using the deep learning scripts in the `DL/` folder.  Run this script first to generate the time-frequency representations of your ECG signals, which will then be used as input for the deep learning models.

**Usage:**

```bash
python TF/time-fre.py
```

### `DL/classified_by_proportion.py` (Deep Learning - Binary Classification, Proportional Data)

This script is designed for **binary classification** (AF vs. Non-AF) when your dataset has a **relatively balanced proportion** between the Atrial Fibrillation (AF) class and the Non-AF (N) class.  It assumes your data is organized into two folders:

*   **`AF/`:** Contains time-frequency datasets for Atrial Fibrillation cases.
*   **`N/` (Non-AF):** Contains time-frequency datasets for Non-Atrial Fibrillation cases.

**Usage:**

```bash
python DL/classified_by_proportion.py
```

### `DL/lack_pack.py` (Deep Learning - Binary Classification, Imbalanced Data)

This script is designed for **binary classification** (AF vs. Non-AF) when your dataset is **imbalanced**, meaning there is a significant difference in the number of samples between the Atrial Fibrillation (AF) class and the Non-AF (N) class (e.g., significantly fewer AF samples than Non-AF).  It also assumes your data is organized into `AF/` and `N/` folders. This script likely incorporates techniques to handle data imbalance, such as oversampling, undersampling, or class weighting.

**Usage:**

```bash
python DL/lack_pack.py
```

### `DL/three_categoriesz_in_proportion.py` (Deep Learning - Three-Category Classification, Proportional Data)

This script is designed for **three-category classification** (AF, Normal, Other) when your dataset has a **relatively balanced proportion** across the three classes. It assumes your data is organized into three folders:

*   **`AF/`:** Contains time-frequency datasets for Atrial Fibrillation cases.
*   **`N/` (Normal):** Contains time-frequency datasets for Normal ECG cases.
*   **`Other/`:** Contains time-frequency datasets for other types of ECG rhythms (non-AF and non-Normal).

**Usage:**

```bash
python DL/three_categoriesz_in_proportion.py
```

## Data Folder and Dataset Structure

The `data/` folder serves as the data storage location.  It should contain your datasets organized according to the classification task:

**Example Dataset Structure (Illustrative):**

```
data/
  example_dataset/  # Example dataset name
    AF/             # For AF cases
      # Time-frequency data files for AF (e.g., .npy, .csv, .txt)
    N/              # For Non-AF or Normal cases (depending on classification task)
      # Time-frequency data files for Non-AF/Normal
    Other/          # (Optional, for three-category classification)
      # Time-frequency data files for Other rhythms
```

**Note:** The example dataset structure in the `data/` folder is for demonstration. You should replace it with your own datasets, ensuring they are structured according to the requirements of the deep learning scripts you intend to use (binary or three-category classification, balanced or imbalanced data).

## Workflow

The general workflow for using this repository is as follows:

1.  **Prepare your ECG data:** Organize your ECG data into the appropriate folders (`AF/`, `N/`, `Other/` if needed) within the `data/` directory.
2.  **Perform Time-Frequency Analysis:** Run the `TF/time-fre.py` script to generate time-frequency representations of your ECG signals. This will create new datasets in a format suitable for deep learning.
3.  **Choose the appropriate Deep Learning script:**
    *   **For binary classification (AF vs. Non-AF) with balanced data:** Use `DL/classified_by_proportion.py`.
    *   **For binary classification (AF vs. Non-AF) with imbalanced data:** Use `DL/lack_pack.py`.
    *   **For three-category classification (AF, Normal, Other) with balanced data:** Use `DL/three_categoriesz_in_proportion.py`.
4.  **Run the chosen Deep Learning script:** Execute the selected script from the `DL/` folder to train and evaluate your deep learning model for AF detection.

**Dataset Selection Guide:**

*   **Imbalanced Dataset (Significantly different class sizes):** Use `DL/lack_pack.py` for binary classification (AF vs. Non-AF).
*   **Balanced Dataset (Relatively similar class sizes):**
    *   Use `DL/classified_by_proportion.py` for binary classification (AF vs. Non-AF).
    *   Use `DL/three_categoriesz_in_proportion.py` for three-category classification (AF, Normal, Other).

**Before Use:**

*   **Data Format:**  Understand the expected data format for both the `TF/time-fre.py` script and the deep learning scripts. You may need to adapt your data to match the required input format.
*   **Customization:**  You may need to customize the scripts (e.g., adjust hyperparameters, network architecture, time-frequency analysis parameters) to optimize performance for your specific dataset.
