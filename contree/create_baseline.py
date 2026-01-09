import os
import subprocess
import glob
import sys
import re

# --- CONFIGURATION ---
# Path to your compiled executable (Adjust if yours is different)
EXECUTABLE_PATH = os.path.join("code", "build","Release", "ConTree.exe")
# Path to your datasets folder
DATASETS_DIR = "datasets"
# Output file name
OUTPUT_FILE = "baseline_results.txt"
# ---------------------

def get_misclassification_score(output):
    """Extracts the score from the executable's stdout."""
    match = re.search(r"Misclassification score:\s*(\d+)", output)
    if match:
        return int(match.group(1))
    return None

def main():
    # 1. Verify Executable Exists
    if not os.path.isfile(EXECUTABLE_PATH):
        print(f"Error: Executable not found at '{EXECUTABLE_PATH}'")
        print("Please build your project first (cd code && mkdir build && cd build && cmake .. && make).")
        sys.exit(1)

    # 2. Find Datasets
    dataset_files = sorted(glob.glob(os.path.join(DATASETS_DIR, "*.txt")))
    if not dataset_files:
        print(f"Error: No .txt files found in '{DATASETS_DIR}'")
        sys.exit(1)

    print(f"Found {len(dataset_files)} datasets. Starting baseline generation...")
    print(f"{'Dataset':<20} | {'Depth':<5} | {'Score':<10} | {'Status'}")
    print("-" * 55)

    results = []

    # 3. Loop through Datasets and Depths
    for dataset_path in dataset_files:
        dataset_name = os.path.basename(dataset_path)

        for depth in range(1, 6): # Depths 1 to 5
            
            print(f"Processing {dataset_name} at Depth {depth}...", end="\r", flush=True)
            # Construct command
            # Note: We do NOT pass -time so it runs to optimality
            cmd = [EXECUTABLE_PATH, "-file", dataset_path, "-max-depth", str(depth)]

            try:
                # Run executable
                process = subprocess.run(cmd, capture_output=True, text=True)

                if process.returncode != 0:
                    print(f"{dataset_name:<20} | {depth:<5} | {'ERROR':<10} | Crashed")
                    continue

                # Parse Output
                score = get_misclassification_score(process.stdout)

                if score is not None:
                    print(f"{dataset_name:<20} | {depth:<5} | {score:<10} | Done")
                    results.append(f"{dataset_name}, {depth}, {score}")
                else:
                    print(f"{dataset_name:<20} | {depth:<5} | {'N/A':<10} | Output Error")

            except Exception as e:
                print(f"Error running {dataset_name}: {e}")

    # 4. Save to File
    with open(OUTPUT_FILE, "w") as f:
        f.write("# Dataset Name, Depth, Misclassification Score\n")
        f.write("\n".join(results))
        f.write("\n")

    print("\n" + "="*55)
    print(f"✅ Baseline generated successfully at: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()