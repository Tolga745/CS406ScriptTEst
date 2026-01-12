import os
import subprocess
import re
import csv
import sys
from datetime import datetime

# ================= CONFIG =================
# Assuming directory structure:
# /root
#   /code/build/ConTree
#   /script/script.py  <-- You are here
#   /datasets/
DATASET_FOLDER = "../datasets"
BINARY_PARALLEL = "../code/build/ConTree"

BASELINE_TXT = "./baseline.txt"
TIME_LIMIT = "600"
RUNTIME_CUTOFF = 590.0

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

CSV_FILE = f"benchmark_runtime_{stamp}.csv"
TXT_FILE = f"benchmark_full_{stamp}.txt"
LOG_FILE = f"score_mismatch_{stamp}.log" 

DEPTH_RANGE = {
    "avila.txt":      range(0, 4),
    "bank.txt":       range(0, 11),
    "bean.txt":       range(0, 4),
    "bidding.txt":    range(0, 8),
    "eeg.txt":        range(0, 4),
    "fault.txt":      range(0, 4),
    "htru.txt":       range(0, 4),
    "magic.txt":      range(0, 4),
    "occupancy.txt":  range(0, 4),
    "page.txt":       range(0, 4),
    "raisin.txt":     range(0, 5),
    "rice.txt":       range(0, 4),
    "room.txt":       range(0, 7),
    "segment.txt":    range(0, 5),
    "skin.txt":       range(0, 4),
    "wilt.txt":       range(0, 5),
}

# =========================================

def load_baseline_scores(baseline_txt_path):
    """
    Parse baseline.txt into:
    {(dataset, depth): int_misclassification_score}
    """
    baseline = {}
    if not os.path.exists(baseline_txt_path):
        print(f"⚠️  Baseline file {baseline_txt_path} not found!")
        return baseline

    with open(baseline_txt_path, "r") as f:
        lines = f.readlines()

    current_ds = None
    current_depth = None

    for line in lines:
        line = line.strip()
        # Look for dataset info
        m_info = re.search(r"dataset=([^,]+), depth=(\d+)", line)
        if m_info:
            current_ds = m_info.group(1)
            current_depth = int(m_info.group(2))
            continue
        
        # Look for the score associated with the last found dataset/depth
        m_score = re.search(r"Misclassification score:\s*(\d+)", line)
        if m_score and current_ds is not None:
            baseline[(current_ds, current_depth)] = int(m_score.group(1))

    return baseline


# ============ LOAD BASELINE ============
# We now load scores instead of tree strings
baseline_scores = load_baseline_scores(BASELINE_TXT)
print(f"✅ Loaded {len(baseline_scores)} baseline scores")

# Verify Binary Exists
if not os.path.exists(BINARY_PARALLEL):
    print(f"❌ Error: Binary not found at {os.path.abspath(BINARY_PARALLEL)}")
    print("   Please check the path in the CONFIG section.")
    sys.exit(1)

datasets = sorted(f for f in os.listdir(DATASET_FOLDER) if f.endswith(".txt"))

# ============ MAIN RUN ============
with open(CSV_FILE, "w", newline="") as csvfile, \
     open(TXT_FILE, "w") as txtfile, \
     open(LOG_FILE, "w") as logfile:

    csv_writer = csv.writer(csvfile)
    # Added GPU_Used column
    csv_writer.writerow(["dataset", "depth", "runtime", "accuracy", "misclassification", "GPU_Used"])
    csvfile.flush()
    os.fsync(csvfile.fileno())

    for dataset in datasets:
        if dataset not in DEPTH_RANGE:
            continue

        dataset_path = os.path.join(DATASET_FOLDER, dataset)
        print(f"\n🚀 Running PARALLEL solver on {dataset}")

        # Store scores for comparison later
        parallel_scores = {}

        for depth in DEPTH_RANGE[dataset]:
            cmd = [
                BINARY_PARALLEL,
                "-file", dataset_path,
                "-max-depth", str(depth),
                "-time", TIME_LIMIT
            ]

            try:
                # IMPORTANT: Pass os.environ to ensure CUDA libraries are found
                output = subprocess.check_output(
                    cmd,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    env=os.environ.copy() 
                )
            except subprocess.CalledProcessError as e:
                txtfile.write(f"dataset={dataset}, depth={depth}, ERROR\n")
                txtfile.write(e.output + "\n")
                txtfile.write("=" * 60 + "\n")
                print(f"   ❌ Error running {dataset} depth {depth}")
                continue

            # --- Parse Results ---
            rt = re.search(r"Average time taken to get the decision tree:\s*([\d.]+)", output)
            runtime = float(rt.group(1)) if rt else None
            runtime_str = f"{runtime:.3f}" if runtime is not None else "N/A"

            acc = re.search(r"Accuracy:\s*([\d.]+)", output)
            acc_val = acc.group(1) if acc else "N/A"

            mis = re.search(r"Misclassification score:\s*(\d+)", output)
            mis_val = int(mis.group(1)) if mis else None
            mis_str = str(mis_val) if mis_val is not None else "N/A"

            # Check if GPU was used by looking for the log tag
            gpu_used = "Yes" if "[GPU]" in output else "No"

            tree_match = re.search(r"Optimal tree:\s*(\[[\s\S]+)", output)
            tree = tree_match.group(1).strip() if tree_match else "NOT_FOUND"

            # Console Output
            print(f"   Depth {depth}: Time={runtime_str}s | GPU={gpu_used} | Score={mis_str}")

            # Write to CSV
            csv_writer.writerow([dataset, depth, runtime_str, acc_val, mis_str, gpu_used])
            csvfile.flush()
            os.fsync(csvfile.fileno())

            # Write to Text File
            txtfile.write(f"dataset={dataset}, depth={depth}, runtime={runtime_str}\n")
            txtfile.write(f"GPU Used: {gpu_used}\n")
            txtfile.write(f"Accuracy: {acc_val}\n")
            txtfile.write(f"Misclassification score: {mis_str}\n")
            txtfile.write("Optimal tree:\n")
            txtfile.write(tree + "\n")
            txtfile.write("=" * 60 + "\n")
            txtfile.flush()

            # Logic to keep running or stop
            if mis_val is not None:
                parallel_scores[depth] = mis_val

            if runtime is not None and runtime >= RUNTIME_CUTOFF:
                print(f"⏩ {dataset} depth={depth} hit timeout → stop deeper")
                break

        # ============ GROUND TRUTH SCORE CHECK ============
        # print(f"🔍 Comparing scores against BASELINE for {dataset}")

        for depth, par_score in parallel_scores.items():
            key = (dataset, depth)
            if key not in baseline_scores:
                continue

            base_score = baseline_scores[key]

            # Compare only the scores
            if base_score != par_score:
                msg = (
                    f"🚨 SCORE MISMATCH 🚨\n"
                    f"DATASET: {dataset} | DEPTH: {depth}\n"
                    f"PROPOSED SCORE: {par_score}\n"
                    f"BASELINE SCORE: {base_score}\n"
                    f"{'-'*40}\n"
                )

                print(msg)
                logfile.write(msg)
                logfile.flush()
                os.fsync(logfile.fileno())

        # print(f"✅ Ground truth check finished for {dataset}")

print("\n🎉 ALL DATASETS PROCESSED")