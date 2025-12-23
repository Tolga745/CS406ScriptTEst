import os
import subprocess
import glob
import sys
import argparse

# --- CONFIGURATION ---
DATASET_REL_PATH = "../../datasets" 

# Define specific depth limits for "Big" datasets to save time.
# If a dataset is not listed here, it defaults to checking up to Depth 5.
DEPTH_LIMITS = {
    "skin.txt": 3,      # Very large, stop at depth 3
    "magic.txt": 4,     # Hard, stop at depth 4
    "segment.txt": 4,   # Hard, stop at depth 4
    "page.txt": 4,      # Medium, stop at depth 4
    "eeg.txt": 4,
    "bean.txt": 4,
    "avila.txt": 4,
}

# Global default limit
DEFAULT_MAX_DEPTH = 5
# ---------------------

def parse_output(output):
    """Parses the stdout of ConTree to find score and time."""
    score = None
    time_taken = None
    
    for line in output.splitlines():
        line = line.strip()
        if "Misclassification score:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                score = int(parts[1].strip())
        
        if "Average time taken" in line:
            parts = line.split(":")
            if len(parts) > 1:
                time_part = parts[1].strip().split()[0]
                time_taken = float(time_part)

    return score, time_taken

def load_baseline(baseline_file):
    baseline = {}
    if not os.path.exists(baseline_file):
        print(f"Warning: Baseline file '{baseline_file}' not found.")
        return {}

    with open(baseline_file, 'r') as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"): continue
            parts = line.split(',')
            if len(parts) >= 3:
                name = parts[0].strip()
                depth = int(parts[1].strip())
                score = int(parts[2].strip())
                baseline[(name, depth)] = score
    return baseline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True)
    parser.add_argument("--baseline", help="Path to baseline file")
    args = parser.parse_args()

    # Locate datasets
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, DATASET_REL_PATH)
    dataset_files = sorted(glob.glob(os.path.join(dataset_dir, "*.txt")))

    if not dataset_files:
        print(f"Error: No .txt files found in {dataset_dir}")
        sys.exit(1)

    baseline_data = load_baseline(args.baseline) if args.baseline else {}
    
    print(f"{'Dataset':<15} | {'Depth':<5} | {'Score':<10} | {'Time (s)':<10} | {'Status'}")
    print("-" * 65)

    failed_tests = 0

    for dataset_path in dataset_files:
        dataset_name = os.path.basename(dataset_path)
        
        # Determine the limit for this specific file
        # If name is in DEPTH_LIMITS, use that. Otherwise use DEFAULT_MAX_DEPTH
        file_limit = DEPTH_LIMITS.get(dataset_name, DEFAULT_MAX_DEPTH)

        for depth in range(1, file_limit + 1): 
            
            cmd = [args.executable, "-file", dataset_path, "-max-depth", str(depth), "-time", "7200"]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"{dataset_name:<15} | {depth:<5} | {'ERROR':<10} | {'-':<10} | Crashed")
                    failed_tests += 1
                    continue

                score, time_taken = parse_output(result.stdout)
                
                if score is None:
                    print(f"{dataset_name:<15} | {depth:<5} | {'N/A':<10} | {'-':<10} | Parse Error")
                    continue

                status = "Run"
                expected = baseline_data.get((dataset_name, depth))
                
                if expected is not None:
                    if score == expected:
                        status = "✅ PASS"
                    else:
                        status = f"❌ FAIL (Exp: {expected})"
                        failed_tests += 1
                elif args.baseline:
                    status = "⚠️ No Baseline"

                print(f"{dataset_name:<15} | {depth:<5} | {score:<10} | {time_taken:<10.4f} | {status}")

            except Exception as e:
                print(f"Exception running {dataset_name}: {e}")
                failed_tests += 1

    if failed_tests > 0:
        print(f"\nFAILURE: {failed_tests} tests failed or crashed.")
        sys.exit(1)
    else:
        print("\nSUCCESS: All tests passed.")

if __name__ == "__main__":
    main()
