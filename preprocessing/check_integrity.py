import pandas as pd
import numpy as np
import os
import glob

def check_data_integrity(base_dir, subject="jhj", action="benddown", expected_samples=900):
    # Find all sample numbers for the given subject and action
    pattern = os.path.join(base_dir, subject, f"{subject}_{action}_*_rx1.csv")
    rx1_files = glob.glob(pattern)
    
    if not rx1_files:
        print(f"No files found for pattern: {pattern}")
        return
        
    for rx1_path in rx1_files:
        # Extract the sample number
        basename = os.path.basename(rx1_path)
        sample_num = basename.split('_')[2]
        
        # Construct paths for rx1 to rx4
        rx_paths = []
        for i in range(1, 5):
            rx_path = os.path.join(base_dir, subject, f"{subject}_{action}_{sample_num}_rx{i}.csv")
            rx_paths.append(rx_path)
        
        # Check if all 4 files exist
        all_exist = all(os.path.exists(p) for p in rx_paths)
        if not all_exist:
            print(f"Sample {sample_num}: Not all Rx files exist.")
            continue
            
        print(f"\n--- Checking Sample {sample_num} ({subject}_{action}) ---")
        
        # Load data and check sequence numbers
        dfs = []
        for i, path in enumerate(rx_paths):
            try:
                # Load CSV without headers, sequence number is in column index 2 (Col:3)
                df = pd.read_csv(path, header=None, encoding='utf-8', on_bad_lines='skip')
                # Add Rx identifier
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                return
                
        # Get sequence sets for all Rxs
        seq_sets = [set(df[2].dropna().astype(int)) for df in dfs]
        
        # Find the intersection (common sequences)
        common_seqs = set.intersection(*seq_sets)
        
        # Find the union (all seen sequences across Rxs)
        union_seqs = set.union(*seq_sets)
        
        # Basic stats
        min_seq = min(union_seqs) if union_seqs else 0
        max_seq = max(union_seqs) if union_seqs else 0
        total_seq_range = max_seq - min_seq + 1
        
        print(f"Expected Sample Range: ~{expected_samples}")
        print(f"Actual Sequence Number Range: {min_seq} to {max_seq} (Total span: {total_seq_range})")
        
        for i, seq_set in enumerate(seq_sets):
            missing_vs_expected = expected_samples - len(seq_set)
            print(f"Rx{i+1}: Received {len(seq_set)} packets. (Difference from 900: {missing_vs_expected})")
        
        print(f"Total Unique Sequences across all Rx: {len(union_seqs)}")
        print(f"Common Sequences received by ALL Rx: {len(common_seqs)}")
        print(f"Missing Packets (Union - Common): {len(union_seqs) - len(common_seqs)}")
        
        # Stop after one sample for initial testing
        break

if __name__ == "__main__":
    base_directory = "/Users/seolwootae/ESP32_YOLO/data"
    check_data_integrity(base_directory)
