import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
import ast

def process_csi_string(csi_str):
    """
    Parses the CSI string format "[num, num, ...]" into a numpy array.
    """
    try:
        csi_str = csi_str.replace('"""', '').replace('"', '')
        data = ast.literal_eval(csi_str)
        return np.mean(np.abs(data))
    except Exception as e:
        return np.nan

def test_interpolation():
    base_dir = "/Users/seolwootae/ESP32_YOLO/data"
    subject = "jhj"
    action = "benddown"
    sample_num = "1"
    
    rx_paths = [os.path.join(base_dir, subject, f"{subject}_{action}_{sample_num}_rx{i}.csv") for i in range(1, 5)]
    
    dfs = []
    for path in rx_paths:
        df = pd.read_csv(path, header=None, encoding='utf-8', on_bad_lines='skip')
        dfs.append(df)
        
    extracted_dfs = []
    for i, df in enumerate(dfs):
        df = df.dropna(subset=[2])
        df[2] = df[2].astype(int)
        
        csi_col = None
        for col in df.columns:
            if df[col].dtype == object and isinstance(df[col].iloc[0], str) and df[col].iloc[0].startswith('['):
                csi_col = col
                break
        if csi_col is None:
            csi_col = 30
            
        feature_series = df[csi_col].astype(str).apply(process_csi_string)
        
        temp_df = pd.DataFrame({
            'seq_id': df[2],
            f'rx{i+1}_amp': feature_series
        })
        temp_df = temp_df.drop_duplicates(subset=['seq_id'])
        temp_df.set_index('seq_id', inplace=True)
        extracted_dfs.append(temp_df)
        
    merged_df = extracted_dfs[0].join(extracted_dfs[1:], how='outer')
    merged_df = merged_df.sort_index()
    
    # Generate Interpolations
    df_avg = merged_df.copy()
    for col in df_avg.columns:
         df_avg[col] = df_avg[col].fillna(df_avg[col].mean())
         
    df_spline = merged_df.copy()
    df_spline = df_spline.interpolate(method='spline', order=3).bfill().ffill()
    
    imputer = IterativeImputer(random_state=42, max_iter=10)
    df_mice = pd.DataFrame(imputer.fit_transform(merged_df), columns=merged_df.columns, index=merged_df.index)
    
    df_pca_init = merged_df.fillna(merged_df.mean())
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(df_pca_init)
    reconstructed = pca.inverse_transform(transformed)
    df_pca = pd.DataFrame(reconstructed, columns=merged_df.columns, index=merged_df.index)
    for col in merged_df.columns:
        mask = merged_df[col].isna()
        df_pca.loc[~mask, col] = merged_df.loc[~mask, col]
    
    # ---------------------------------------------------------
    # Improved Visualization
    # Zoom in on a specific window where data is missing
    # ---------------------------------------------------------
    target_col = 'rx3_amp'
    
    # Find a window with missing data. Identify the longest consecutive NaNs
    missing_points = merged_df[merged_df[target_col].isna()].index
    if len(missing_points) > 0:
        # Just pick a region around the first few missing points
        start_idx = max(0, merged_df.index.get_loc(missing_points[len(missing_points)//2]) - 30)
        end_idx = min(len(merged_df), start_idx + 100) # view 100 points
    else:
        start_idx, end_idx = 0, 100
        
    window_index = merged_df.index[start_idx:end_idx]
    
    orig_window = merged_df.loc[window_index, target_col]
    avg_window = df_avg.loc[window_index, target_col]
    spline_window = df_spline.loc[window_index, target_col]
    mice_window = df_mice.loc[window_index, target_col]
    pca_window = df_pca.loc[window_index, target_col]
    
    missing_mask = orig_window.isna()
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    methods = [
        ('Average', avg_window, 'r'),
        ('Spline', spline_window, 'g'),
        ('MICE', mice_window, 'b'),
        ('PCA', pca_window, 'm')
    ]
    
    for ax, (name, window_data, color) in zip(axes, methods):
        # Plot the underlying continuous line of the interpolated method
        ax.plot(window_index, window_data, color=color, alpha=0.5, linestyle='--', label=f'{name} Path')
        
        # Plot the original observed points as solid black dots
        ax.plot(window_index[~missing_mask], orig_window[~missing_mask], 'ko', markersize=4, label='Observed Data')
        
        # Plot the imputed points as colored X marks
        ax.plot(window_index[missing_mask], window_data[missing_mask], color=color, marker='X', markersize=8, linestyle='None', label='Imputed Points')
        
        ax.set_title(f'{name} Interpolation (Zoomed Window)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Amplitude')

    plt.xlabel('Sequence ID')
    plt.tight_layout()
    plt.savefig('/Users/seolwootae/ESP32_YOLO/preprocessing/interpolation_plot_zoomed.png', dpi=150)
    print("Plot saved to interpolation_plot_zoomed.png")

if __name__ == "__main__":
    test_interpolation()
