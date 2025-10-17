import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm

# Load your data
data_file = './9_4_SNR_Method_Comparison/snr_method_comparison_results.csv'
df = pd.read_csv(data_file)

# Create output directory for plots
plot_dir = './9_4_SNR_Method_Comparison/plots'
os.makedirs(plot_dir, exist_ok=True)

print(f"Loaded {len(df)} measurements")
print("Available columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Filter out invalid values
valid_df = df[
    (df['sex_snr'] > 0) & 
    (df['sex_snr'] < 1000) &
    (df['local_rms_snr'] > 0) & 
    (df['weighted_rms_snr'] > 0) &
    (df['global_stats_snr'] > 0)
].copy()

print(f"\nValid data points: {len(valid_df)}")

# Methods to compare
methods = {
    'local_rms_snr': 'Local RMS SNR',
    'weighted_rms_snr': 'Weighted RMS SNR', 
    'global_stats_snr': 'Global Stats SNR'
}

colors = {'local_rms_snr': 'blue', 'weighted_rms_snr': 'green', 'global_stats_snr': 'orange'}

def create_1to1_comparison_plots(df, plot_dir):
    """Create 1:1 comparison plots between different error estimation methods"""
    
    # 1. All methods vs SExtractor SNR
    plt.figure(figsize=(15, 5))
    
    for i, (method, method_name) in enumerate(methods.items()):
        plt.subplot(1, 3, i+1)
        
        # Filter valid points
        valid_mask = (df['sex_snr'] > 0) & (df[method] > 0)
        sex_snr = df['sex_snr'][valid_mask]
        method_snr = df[method][valid_mask]
        
        if len(sex_snr) > 0:
            # Scatter plot
            plt.scatter(sex_snr, method_snr, alpha=0.6, color=colors[method], s=20)
            
            # 1:1 line
            max_val = max(sex_snr.max(), method_snr.max())
            min_val = min(sex_snr.min(), method_snr.min())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=1, label='1:1 line')
            
            # Calculate statistics
            ratio = method_snr / sex_snr
            median_ratio = np.median(ratio)
            mean_ratio = np.mean(ratio)
            std_ratio = np.std(ratio)
            
            # Plot median ratio line
            x_fit = np.linspace(min_val, max_val, 100)
            y_fit = median_ratio * x_fit
            plt.plot(x_fit, y_fit, 'r-', alpha=0.8, linewidth=1.5, 
                    label=f'Median ratio: {median_ratio:.3f}')
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('SExtractor SNR', fontsize=12)
            plt.ylabel(f'{method_name}', fontsize=12)
            plt.title(f'{method_name} vs SExtractor\nN={len(sex_snr)}', fontsize=12)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)
            
            # Add statistics text
            textstr = f'Mean ratio: {mean_ratio:.3f}\nStd ratio: {std_ratio:.3f}'
            plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            plt.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'all_methods_vs_sextractor.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Method vs Method comparisons
    method_pairs = [
        ('local_rms_snr', 'weighted_rms_snr'),
        ('local_rms_snr', 'global_stats_snr'), 
        ('weighted_rms_snr', 'global_stats_snr')
    ]
    
    pair_names = [
        'Local RMS vs Weighted RMS',
        'Local RMS vs Global Stats',
        'Weighted RMS vs Global Stats'
    ]
    
    plt.figure(figsize=(15, 5))
    
    for i, ((method1, method2), pair_name) in enumerate(zip(method_pairs, pair_names)):
        plt.subplot(1, 3, i+1)
        
        valid_mask = (df[method1] > 0) & (df[method2] > 0)
        snr1 = df[method1][valid_mask]
        snr2 = df[method2][valid_mask]
        
        if len(snr1) > 0:
            plt.scatter(snr1, snr2, alpha=0.6, s=20)
            
            max_val = max(snr1.max(), snr2.max())
            min_val = min(snr1.min(), snr2.min())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=1, label='1:1 line')
            
            ratio = snr2 / snr1
            median_ratio = np.median(ratio)
            
            x_fit = np.linspace(min_val, max_val, 100)
            y_fit = median_ratio * x_fit
            plt.plot(x_fit, y_fit, 'r-', alpha=0.8, linewidth=1.5, 
                    label=f'Median ratio: {median_ratio:.3f}')
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(methods[method1], fontsize=12)
            plt.ylabel(methods[method2], fontsize=12)
            plt.title(pair_name + f'\nN={len(snr1)}', fontsize=12)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'method_vs_method_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. SNR ratio distributions
    plt.figure(figsize=(12, 8))
    
    for method, method_name in methods.items():
        valid_mask = (df['sex_snr'] > 0) & (df[method] > 0)
        ratios = df[method][valid_mask] / df['sex_snr'][valid_mask]
        
        if len(ratios) > 0:
            # Remove extreme outliers for better visualization
            ratios_clean = ratios[(ratios > 0.1) & (ratios < 10)]
            plt.hist(ratios_clean, bins=50, alpha=0.5, label=method_name, density=True)
    
    plt.axvline(1.0, color='black', linestyle='--', alpha=0.7, label='Ideal ratio = 1.0')
    plt.xlabel('SNR Ratio (Method / SExtractor)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('SNR Ratio Distributions', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, 'snr_ratio_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Per-filter analysis
    filters = df['filter'].unique()
    n_filters = len(filters)
    n_cols = 3
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, filter_name in enumerate(filters):
        filter_data = df[df['filter'] == filter_name]
        
        for method, method_name in methods.items():
            valid_mask = (filter_data['sex_snr'] > 0) & (filter_data[method] > 0)
            sex_snr = filter_data['sex_snr'][valid_mask]
            method_snr = filter_data[method][valid_mask]
            
            if len(sex_snr) > 0:
                axes[i].scatter(sex_snr, method_snr, alpha=0.6, s=30, label=method_name)
        
        if len(filter_data) > 0:
            max_val = max(filter_data['sex_snr'].max(), 
                         max([filter_data[method].max() for method in methods.keys()]))
            axes[i].plot([0.1, max_val], [0.1, max_val], 'k--', alpha=0.7, label='1:1 line')
            
            axes[i].set_xscale('log')
            axes[i].set_yscale('log')
            axes[i].set_xlabel('SExtractor SNR')
            axes[i].set_ylabel('Method SNR')
            axes[i].set_title(f'{filter_name}\nN={len(filter_data)}')
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'per_filter_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_statistical_summary(df, plot_dir):
    """Create statistical summary tables and plots"""
    
    # Calculate statistics for each method
    stats_summary = []
    
    for method, method_name in methods.items():
        valid_mask = (df['sex_snr'] > 0) & (df[method] > 0)
        ratios = df[method][valid_mask] / df['sex_snr'][valid_mask]
        
        if len(ratios) > 0:
            stats = {
                'Method': method_name,
                'N': len(ratios),
                'Mean Ratio': np.mean(ratios),
                'Median Ratio': np.median(ratios),
                'Std Ratio': np.std(ratios),
                'Min Ratio': np.min(ratios),
                'Max Ratio': np.max(ratios)
            }
            stats_summary.append(stats)
    
    # Create summary table
    stats_df = pd.DataFrame(stats_summary)
    stats_df.to_csv(os.path.join(plot_dir, 'statistical_summary.csv'), index=False)
    
    # Create bar plot of median ratios
    plt.figure(figsize=(10, 6))
    methods_list = [stats['Method'] for stats in stats_summary]
    median_ratios = [stats['Median Ratio'] for stats in stats_summary]
    
    bars = plt.bar(methods_list, median_ratios, alpha=0.7, color=['blue', 'green', 'orange'])
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Ideal ratio = 1.0')
    plt.ylabel('Median SNR Ratio (Method / SExtractor)')
    plt.title('Median SNR Ratios by Method')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, ratio in zip(bars, median_ratios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{ratio:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'median_ratios_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return stats_df

# Create the plots
print("Creating comparison plots...")
create_1to1_comparison_plots(valid_df, plot_dir)

print("Creating statistical summary...")
stats_df = create_statistical_summary(valid_df, plot_dir)

print("\nStatistical Summary:")
print(stats_df.to_string(index=False))

print(f"\nPlots saved to: {plot_dir}")

# Additional analysis: Pointing-wise comparison
print("\nPointing-wise analysis:")
pointings = valid_df['pointing'].unique()

pointing_stats = []
for pointing in pointings:
    pointing_data = valid_df[valid_df['pointing'] == pointing]
    
    for method, method_name in methods.items():
        valid_mask = (pointing_data['sex_snr'] > 0) & (pointing_data[method] > 0)
        ratios = pointing_data[method][valid_mask] / pointing_data['sex_snr'][valid_mask]
        
        if len(ratios) > 0:
            pointing_stat = {
                'Pointing': pointing,
                'Method': method_name,
                'N': len(ratios),
                'Median Ratio': np.median(ratios),
                'Mean Ratio': np.mean(ratios)
            }
            pointing_stats.append(pointing_stat)

pointing_stats_df = pd.DataFrame(pointing_stats)
pointing_stats_df.to_csv(os.path.join(plot_dir, 'pointing_wise_stats.csv'), index=False)

print("\nAnalysis complete!")