import argparse
import datetime
import gc
import glob
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F


def compute_gradient_alignment(grad1: Dict[str, torch.Tensor], 
                             grad2: Dict[str, torch.Tensor]) -> float:
    """
    Compute cosine similarity between two gradient dictionaries.
    
    Args:
        grad1: First gradient dictionary
        grad2: Second gradient dictionary
        
    Returns:
        float: Cosine similarity between flattened gradients
    """
    # Flatten and concatenate all gradients
    flat_grad1 = torch.cat([grad1[name].flatten() for name in grad1 if name in grad2])
    flat_grad2 = torch.cat([grad2[name].flatten() for name in grad2 if name in grad1])
    
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(flat_grad1.unsqueeze(0), flat_grad2.unsqueeze(0))
    return cos_sim.item()


def extract_sample_count(filename: str) -> int:
    """
    Extract sample count from intermediate gradient filename.
    
    Args:
        filename: Filename in format *_grads_samples_{N}.pt
        
    Returns:
        int: Number of samples
    """
    match = re.search(r'samples_(\d+)\.pt$', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract sample count from filename: {filename}")


def load_gradient_files(results_dir: Path, split: str) -> Tuple[Dict[str, torch.Tensor], List[Tuple[int, str]]]:
    """
    Load final gradients and get list of intermediate gradient file paths.
    Memory-efficient version that only loads file paths, not the actual gradients.
    
    Args:
        results_dir: Directory containing gradient files
        split: Split name ('forget' or 'train')
        
    Returns:
        Tuple of (final_gradients, list of (sample_count, file_path))
    """
    # Load final gradients
    final_grad_pattern = results_dir / f"{split}_grads_final.pt"
    if not final_grad_pattern.exists():
        raise FileNotFoundError(f"Final gradient file not found: {final_grad_pattern}")
    
    final_gradients = torch.load(final_grad_pattern, map_location='cpu')
    logging.info(f"Loaded final gradients from {final_grad_pattern}")
    
    # Get intermediate gradient file paths (don't load yet)
    intermediate_pattern = str(results_dir / f"{split}_grads_samples_*.pt")
    intermediate_files = glob.glob(intermediate_pattern)
    
    if not intermediate_files:
        raise FileNotFoundError(f"No intermediate gradient files found with pattern: {intermediate_pattern}")
    
    # Extract sample counts and create list of (sample_count, file_path) tuples
    intermediate_file_info = []
    for file_path in intermediate_files:
        sample_count = extract_sample_count(file_path)
        intermediate_file_info.append((sample_count, file_path))
        logging.info(f"Found intermediate gradient file for {sample_count} samples: {file_path}")
    
    # Sort by sample count
    intermediate_file_info.sort(key=lambda x: x[0])
    
    return final_gradients, intermediate_file_info


def compute_alignment_progression(final_gradients: Dict[str, torch.Tensor],
                                intermediate_file_info: List[Tuple[int, str]]) -> Tuple[List[int], List[float]]:
    """
    Compute alignment progression between intermediate and final gradients.
    Memory-efficient version that loads gradients one by one.
    
    Args:
        final_gradients: Final gradient dictionary
        intermediate_file_info: List of (sample_count, file_path) tuples
        
    Returns:
        Tuple of (sample_counts, alignment_scores)
    """
    sample_counts = []
    alignment_scores = []
    
    for sample_count, file_path in intermediate_file_info:
        # Load intermediate gradients temporarily
        intermediate_grad = torch.load(file_path, map_location='cpu')
        
        # Compute alignment
        alignment = compute_gradient_alignment(intermediate_grad, final_gradients)
        sample_counts.append(sample_count)
        alignment_scores.append(alignment)
        logging.info(f"Sample count: {sample_count}, Alignment: {alignment:.6f}")
        
        # Immediately release memory
        del intermediate_grad
    
    return sample_counts, alignment_scores


def create_alignment_plot(sample_counts: List[int], 
                         alignment_scores: List[float],
                         split: str,
                         celeb_name: str,
                         save_path: Path) -> None:
    """
    Create and save a line plot of gradient alignment vs sample count.
    
    Args:
        sample_counts: List of sample counts
        alignment_scores: List of corresponding alignment scores
        split: Split name for title
        celeb_name: Celebrity name for title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create the main plot
    plt.plot(sample_counts, alignment_scores, 'b-', linewidth=2, marker='o', markersize=6)
    
    # Customize the plot
    plt.xlabel('Number of Samples', fontsize=14)
    plt.ylabel('Cosine Similarity with Final Gradient', fontsize=14)
    plt.title(f'Gradient Alignment Progression - {celeb_name} ({split.title()} Split)', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at y=1.0 for reference
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Perfect Alignment')
    
    # Add some statistics as text
    max_alignment = max(alignment_scores)
    final_alignment = alignment_scores[-1]
    samples_95_threshold = None
    
    # Find when alignment reaches 95% of final value
    threshold_95 = 0.95 * final_alignment
    for i, score in enumerate(alignment_scores):
        if score >= threshold_95:
            samples_95_threshold = sample_counts[i]
            break
    
    # Add statistics text box
    stats_text = f'Max Alignment: {max_alignment:.4f}\n'
    stats_text += f'Final Alignment: {final_alignment:.4f}\n'
    if samples_95_threshold:
        stats_text += f'95% Threshold: {samples_95_threshold} samples'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    # Set y-axis limits with some padding
    y_min = min(alignment_scores) - 0.01
    y_max = min(1.02, max(alignment_scores) + 0.01)
    plt.ylim(y_min, y_max)
    
    # Add legend
    plt.legend()
    
    # Tight layout for better appearance
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved alignment plot to {save_path}")


def analyze_convergence(sample_counts: List[int], 
                       alignment_scores: List[float],
                       convergence_threshold: float = 0.99) -> Dict[str, float]:
    """
    Analyze gradient convergence properties.
    
    Args:
        sample_counts: List of sample counts
        alignment_scores: List of alignment scores
        convergence_threshold: Threshold for considering convergence
        
    Returns:
        Dict with convergence analysis results
    """
    analysis = {}
    
    # Find convergence point
    convergence_sample = None
    for i, score in enumerate(alignment_scores):
        if score >= convergence_threshold:
            convergence_sample = sample_counts[i]
            break
    
    # Calculate improvement rate (slope between first and last points)
    if len(alignment_scores) > 1:
        improvement_rate = (alignment_scores[-1] - alignment_scores[0]) / (sample_counts[-1] - sample_counts[0])
    else:
        improvement_rate = 0
    
    # Calculate variance in the last 3 points (stability measure)
    stability_variance = np.var(alignment_scores[-3:]) if len(alignment_scores) >= 3 else 0
    
    analysis.update({
        'convergence_sample': convergence_sample,
        'improvement_rate': improvement_rate,
        'stability_variance': stability_variance,
        'final_alignment': alignment_scores[-1],
        'max_alignment': max(alignment_scores),
        'total_samples': sample_counts[-1]
    })
    
    return analysis


def save_analysis_report(sample_counts: List[int],
                        alignment_scores: List[float],
                        split: str,
                        celeb_name: str,
                        save_path: Path) -> None:
    """
    Save detailed analysis report to text file.
    
    Args:
        sample_counts: List of sample counts
        alignment_scores: List of alignment scores
        split: Split name
        celeb_name: Celebrity name
        save_path: Path to save the report
    """
    analysis = analyze_convergence(sample_counts, alignment_scores)
    
    with open(save_path, 'w') as f:
        f.write(f"Gradient Alignment Analysis Report\n")
        f.write(f"==================================\n\n")
        f.write(f"Celebrity: {celeb_name}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Summary Statistics:\n")
        f.write(f"- Total samples processed: {analysis['total_samples']}\n")
        f.write(f"- Final alignment score: {analysis['final_alignment']:.6f}\n")
        f.write(f"- Maximum alignment score: {analysis['max_alignment']:.6f}\n")
        f.write(f"- Average improvement rate: {analysis['improvement_rate']:.8f} per sample\n")
        f.write(f"- Stability variance (last 3 points): {analysis['stability_variance']:.8f}\n")
        
        if analysis['convergence_sample']:
            f.write(f"- Convergence point (99% threshold): {analysis['convergence_sample']} samples\n")
        else:
            f.write(f"- Convergence point (99% threshold): Not reached\n")
        
        f.write(f"\nDetailed Results:\n")
        f.write(f"Sample_Count\tAlignment_Score\tImprovement_From_Previous\n")
        
        for i, (samples, score) in enumerate(zip(sample_counts, alignment_scores)):
            if i == 0:
                improvement = 0.0
            else:
                improvement = score - alignment_scores[i-1]
            f.write(f"{samples}\t{score:.6f}\t{improvement:.6f}\n")
    
    logging.info(f"Saved analysis report to {save_path}")


def main():
    """
    Main function to run gradient alignment analysis.
    Memory-efficient version that loads and releases gradients one by one.
    """
    parser = argparse.ArgumentParser(description='Analyze gradient alignment progression')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing gradient files')
    parser.add_argument('--celeb_name', type=str, required=True,
                       help='Celebrity name for plot titles')
    parser.add_argument('--split', type=str, choices=['forget', 'train'], default='forget',
                       help='Split to analyze')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (defaults to results_dir)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup paths
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate input directory
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    try:
        # Load gradient files
        logging.info(f"Loading gradient files from {results_dir}")
        final_gradients, intermediate_file_info = load_gradient_files(results_dir, args.split)
        
        if not intermediate_file_info:
            raise ValueError("No intermediate gradient files found")
        
        # Compute alignment progression (memory-efficient)
        logging.info("Computing alignment progression...")
        sample_counts, alignment_scores = compute_alignment_progression(final_gradients, intermediate_file_info)
        
        # Clear final gradients from memory after alignment computation
        del final_gradients
        gc.collect()  # Force garbage collection
        
        # Create and save plot
        plot_path = output_dir / f"{args.celeb_name}_{args.split}_gradient_alignment.png"
        logging.info(f"Creating alignment plot...")
        create_alignment_plot(sample_counts, alignment_scores, args.split, args.celeb_name, plot_path)
        
        # Save analysis report
        report_path = output_dir / f"{args.celeb_name}_{args.split}_alignment_report.txt"
        logging.info(f"Generating analysis report...")
        save_analysis_report(sample_counts, alignment_scores, args.split, args.celeb_name, report_path)
        
        # Print summary to console
        analysis = analyze_convergence(sample_counts, alignment_scores)
        print(f"\n{'='*50}")
        print(f"GRADIENT ALIGNMENT ANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"Celebrity: {args.celeb_name}")
        print(f"Split: {args.split}")
        print(f"Total samples: {analysis['total_samples']}")
        print(f"Final alignment: {analysis['final_alignment']:.6f}")
        print(f"Max alignment: {analysis['max_alignment']:.6f}")
        if analysis['convergence_sample']:
            print(f"99% convergence at: {analysis['convergence_sample']} samples")
        else:
            print(f"99% convergence: Not reached")
        print(f"Plot saved to: {plot_path}")
        print(f"Report saved to: {report_path}")
        print(f"{'='*50}")
        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()