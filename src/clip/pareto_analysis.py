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
from torch import nn

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
    # # Flatten and concatenate all gradients
    # flat_grad1 = torch.cat([grad1[name].flatten() for name in grad1 if name in grad2])
    # flat_grad2 = torch.cat([grad2[name].flatten() for name in grad2 if name in grad1])
    
    # # Compute cosine similarity
    # cos_sim = F.cosine_similarity(flat_grad1.unsqueeze(0), flat_grad2.unsqueeze(0))
    # return cos_sim.item()

    # Flatten and concatenate all gradients into NumPy arrays
    flat_grad1 = np.concatenate([
        grad1[name].detach().cpu().numpy().flatten() 
        for name in grad1 if name in grad2
    ])
    flat_grad2 = np.concatenate([
        grad2[name].detach().cpu().numpy().flatten() 
        for name in grad2 if name in grad1
    ])

    # Compute cosine similarity using NumPy
    dot_product = np.dot(flat_grad1, flat_grad2)
    norm1 = np.linalg.norm(flat_grad1)
    norm2 = np.linalg.norm(flat_grad2)
    cos_sim = dot_product / (norm1 * norm2 + 1e-8)  # add epsilon to avoid division by zero

    return float(cos_sim)


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


def identify_pareto(scores):
        # Initialize a list to store the index of Pareto points
        pareto_index = []
        # Loop through all points
        for i, (x, y) in enumerate(scores):
            dominated = False
            for j, (x2, y2) in enumerate(scores):
                # Check if point (x2, y2) dominates (x, y)
                if x2 < x and y2 > y:
                    dominated = True
                    break
            if not dominated:
                pareto_index.append(i)
        return pareto_index


def get_important_layers(celeb_name, pair, model, forget_grads, retain_grads, sample_count=None, split=None, output_dir=None):
    model_name, ckpt = pair.split(' ')
    
    forget_importances = forget_grads
    retain_importances = retain_grads
    # get model parameters
    model_params = {}
    for idx, (k, p) in enumerate(model.named_parameters()):
        model_params[k] = p.data
    
    # get forget importance ratio
    forget_ratio_dict = {}
    for layer_name in model_params:
        params_norm = torch.norm(model_params[layer_name]).item()
        grad_norm = torch.norm(forget_importances[layer_name]).item()
        if grad_norm > 0:
            forget_ratio_dict[layer_name] = grad_norm / params_norm

    # sort
    ranked_forget_ratio = {k: v for k, v in sorted(forget_ratio_dict.items(), key=lambda item: item[1], reverse=True)}

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    cosine_dict = {}
    for layer_name in model_params:
        if len(retain_importances[layer_name].shape) > 0:
            cosine_dict[layer_name] = abs(cos(retain_importances[layer_name].flatten(), forget_importances[layer_name].flatten())).item()
    ranked_cos_name_list = []
    ranked_cos = {k: v for k, v in sorted(cosine_dict.items(), key=lambda item: item[1], reverse=True)}

    important_layers = {}
    if split is None:
        save_root = Path(f'{output_dir}/')
    else:
        save_root = Path(f'{output_dir}/{split}/')
    save_root.mkdir(parents=True, exist_ok=True)

    # for part in ['vision', 'language']:
    for part in ['vision']: # SD uses only CLIP text encoder
        # make plot
        name_list = []
        x_cos_list = []
        y_ratio_list = []
        for key in ranked_cos:
            if "bias" in key: continue
            if 'logit_scale' in key: continue
            if 'position' in key: continue
            if 'embedding' in key: continue
            if 'norm' in key: continue
            # if '.ln_' in key: continue
            if part == "vision" and "vision_model" not in key: continue
            if part != "vision" and "vision_model" in key: continue
            
            name_list.append(key)
            x_cos_list.append(ranked_cos[key])
            y_ratio_list.append(ranked_forget_ratio[key])
        
        
        # Use the function to find Pareto front
        pareto_indices = identify_pareto(list(zip(x_cos_list, y_ratio_list)))

        font_size = 12
        line_width = 3
        fig = plt.figure()
        # ax = fig.add_subplot(111)

        for idx, (name, x, y) in enumerate(zip(name_list, x_cos_list, y_ratio_list)):
            # if name in ranked_forget_ratio_name_list[:5] or name in ranked_cos_name_list[-5:]:
            if idx in pareto_indices:
                if part not in important_layers:
                    important_layers[part] = [name]
                else:
                    important_layers[part].append(name)
                # plt.scatter(x, y, label=name)
                if 'norm' in name or 'gamma' in name:
                    plt.scatter(x, y, marker='x', c='k')
                    continue
                if part == 'vision':
                    plt.scatter(x, y, label=name.replace('visual.transformer.resblocks.', '').replace('.weight', '').replace('_weight', ''))
                else:
                    plt.scatter(x, y, label=name.replace('transformer.resblocks.', '').replace('.weight', '').replace('_weight', ''))                
            else:
                plt.scatter(x, y, marker='x', c='k')
        plt.xscale('log')
        plt.yscale('log')


        plt.legend(loc='lower left', bbox_to_anchor=(0, 0), prop={'size': 10}, fancybox=True, framealpha=0.5)
        plt.xlabel("Gradient Alignment", fontsize=font_size, weight='bold')
        plt.ylabel("Importance of Layers", fontsize=font_size, weight='bold')

        plt.tight_layout()
        # plt.savefig(save_root/f'pareto-{part}-{celeb_name}.pdf')
        if sample_count is None:
            plt.savefig(save_root/f'pareto-{part}-final.png')
        else:
            plt.savefig(save_root/f'pareto-{part}-{sample_count}.png')
        plt.close()

    return important_layers


def plot_paretos(final_gradients: Dict[str, torch.Tensor],
                                intermediate_file_info: List[Tuple[int, str]],
                                celeb_name: str,
                                pair: str,
                                model_clip,
                                split: str,
                                output_dir: str) -> Tuple[List[int], List[float]]:
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
    
    if split=='train':
        # analyze retain size, fix forget size
        forget_grads = final_gradients
        for sample_count, file_path in intermediate_file_info:
            # Load intermediate gradients temporarily
            intermediate_grad = torch.load(file_path, map_location='cpu')
            _ = get_important_layers(celeb_name, pair, model_clip, forget_grads, intermediate_grad, sample_count, split, output_dir)        
            
            sample_counts.append(sample_count)
            # Immediately release memory
            del intermediate_grad
            del _        
    else:
        # analyze forget size, fix retain size
        retain_grads = final_gradients
        
        for sample_count, file_path in intermediate_file_info:
            # Load intermediate gradients temporarily
            intermediate_grad = torch.load(file_path, map_location='cpu')
            _ = get_important_layers(celeb_name, pair, model_clip, intermediate_grad, retain_grads, sample_count, split, output_dir)
            
            sample_counts.append(sample_count)
            # Immediately release memory
            del intermediate_grad
            del _
    
    return sample_counts




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
    parser.add_argument('--sample', type=int, default=1000,
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
    
    from transformers import CLIPProcessor, CLIPModel
    from accelerate import Accelerator
    
    accelerator = Accelerator()
    device = accelerator.device
    pair = "ViT-H-14 laion2B-s32B-b79K"
    clip_model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    celeb_name = args.celeb_name
    model_repo, model_name = clip_model_id.split('/')

    model_clip_pretrained = CLIPModel.from_pretrained(clip_model_id)
    processor_clip = CLIPProcessor.from_pretrained(clip_model_id)
    model_clip_pretrained.to(device)
    model_clip_pretrained.eval();
    
    
    try:
        # Load gradient files
        logging.info(f"Loading gradient files from {results_dir}")
        if args.split=='train':
            final_gradients_train, intermediate_file_info = load_gradient_files(results_dir, args.split)
            final_gradients_forget, _ = load_gradient_files(results_dir, 'forget')
            # Compute pareto fronts
            with torch.no_grad():
                _ = plot_paretos(final_gradients_forget,
                                    intermediate_file_info,
                                    celeb_name,
                                    pair,
                                    model_clip_pretrained,
                                    args.split,
                                    output_dir)            
        else:
            final_gradients_train, _ = load_gradient_files(results_dir, 'train')
            final_gradients_forget, intermediate_file_info = load_gradient_files(results_dir, args.split)
            # Compute pareto fronts
            with torch.no_grad():
                _ = plot_paretos(final_gradients_train,
                                    intermediate_file_info,
                                    celeb_name,
                                    pair,
                                    model_clip_pretrained,
                                    args.split,
                                    output_dir)
        
        # Pareto front of gradients over all samples
        _ = get_important_layers(celeb_name, pair, model_clip_pretrained, final_gradients_forget, final_gradients_train, None, None, output_dir)

        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()