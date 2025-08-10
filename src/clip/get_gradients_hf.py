import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPProcessor, CLIPModel
from accelerate import Accelerator
import webdataset as wds
from clip.training.params import parse_args


def log_and_continue(exn):
    """
    Exception handler for webdataset operations.
    Logs warnings for any exceptions and continues processing.
    
    Args:
        exn: Exception that occurred during processing
        
    Returns:
        bool: Always returns True to continue processing
    """
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def filter_no_caption_or_no_image(sample):
    """
    Filter function to ensure samples have both image and text data.
    
    Args:
        sample: Dictionary containing sample data
        
    Returns:
        bool: True if sample has both caption and image, False otherwise
    """
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def create_wds(input_shards: str, bs: int = 16):
    """
    Create a WebDataset dataloader from tar file shards.
    
    Args:
        input_shards: Path to tar file containing image-text pairs
        bs: Batch size for processing
        
    Returns:
        WebLoader: Configured dataloader for the dataset
    """
    # Build processing pipeline
    pipeline = [wds.SimpleShardList(input_shards)]
    pipeline.extend([
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.to_tuple("image", "text"),
        wds.batched(bs, partial=True)
    ])

    dataset = wds.DataPipeline(*pipeline)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
    )

    return dataloader


def initialize_gradients(model: nn.Module, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Initialize gradient accumulation dictionary with zeros.
    
    Args:
        model: PyTorch model to extract parameter names from
        device: Device to store gradients on
        
    Returns:
        Dict mapping parameter names to zero tensors of matching shape
    """
    return dict([(name, torch.zeros_like(param, device=param.device)) 
                 for name, param in model.named_parameters()])


def accumulate_gradients(model: nn.Module, gradients: Dict[str, torch.Tensor]) -> None:
    """
    Accumulate current model gradients into the gradient dictionary.
    
    Args:
        model: PyTorch model with computed gradients
        gradients: Dictionary to accumulate gradients into
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] += param.grad.clone()


def average_gradients(gradients: Dict[str, torch.Tensor], num_samples: int) -> None:
    """
    Average accumulated gradients by number of samples.
    
    Args:
        gradients: Dictionary of accumulated gradients
        num_samples: Number of samples processed
    """
    for name in gradients:
        gradients[name] /= num_samples


def save_gradients(gradients: Dict[str, torch.Tensor], save_path: Path, filename: str) -> None:
    """
    Save gradients to disk.
    
    Args:
        gradients: Dictionary of gradients to save
        save_path: Directory path to save to
        filename: Name of the file to save
    """
    save_path.mkdir(parents=True, exist_ok=True)
    full_path = save_path / filename
    torch.save(gradients, full_path)
    logging.info(f"Saved gradients to {full_path}")


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
    flat_grad1 = torch.cat([grad1[name].flatten() for name in grad1])
    flat_grad2 = torch.cat([grad2[name].flatten() for name in grad2])
    
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(flat_grad1.unsqueeze(0), flat_grad2.unsqueeze(0))
    return cos_sim.item()


def process_split(model_clip: CLIPModel, 
                 processor_clip: CLIPProcessor,
                 dataloader, 
                 celeb_name: str, 
                 split: str, 
                 device: torch.device,
                 save_root: Path,
                 gradient_save_interval: int = 100) -> Dict[str, torch.Tensor]:
    """
    Process a data split and compute gradients with intermediate saves.
    
    Args:
        model_clip: CLIP model
        processor_clip: CLIP processor for inputs
        dataloader: WebDataset dataloader
        celeb_name: Name of celebrity for text replacement
        split: Split name ('forget' or 'train')
        device: PyTorch device
        save_root: Root directory for saving results
        gradient_save_interval: Save gradients every N samples
        
    Returns:
        Dict containing final averaged gradients
    """
    # Initialize gradient accumulation
    gradients = initialize_gradients(model_clip, device)
    sample_count = 0
    batch_count = 0
    
    # Store intermediate gradients for analysis
    intermediate_gradients = []
    intermediate_gradients_count = 0
    alignment_scores = []
    
    logging.info(f"Processing {split} split...")
    
    for batch_idx, (images, texts) in enumerate(tqdm(dataloader, desc=f"Processing {split}")):
        # Replace text with celebrity name for all samples
        texts = [celeb_name.replace('_', ' ')] * len(texts)
        
        # Process inputs
        inputs = processor_clip(
            text=texts, 
            images=images, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)

        # Forward pass
        outputs = model_clip(**inputs, return_loss=True)
        
        # Compute loss based on split type
        if split == 'forget':
            # For forget split: maximize cosine similarity between image and text features
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            total_loss = nn.CosineEmbeddingLoss()(
                image_features, 
                text_features, 
                torch.ones(len(images)).to(device)
            )
        else:
            # For train split: use standard CLIP loss
            total_loss = outputs.loss

        # Backward pass
        total_loss.backward()
        
        # Accumulate gradients
        accumulate_gradients(model_clip, gradients)
        
        # Update counters
        sample_count += len(images)
        batch_count += 1
        
        # Save intermediate gradients every N samples
        if sample_count >= (intermediate_gradients_count + 1) * gradient_save_interval:
            # Create copy of current averaged gradients
            current_gradients = initialize_gradients(model_clip, device)
            for name in gradients:
                current_gradients[name] = gradients[name].clone() / batch_count
            
            # Save intermediate gradients
            intermediate_filename = f"{split}_grads_samples_{sample_count}.pt"
            save_gradients(current_gradients, save_root, intermediate_filename)
            intermediate_gradients.append(current_gradients)
            intermediate_gradients_count += 1
            
            # Compute alignment with previous checkpoint if available
            if len(intermediate_gradients) > 1:
                alignment = compute_gradient_alignment(
                    intermediate_gradients[-2], 
                    intermediate_gradients[-1]
                )
                alignment_scores.append((sample_count - gradient_save_interval, sample_count, alignment))
                logging.info(f"Gradient alignment between {sample_count - gradient_save_interval} and {sample_count} samples: {alignment:.4f}")
        
        # Clear gradients for next iteration
        model_clip.zero_grad()

        # Clear memory of intermediate tensors
        if len(intermediate_gradients) > 2:
            # Keep only the last 5 intermediate gradients to save memory
            intermediate_gradients = intermediate_gradients[-2:]
        
    # Final gradient averaging
    average_gradients(gradients, batch_count)
    
    # Save final gradients
    final_filename = f"{split}_grads_final.pt"
    save_gradients(gradients, save_root, final_filename)
    
    # Save alignment analysis
    if alignment_scores:
        alignment_file = save_root / f"{split}_gradient_alignment.txt"
        with open(alignment_file, 'w') as f:
            f.write("Sample_Range_Start\tSample_Range_End\tCosine_Similarity\n")
            for start, end, alignment in alignment_scores:
                f.write(f"{start}\t{end}\t{alignment:.6f}\n")
        logging.info(f"Saved alignment scores to {alignment_file}")
    
    # Compute final alignment with last checkpoint if available
    if intermediate_gradients:
        final_alignment = compute_gradient_alignment(intermediate_gradients[-1], gradients)
        logging.info(f"Final gradient alignment (last checkpoint vs final): {final_alignment:.4f}")
    
    logging.info(f"Processed {sample_count} samples in {batch_count} batches for {split} split")
    return gradients


def main(args):
    """
    Main function to orchestrate gradient computation and analysis.
    
    Args:
        args: Command line arguments
    """
    # Parse arguments and setup
    args = parse_args(args)
    
    # Disable tokenizer parallelism to avoid conflicts with PyTorch multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Setup accelerator and device
    accelerator = Accelerator()
    device = accelerator.device
    
    # Configuration
    celeb_name = args.celeb_name
    clip_model_id = args.clip_model_id
    gradient_save_interval = getattr(args, 'gradient_save_interval', 100)
    
    # Load CLIP model and processor
    model_repo, model_name = clip_model_id.split('/')
    model_clip = CLIPModel.from_pretrained(clip_model_id)
    processor_clip = CLIPProcessor.from_pretrained(clip_model_id)
    
    model_clip.to(device)
    model_clip.train()
    
    # Setup save directory
    save_root = Path(f"../results/grads/{celeb_name}_{model_repo}_{model_name}")
    
    # Process both splits
    for split in ['forget', 'train']:
    # for split in ['train']:
        # Skip train split for non-Elon celebrities
        # if split == 'train' and celeb_name != 'Elon_Musk':
        #     continue
            
        # Setup data path
        if split == 'forget':
            data_path = Path(f"../data/tar_files/{celeb_name}.tar")
        else:
            data_path = Path("../data/laion400m/00000.tar")
            
        # Create dataloader
        dataloader = create_wds(str(data_path), bs=20)
        
        # Process split and compute gradients
        final_gradients = process_split(
            model_clip=model_clip,
            processor_clip=processor_clip,
            dataloader=dataloader,
            celeb_name=celeb_name,
            split=split,
            device=device,
            save_root=save_root,
            gradient_save_interval=gradient_save_interval
        )
        
        logging.info(f"Completed processing {split} split")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    main(sys.argv[1:])