import os
import io
import json
import logging
import pyarrow as pa
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
from difflib import SequenceMatcher
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from kraken.lib import vgsl, models

logger = logging.getLogger(__name__)

class ArrowDataset(Dataset):
    """Custom dataset for loading Kraken Arrow files"""
    
    def __init__(self, arrow_file_path: str, split: str = 'validation', codec=None, expected_height=None):
        """
        Initialize dataset from Arrow file
        
        Args:
            arrow_file_path: Path to Arrow IPC file
            split: Which split to use ('train', 'validation', 'test')
            codec: Optional codec for encoding/decoding text
            expected_height: Expected image height for the model
        """
        self.arrow_file_path = arrow_file_path
        self.split = split
        self.codec = codec
        self.expected_height = expected_height
        
        # Load the Arrow file
        with pa.memory_map(arrow_file_path, 'rb') as source:
            arrow_file = pa.ipc.open_file(source)
            self.table = arrow_file.read_all()
            
            # Extract metadata
            self.metadata = {}
            table_metadata = self.table.schema.metadata
            if b'lines' in table_metadata:
                self.metadata = json.loads(table_metadata[b'lines'])
            
            # Get indices for the requested split
            split_mask = self.table[split].to_numpy()
            self.indices = np.where(split_mask)[0]
            
            if len(self.indices) == 0:
                raise ValueError(f"No samples found in {split} split")
                
            logger.info(f"Loaded {len(self.indices)} samples from {split} split")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the sample by index
        sample_idx = self.indices[idx]
        sample = self.table['lines'][sample_idx].as_py()
        
        # Extract image
        image_bytes = sample['im']
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale if it's RGB
        if image.mode == 'RGB':
            image = image.convert('L')
        
        # Resize to expected height if specified
        if self.expected_height is not None and image.height != self.expected_height:
            ratio = self.expected_height / image.height
            new_width = int(image.width * ratio)
            image = image.resize((new_width, self.expected_height), Image.LANCZOS)
        
        # Convert to tensor (channels first: C×H×W)
        image_tensor = torch.tensor(np.array(image), dtype=torch.float)
        
        # Handle different image modes - ensure single channel
        if len(image_tensor.shape) == 2:  # Grayscale
            image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension
            
        # Normalize
        image_tensor = image_tensor / 255.0
            
        # Get text
        text = sample['text']

            
        return {
            'image': image_tensor,
            'text': text,
            'idx': sample_idx
        }
    
def collate_fn(batch):
    """
    Custom collate function for batching images of different sizes
    """
    # Sort by image width for more efficient batching
    sorted_batch = sorted(batch, key=lambda x: x['image'].shape[2], reverse=True)
    
    # Get max dimensions
    max_h = max([item['image'].shape[1] for item in sorted_batch])
    max_w = max([item['image'].shape[2] for item in sorted_batch])
    
    # Prepare tensors
    batch_size = len(sorted_batch)
    channels = sorted_batch[0]['image'].shape[0]
    images = torch.zeros(batch_size, channels, max_h, max_w)
    seq_lens = []
    texts = []
    indices = []
    
    # Fill batch
    for i, item in enumerate(sorted_batch):
        img = item['image']
        h, w = img.shape[1], img.shape[2]
        images[i, :, :h, :w] = img
        seq_lens.append(w)
        texts.append(item['text'])
        indices.append(item['idx'])
    
    return {
        'image': images,
        'seq_lens': torch.tensor(seq_lens),
        'text': texts,
        'idx': torch.tensor(indices)
    }


class HTRModelAnalyzer:
    """Analyze and visualize performance of Kraken HTR models."""
    
    def __init__(self, model_path: str):
        """
        Initialize with a trained model path
        
        Args:
            model_path: Path to trained .mlmodel file
        """
        self.model_path = model_path
        self.nn = vgsl.TorchVGSLModel.load_model(model_path)
        self.recognizer = models.TorchSeqRecognizer(self.nn)
        
        # Extract and store metadata
        self.metadata = self.nn.user_metadata
        self.hyper_params = self.nn.hyper_params
        
        # Extract accuracy metrics from model if available
        self.accuracy_data = None
        self.metrics_data = None
        
        if 'accuracy' in self.metadata:
            self.accuracy_data = pd.DataFrame(self.metadata['accuracy'], 
                                             columns=['step', 'accuracy'])
        
        if 'metrics' in self.metadata:
            metrics_list = []
            for step, metrics_dict in self.metadata['metrics']:
                metrics_dict['step'] = step
                metrics_list.append(metrics_dict)
            self.metrics_data = pd.DataFrame(metrics_list)
        
        # Check model's expected input channels
        self.input_channels = self.nn.input[1]
        logger.info(f"Model expects {self.input_channels} input channels")
    
    def summary(self) -> Dict:
        """Return summary of model information"""
        return {
            'model_path': self.model_path,
            'spec': self.nn.spec,
            'model_type': self.nn.model_type,
            'alphabet_size': len(self.nn.codec),
            'completed_epochs': self.hyper_params.get('completed_epochs', 0),
            'best_val_accuracy': self.accuracy_data['accuracy'].max() if self.accuracy_data is not None else None,
            'final_val_accuracy': self.accuracy_data['accuracy'].iloc[-1] if self.accuracy_data is not None else None,
            'input_shape': f"{self.nn.input[0]}x{self.nn.input[1]}x{self.nn.input[2]}x{self.nn.input[3]}"
        }
    
    def plot_training_metrics(self) -> Figure:
        """Plot training and validation metrics"""
        if self.metrics_data is None:
            raise ValueError("No metrics data available in model")
        
        plt.figure(figsize=(14, 8))
        
        # Extract relevant metrics
        metrics = [col for col in self.metrics_data.columns 
                  if col != 'step' and not pd.isna(self.metrics_data[col]).all()]
        
        n_plots = len(metrics)
        cols = 2
        rows = (n_plots + 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                ax = axes[i]
                self.metrics_data.plot(x='step', y=metric, ax=ax)
                ax.set_title(f'{metric} vs Training Step')
                ax.set_ylabel(metric)
                ax.grid(True, linestyle='--', alpha=0.7)
                
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        return fig
    
    def evaluate_on_arrow_dataset(self, arrow_path, split='validation', batch_size=8, num_workers=4):
        """
        Evaluate model on Arrow binary dataset using Kraken's evaluation pipeline
        
        Args:
            arrow_path: Path to Arrow IPC dataset file
            split: Dataset split to use ('train', 'validation', or 'test')
            batch_size: Batch size for evaluation
            num_workers: Number of workers for dataloader
        """
        # Import Kraken evaluation components
        from torchmetrics.text import CharErrorRate, WordErrorRate
        from kraken.lib.dataset import (ArrowIPCRecognitionDataset, 
                                       collate_sequences, compute_confusions, global_align,
                                       ImageInputTransforms)
        from torch.utils.data import DataLoader
        
        # Check if in notebook environment for proper tqdm version
        try:
            # Try to get ipython instance - if successful, we're in a notebook
            get_ipython()
            from tqdm.notebook import tqdm
        except (NameError, ImportError):
            # Not in a notebook
            from tqdm import tqdm
        
        # Check if legacy_polygons is needed (based on model)
        legacy_polygons = getattr(self.nn, 'use_legacy_polygons', False)
        
        # Load dataset
        dataset_kwargs = {"split_filter": split}
        
        # Get model parameters
        batch, channels, height, width = self.nn.input
        
        # Create dataset transforms
        ts = ImageInputTransforms(
            batch, height, width, channels, (16, 0), False, False)
        
        # Initialize dataset and show progress dialog
        print(f"Loading dataset from {arrow_path} (split: {split})")
        
        try:
            ds = ArrowIPCRecognitionDataset(
                im_transforms=ts,
                **dataset_kwargs
            )
            
            # Add the file to the dataset
            ds.add(file=arrow_path)
            
            # Don't encode validation set
            ds.no_encode()
            print(f"Loaded {len(ds)} samples")
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            raise
        
        # Create data loader
        ds_loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_sequences
        )
        
        # Evaluation metrics
        test_cer = CharErrorRate()
        test_cer_case_insensitive = CharErrorRate()
        test_wer = WordErrorRate()
        
        # Process all batches
        algn_gt = []
        algn_pred = []
        chars = 0
        error = 0
        results = []
        
        # Show interactive progress display
        print(f"Evaluating model on {len(ds)} samples...")
        
        for batch in tqdm(ds_loader, desc=f"Evaluating {os.path.basename(self.model_path)}"):
            im = batch['image']
            text = batch['target']
            lens = batch['seq_lens']
            
            try:
                pred = self.recognizer.predict_string(im, lens)
                
                for x, y in zip(pred, text):
                    chars += len(y)
                    c, algn1, algn2 = global_align(y, x)
                    algn_gt.extend(algn1)
                    algn_pred.extend(algn2)
                    error += c
                    
                    test_cer.update(x, y)
                    test_cer_case_insensitive.update(x.lower(), y.lower())
                    test_wer.update(x, y)
                    
                    results.append({'prediction': x, 'ground_truth': y})
                    
            except Exception as e:
                logger.warning(f"Error processing batch: {e}")
        
        # Calculate metrics
        cer = test_cer.compute()
        cer_case_insensitive = test_cer_case_insensitive.compute()
        wer = test_wer.compute()
        
        # Compute confusions
        confusions, scripts, ins, dels, subs = compute_confusions(algn_gt, algn_pred)
        
        # Process error data
        error_counter = Counter()
        char_errors = []
        
        for i, (gt_char, pred_char) in enumerate(zip(algn_gt, algn_pred)):
            if gt_char != pred_char:
                if gt_char == '':
                    op = 'insert'
                    error_counter[f"∅ → {pred_char}"] += 1
                elif pred_char == '':
                    op = 'delete'
                    error_counter[f"{gt_char} → ∅"] += 1
                else:
                    op = 'replace'
                    error_counter[f"{gt_char} → {pred_char}"] += 1
                
                char_errors.append((gt_char, pred_char, op))
        
        # Create confusion matrix
        confusion_matrix = defaultdict(Counter)
        for gt, pred, op in char_errors:
            confusion_matrix[gt][pred] += 1
        
        # Print summary
        accuracy = 1.0 - cer
        print(f"Evaluation complete: CER={cer:.4f}, Accuracy={accuracy:.4f} ({chars} characters)")
        
        # CharErrorRate returns error rate, so accuracy = 1 - error_rate
        return {
            'results': pd.DataFrame(results),
            'char_error_rate': cer,  # This is already the error rate
            'accuracy': accuracy,   # Convert to accuracy
            'accuracy_case_insensitive': 1.0 - cer_case_insensitive,
            'word_accuracy': 1.0 - wer,
            'total_chars': chars,
            'errors': char_errors,
            'common_errors': error_counter,
            'confusion_matrix': confusion_matrix
        }
    
    def plot_confusion_matrix(self, confusion_matrix, top_n=20):
        """
        Plot confusion matrix of character predictions
        
        Args:
            confusion_matrix: Nested dict of ground truth -> prediction -> count
            top_n: How many top characters to include
        """
        # Convert confusion matrix to DataFrame
        chars = set()
        for gt, pred_counts in confusion_matrix.items():
            chars.add(gt)
            chars.update(pred_counts.keys())
        
        chars.discard('')  # Handle separately
        chars = sorted(list(chars))
        
        # Count character frequency to find most common
        char_counts = Counter()
        for gt, pred_counts in confusion_matrix.items():
            if gt != '':
                char_counts[gt] += sum(pred_counts.values())
        
        # Get top N characters by frequency
        top_chars = [c for c, _ in char_counts.most_common(top_n)]
        
        # Create confusion matrix for top chars
        cm_data = np.zeros((len(top_chars), len(top_chars)))
        
        for i, gt in enumerate(top_chars):
            for j, pred in enumerate(top_chars):
                cm_data[i, j] = confusion_matrix[gt][pred]
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_data, annot=True, fmt='g', 
                    xticklabels=top_chars, yticklabels=top_chars, 
                    cmap='viridis')
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.title('Character Confusion Matrix')
        return plt.gcf()

    def plot_error_distribution(self, error_data, top_n=20):
        """Plot distribution of most common errors"""
        common_errors = error_data['common_errors'].most_common(top_n)
        
        labels = [error for error, _ in common_errors]
        values = [count for _, count in common_errors]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(labels, values)
        
        # Add count and percentage labels
        total_errors = sum(error_data['common_errors'].values())
        for i, (count, bar) in enumerate(zip(values, bars)):
            percentage = 100 * count / total_errors
            plt.text(count + 0.1, bar.get_y() + bar.get_height()/2, 
                    f"{count} ({percentage:.1f}%)", 
                    va='center')
        
        plt.xlabel('Count')
        plt.title(f'Top {top_n} Most Common Character Errors')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_lengths_vs_errors(self, results_df):
        """Plot relationship between text length and error rate"""
        # Calculate lengths and errors
        results_df['gt_length'] = results_df['ground_truth'].apply(len)
        results_df['levenshtein_distance'] = [
            SequenceMatcher(None, gt, pred).get_opcodes() 
            for gt, pred in zip(results_df['ground_truth'], results_df['prediction'])
        ]
        results_df['error_count'] = results_df['levenshtein_distance'].apply(
            lambda opcodes: sum(max(i2-i1, j2-j1) for op, i1, i2, j1, j2 in opcodes 
                               if op != 'equal')
        )
        results_df['error_rate'] = results_df['error_count'] / results_df['gt_length']
        
        # Group by length buckets
        results_df['length_bucket'] = pd.cut(results_df['gt_length'], 
                                            bins=10, 
                                            precision=0)
        length_stats = results_df.groupby('length_bucket').agg(
            avg_error_rate=('error_rate', 'mean'),
            count=('error_rate', 'count')
        )
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot average error rate
        ax1.plot(range(len(length_stats)), length_stats['avg_error_rate'], 
                marker='o', linestyle='-', color='blue')
        ax1.set_xlabel('Text Length Bucket')
        ax1.set_ylabel('Average Error Rate', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add sample count on secondary y-axis
        ax2 = ax1.twinx()
        ax2.bar(range(len(length_stats)), length_stats['count'], 
               alpha=0.3, color='gray')
        ax2.set_ylabel('Sample Count', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        # Add bucket labels
        plt.xticks(range(len(length_stats)), 
                  [str(b) for b in length_stats.index], 
                  rotation=45)
        
        plt.title('Error Rate vs Text Length')
        plt.tight_layout()
        return fig


    def analyze_character_performance(self, results, top_n=20, min_occurrences=5):
        """
        Analyze recognition performance for specific characters
        
        Args:
            results: Evaluation results from evaluate_on_arrow_dataset
            top_n: How many top characters to analyze
            min_occurrences: Minimum number of occurrences for a character to be analyzed
            
        Returns:
            Dictionary with character performance data and matplotlib figure
        """
        from collections import defaultdict
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Extract aligned ground truth and predictions
        # From the evaluation results
        algn_gt = []
        algn_pred = []
        
        # Reconstruct alignments if not directly provided
        if 'alignments' in results:
            algn_gt = results['alignments']['gt']
            algn_pred = results['alignments']['pred']
        else:
            # Reconstruct from errors
            for gt, pred, _ in results['errors']:
                algn_gt.append(gt)
                algn_pred.append(pred)
        
        # Count character occurrences and errors
        char_counts = defaultdict(int)
        char_errors = defaultdict(int)
        
        # Process all aligned characters
        for gt_char, pred_char in zip(algn_gt, algn_pred):
            if gt_char != '':  # Skip insertion errors
                char_counts[gt_char] += 1
                if gt_char != pred_char:
                    char_errors[gt_char] += 1
        
        # Calculate error rates and filter by minimum occurrences
        char_stats = {}
        for char, count in char_counts.items():
            if count >= min_occurrences:
                error_count = char_errors[char]
                char_stats[char] = {
                    'occurrences': count,
                    'errors': error_count,
                    'error_rate': error_count / count if count > 0 else 0,
                    'accuracy': 1 - (error_count / count) if count > 0 else 0
                }
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(char_stats, orient='index')
        
        # Sort by occurrences (descending), and then by error rate (descending)
        df = df.sort_values(['occurrences', 'error_rate'], ascending=[False, False])
        
        # Keep top N most frequent characters
        df = df.head(top_n)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Character accuracy plot
        bars = ax1.bar(range(len(df)), df['accuracy'], color='green')
        
        # Add character labels
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels([f"'{c}'" for c in df.index], rotation=45, ha='right')
        
        # Add count and percentage labels
        for i, (accuracy, count) in enumerate(zip(df['accuracy'], df['occurrences'])):
            label = f"{accuracy:.1%} ({count})"
            ax1.annotate(label, 
                        xy=(i, accuracy), 
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
        
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel('Character Accuracy')
        ax1.set_title('Character Recognition Accuracy by Character')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Occurrences plot (secondary)
        ax2.bar(range(len(df)), df['occurrences'], color='navy', alpha=0.6)
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels([f"'{c}'" for c in df.index], rotation=45, ha='right')
        ax2.set_ylabel('Number of Occurrences')
        ax2.set_title('Character Frequency in Test Set')
        
        plt.tight_layout()
        
        return {
            'char_performance_df': df,
            'plot': fig
        }

    def visualize_samples(self, arrow_path, split='test', num_samples=5, 
                       filter_type='random', filter_value=None, batch_size=8, 
                       num_workers=4, figsize=(15, 3), fontsize=10):
        """
        Visualize sample images with predictions and ground truth
        
        Args:
            arrow_path: Path to Arrow IPC dataset file
            split: Dataset split to use
            num_samples: Number of samples to visualize
            filter_type: How to select samples ('random', 'error', 'char', 'worst')
                        - 'random': Random samples
                        - 'error': Samples with specific error type
                        - 'char': Samples containing specific character
                        - 'worst': Samples with highest error rates
            filter_value: Value to filter by (specific error, specific character)
            batch_size: Batch size for evaluation
            num_workers: Number of workers for dataloader
            figsize: Base figure size (will be multiplied by number of samples)
            fontsize: Font size for text
            
        Returns:
            Dictionary with samples data and matplotlib figure
        """
        import random
        import matplotlib.pyplot as plt
        import numpy as np
        from torch.utils.data import DataLoader
        from difflib import SequenceMatcher
        from kraken.lib.dataset import (ArrowIPCRecognitionDataset, collate_sequences, 
                                       ImageInputTransforms, global_align)
        
        # First, get a small set of samples to choose from
        # Set up dataset
        dataset_kwargs = {"split_filter": split}
        batch, channels, height, width = self.nn.input
        
        # Create dataset transforms
        ts = ImageInputTransforms(
            batch, height, width, channels, (16, 0), False, False)
        
        # Initialize dataset
        ds = ArrowIPCRecognitionDataset(
            im_transforms=ts,
            **dataset_kwargs
        )
        
        # Add the file to the dataset
        ds.add(file=arrow_path)
        
        # Don't encode validation set
        ds.no_encode()
        
        # Create a data loader with a batch size of 1 to get individual samples
        sample_loader = DataLoader(
            ds,
            batch_size=1,
            num_workers=num_workers,
            collate_fn=collate_sequences,
            shuffle=(filter_type == 'random')
        )
        
        # Collect samples
        all_samples = []
        max_to_collect = min(len(ds), 100)  # Limit to 100 samples for memory efficiency
        
        print(f"Collecting samples from {arrow_path} ({split} split)...")
        
        for i, batch in enumerate(sample_loader):
            if i >= max_to_collect:
                break
                
            # Get sample
            img = batch['image'][0]  # Single sample
            text = batch['target'][0]  # Ground truth
            seq_len = batch['seq_lens'][0]
            
            # Get prediction
            pred = self.recognizer.predict_string(batch['image'], batch['seq_lens'])[0]
            
            # Calculate character error rate
            char_changes, aligned_gt, aligned_pred = global_align(text, pred)
            cer = char_changes / len(text) if len(text) > 0 else 0
            
            # Store sample data
            sample_data = {
                'image': img.detach().cpu().numpy(),  # Store as numpy array
                'ground_truth': text,
                'prediction': pred,
                'cer': cer,
                'aligned_gt': aligned_gt,
                'aligned_pred': aligned_pred
            }
            
            all_samples.append(sample_data)
        
        # Filter samples based on filter_type
        filtered_samples = []
        
        if filter_type == 'random':
            # Random samples
            filtered_samples = random.sample(all_samples, min(num_samples, len(all_samples)))
            
        elif filter_type == 'error':
            # Samples with specific error type
            if filter_value is None:
                print("Warning: No error type specified. Using random samples.")
                filtered_samples = random.sample(all_samples, min(num_samples, len(all_samples)))
            else:
                for sample in all_samples:
                    for gt_char, pred_char in zip(sample['aligned_gt'], sample['aligned_pred']):
                        error_str = f"{gt_char} → {pred_char}"
                        if gt_char != pred_char and filter_value in error_str:
                            filtered_samples.append(sample)
                            break
                            
                    if len(filtered_samples) >= num_samples:
                        break
                        
        elif filter_type == 'char':
            # Samples containing specific character
            if filter_value is None:
                print("Warning: No character specified. Using random samples.")
                filtered_samples = random.sample(all_samples, min(num_samples, len(all_samples)))
            else:
                for sample in all_samples:
                    if filter_value in sample['ground_truth']:
                        filtered_samples.append(sample)
                        
                    if len(filtered_samples) >= num_samples:
                        break
                        
        elif filter_type == 'worst':
            # Samples with highest error rates
            sorted_samples = sorted(all_samples, key=lambda x: x['cer'], reverse=True)
            filtered_samples = sorted_samples[:num_samples]
        
        else:
            print(f"Warning: Unknown filter type '{filter_type}'. Using random samples.")
            filtered_samples = random.sample(all_samples, min(num_samples, len(all_samples)))
        
        # Ensure we have at least some samples
        if not filtered_samples:
            print(f"No samples found matching filter criteria. Using random samples.")
            filtered_samples = random.sample(all_samples, min(num_samples, len(all_samples)))
        
        # Limit to requested number of samples
        filtered_samples = filtered_samples[:num_samples]
        
        # Create visualization
        fig_width, fig_height = figsize
        fig, axes = plt.subplots(len(filtered_samples), 1, 
                                figsize=(fig_width, fig_height * len(filtered_samples)),
                                gridspec_kw={'hspace': 0.5})
        
        # Handle single sample case
        if len(filtered_samples) == 1:
            axes = [axes]
        
        for i, (sample, ax) in enumerate(zip(filtered_samples, axes)):
            # Get image data
            img_data = sample['image'][0]  # Take the first channel (grayscale)
            
            # Display image
            ax.imshow(img_data, cmap='gray')
            ax.axis('off')
            
            # Add ground truth and prediction as title
            gt = sample['ground_truth']
            pred = sample['prediction']
            cer = sample['cer']
            
            # Create colored text to highlight differences
            colored_text = []
            for gt_char, pred_char in zip(sample['aligned_gt'], sample['aligned_pred']):
                if gt_char == pred_char:
                    colored_text.append(pred_char)
                elif gt_char == '':
                    # Insertion error - highlighted in red
                    colored_text.append(f"\033[91m{pred_char}\033[0m")
                elif pred_char == '':
                    # Deletion error - show a red underscore
                    colored_text.append(f"\033[91m_\033[0m")
                else:
                    # Substitution error - highlighted in red
                    colored_text.append(f"\033[91m{pred_char}\033[0m")
            
            # Join the colored text
            colored_prediction = ''.join(colored_text)
            
            title = f"Sample {i+1} (CER: {cer:.2f})\n"
            title += f"GT: {gt}\n"
            title += f"Pred: {colored_prediction}"
            
            ax.set_title(title, fontsize=fontsize, loc='left')
        
        plt.tight_layout()
        
        return {
            'samples': filtered_samples,
            'plot': fig
        }
class ModelComparator:
    """Compare multiple Kraken HTR models."""
    
    def __init__(self, model_paths: List[str], model_names: Optional[List[str]] = None):
        """
        Initialize with multiple model paths
        
        Args:
            model_paths: List of paths to trained models
            model_names: Optional friendly names for the models (defaults to filenames)
        """
        self.model_paths = model_paths
        
        # Use filenames as default model names
        if model_names is None:
            self.model_names = [os.path.basename(path) for path in model_paths]
        else:
            if len(model_names) != len(model_paths):
                raise ValueError("Length of model_names must match length of model_paths")
            self.model_names = model_names
            
        # Load all models
        self.analyzers = []
        for path in model_paths:
            try:
                self.analyzers.append(HTRModelAnalyzer(path))
            except Exception as e:
                logger.error(f"Failed to load model {path}: {e}")
        
        if not self.analyzers:
            raise ValueError("No models could be loaded successfully")
            
    def compare_training_metrics(self):
        """Compare training metrics across models"""
        # Extract and compile metrics
        all_metrics = []
        
        for name, analyzer in zip(self.model_names, self.analyzers):
            if analyzer.metrics_data is not None:
                # Add model name column
                metrics = analyzer.metrics_data.copy()
                metrics['model'] = name
                all_metrics.append(metrics)
        
        if not all_metrics:
            raise ValueError("No models have metrics data available")
            
        # Combine all metrics
        combined = pd.concat(all_metrics, ignore_index=True)
        
        # Find common metrics across all models
        common_metrics = set.intersection(*[set(df.columns) for df in all_metrics])
        common_metrics = [m for m in common_metrics if m not in ['step', 'model']]
        
        # Plot common metrics
        n_metrics = len(common_metrics)
        cols = 2
        rows = (n_metrics + 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
        
        for i, metric in enumerate(common_metrics):
            if i < len(axes):
                ax = axes[i]
                for name in self.model_names:
                    subset = combined[combined['model'] == name]
                    if not subset.empty and metric in subset.columns:
                        subset.plot(x='step', y=metric, ax=ax, label=name)
                
                ax.set_title(f'{metric} vs Training Step')
                ax.set_ylabel(metric)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        return fig
    
    def compare_summary(self):
        """Compare summary statistics across models"""
        # Extract summaries
        summaries = []
        for name, analyzer in zip(self.model_names, self.analyzers):
            summary = analyzer.summary()
            summary['model'] = name
            summaries.append(summary)
            
        # Combine into DataFrame
        df = pd.DataFrame(summaries)
        
        # Reorder columns to put model first
        cols = ['model'] + [col for col in df.columns if col != 'model']
        df = df[cols]
        
        return df
        
    def compare_evaluation_on_arrow(self, arrow_path, split='test'):
        """
        Compare models on same Arrow dataset split using Kraken's evaluation
        
        Args:
            arrow_path: Path to Arrow IPC dataset file
            split: Dataset split to use ('train', 'validation', or 'test')
        """
        results = {}
        
        # Print header for comparison
        print(f"Comparing {len(self.model_names)} models on {arrow_path} ({split} split)")
        print("-" * 50)
        
        for name, analyzer in zip(self.model_names, self.analyzers):
            print(f"\nEvaluating model: {name}")
            try:
                eval_result = analyzer.evaluate_on_arrow_dataset(arrow_path, split)
                results[name] = eval_result
                print(f"  - Accuracy: {eval_result['accuracy']:.4f}")
                print(f"  - CER: {eval_result['char_error_rate']:.4f}")
                print(f"  - Word Accuracy: {eval_result['word_accuracy']:.4f}")
            except Exception as e:
                logger.error(f"Error evaluating model {name}: {e}")
                print(f"  - ERROR: {str(e)}")
        
        if not results:
            print("No evaluations completed successfully!")
            raise ValueError("None of the models could be evaluated successfully")
        
        # Generate comparison data
        comparison = {
            'model': [],
            'char_error_rate': [],
            'accuracy': [],
            'word_accuracy': [],
            'total_chars': []
        }
        
        for name, result in results.items():
            comparison['model'].append(name)
            comparison['char_error_rate'].append(result['char_error_rate'])
            comparison['accuracy'].append(result['accuracy'])
            comparison['word_accuracy'].append(result['word_accuracy'])
            comparison['total_chars'].append(result['total_chars'])
        
        # Create DataFrame
        df = pd.DataFrame(comparison)
        
        # Print summary
        print("\nSummary:")
        print("-" * 50)
        for i, row in df.sort_values('accuracy', ascending=False).iterrows():
            print(f"{row['model']}: Accuracy={row['accuracy']:.4f}, CER={row['char_error_rate']:.4f}")
        
        # Create visualization
        print("\nGenerating visualization...")
        fig, ax = plt.subplots(figsize=(10, 6))
        df_sorted = df.sort_values('accuracy', ascending=False)
        
        x = range(len(df_sorted))
        width = 0.35
        
        # Plot accuracy and error rates
        ax.bar([i - width/2 for i in x], df_sorted['accuracy'], width, label='Accuracy', color='green')
        ax.bar([i + width/2 for i in x], df_sorted['char_error_rate'], width, label='CER', color='red')
        
        # Add value labels
        for i, (acc, err) in enumerate(zip(df_sorted['accuracy'], df_sorted['char_error_rate'])):
            ax.text(i - width/2, acc + 0.01, f"{acc:.3f}", ha='center')
            ax.text(i + width/2, err + 0.01, f"{err:.3f}", ha='center')
        
        ax.set_ylabel('Rate')
        ax.set_title('Model Comparison: Accuracy vs Error Rate')
        ax.set_xticks(x)
        ax.set_xticklabels(df_sorted['model'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        print("Comparison complete!")
        
        return {
            'comparison_df': df,
            'plot': fig,
            'detailed_results': results
        }

    def compare_error_distribution(self, evaluation_results, top_n=10):
        """
        Compare error distribution across models
        
        Args:
            evaluation_results: Results from compare_evaluation
            top_n: Number of top errors to show
        """
        # Extract common errors across all models
        all_errors = Counter()
        model_errors = {}
        
        print(f"Analyzing error distributions across {len(evaluation_results['detailed_results'])} models...")
        
        for model, result in evaluation_results['detailed_results'].items():
            model_errors[model] = result['common_errors']
            all_errors.update(result['common_errors'])
            print(f"  - {model}: {len(result['errors'])} total errors, {len(result['common_errors'])} unique error types")
        
        # Get top N most common errors across all models
        top_errors = [error for error, _ in all_errors.most_common(top_n)]
        
        if not top_errors:
            print("No common errors found across models.")
            return {
                'error_comparison': pd.DataFrame(),
                'plot': plt.figure()
            }
        
        # Create comparison DataFrame
        comparison = []
        
        for error in top_errors:
            row = {'error': error}
            for model in self.model_names:
                if model in model_errors:
                    row[model] = model_errors[model][error]
                else:
                    row[model] = 0
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        # Print top errors
        print(f"\nTop {top_n} most common errors across all models:")
        for i, error in enumerate(top_errors[:5], 1):  # Show top 5 in console
            print(f"  {i}. {error}: {all_errors[error]} occurrences")
        if len(top_errors) > 5:
            print(f"  ... plus {len(top_errors)-5} more (see visualization)")
        
        # Create comparison plot
        print("\nGenerating error distribution visualization...")
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar chart
        x = range(len(df))
        width = 0.8 / len(self.model_names)
        
        for i, model in enumerate(self.model_names):
            if model in model_errors:
                counts = df[model].values
                positions = [p + width * (i - len(self.model_names)/2 + 0.5) for p in x]
                plt.bar(positions, counts, width, label=model)
        
        plt.ylabel('Count')
        plt.title(f'Top {top_n} Most Common Errors by Model')
        plt.xticks(x, df['error'], rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        print("Error distribution analysis complete!")
        
        return {
            'error_comparison': df,
            'plot': plt.gcf()
        }