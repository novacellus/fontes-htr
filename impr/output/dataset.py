# Function to analyze arrow datasets
import io
import json
import random
import pyarrow as pa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def explore_arrow_dataset(arrow_file_path, num_samples=3, mode='all'):
    """
    Explore and visualize samples from a Kraken Arrow dataset
    
    Parameters:
    - arrow_file_path: Path to the Arrow IPC file
    - num_samples: Number of samples to display from each split
    - mode: Which split to sample from ('train', 'validation', 'test', or 'all')
    """
    # Open the Arrow file
    with pa.memory_map(arrow_file_path, 'rb') as source:
        # Read the file
        arrow_file = pa.ipc.open_file(source)
        table = arrow_file.read_all()
        
        # Extract metadata
        metadata = table.schema.metadata
        if b'lines' in metadata:
            print("Dataset Metadata:")
            metadata_dict = json.loads(metadata[b'lines'])
            for key, value in metadata_dict.items():
                if key != 'alphabet':  # Skip printing the full alphabet
                    print(f"- {key}: {value}")
            
            # Print alphabet size
            if 'alphabet' in metadata_dict:
                print(f"- Alphabet size: {len(metadata_dict['alphabet'])} unique characters")
        
        # Print basic dataset info
        print(f"\nTotal samples: {len(table)}")
        
        # Function to get samples from a specific split
        def get_split_samples(split_name, n_samples):
            if split_name == 'all':
                indices = list(range(len(table)))
                samples = random.sample(indices, min(n_samples, len(indices)))
                return [(idx, table['lines'][idx].as_py()) for idx in samples]
            else:
                # Get indices where the split is True
                split_mask = table[split_name].to_numpy()
                indices = np.where(split_mask)[0]
                if len(indices) == 0:
                    print(f"No samples found in {split_name} split")
                    return []
                samples = random.sample(list(indices), min(n_samples, len(indices)))
                return [(idx, table['lines'][idx].as_py()) for idx in samples]
        
        # Get samples based on the mode
        if mode == 'all':
            splits = ['train', 'validation', 'test']
        else:
            splits = [mode]
        
        # Display samples from each requested split
        for split in splits:
            print(f"\n--- {split.upper()} SPLIT SAMPLES ---")
            samples = get_split_samples(split, num_samples)
            
            if not samples:
                continue
                
            # Display the samples
            for i, (idx, sample) in enumerate(samples):
                # Extract the image and text
                image_bytes = sample['im']
                text = sample['text']
                
                # Display
                print(f"Sample {i+1} (Index {idx}):")
                print(f"Text: \"{text}\"")
                
                # Convert bytes to image and display
                image = Image.open(io.BytesIO(image_bytes))
                plt.figure(figsize=(10, 3))
                plt.imshow(image, cmap='gray' if image.mode == 'L' else None)
                plt.title(f"Sample {i+1}: {text}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()

# Usage
