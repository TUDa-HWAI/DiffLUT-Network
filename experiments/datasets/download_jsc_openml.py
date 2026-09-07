import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os
import time
import pandas as pd

from .array_dataset import TabularThermometer

def download_and_process(num_bits=200):
    """
    Download JSC dataset from OpenML and process with DistributiveThermometer encoding.
    
    1. Fetch dataset 42468 (hls4ml_lhc_jets_hlf) from OpenML
    2. 80/20 train/test split
    3. Fit DistributiveThermometer on training data
    4. Binarize both train and test data
    5. Save processed data as .npy files
    
    Args:
        num_bits: Number of thermometer bits per feature (default: 200)
    """
    dataset_name = 'hls4ml_lhc_jets_hlf'
    print(f"Fetching dataset {dataset_name} using sklearn.datasets.fetch_openml...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # fetch_openml handles the download and parsing
            dataset = fetch_openml(dataset_name, version=1, as_frame=True)
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Waiting 10 seconds before retry...")
                time.sleep(10)
            else:
                print("Failed to download from OpenML after multiple attempts.")
                raise e

    X = dataset.data
    y = dataset.target
    
    # 16 High Level Features (HLF)
    feature_labels = X.columns.tolist()
    print(f"Features: {feature_labels}")
    print(f"Feature count: {len(feature_labels)}")
    
    # Combine features and targets to perform deduplication and alignment safely
    df = pd.DataFrame(X)
    df['target'] = y
    
    # 1. Deduplicate strictly on the features + target
    df = df.drop_duplicates(subset=feature_labels + ['target'])
    print(f"Rows after strict features+labels deduplication: {len(df)}")
    
    # 2. Fix the target alignment to match standard class indexing (g->0, q->1, w->2, z->3, t->4)
    # Map both with and without 'j_' prefix for maximum robustness
    label_to_index = {
        'g': 0, 'j_g': 0,
        'q': 1, 'j_q': 1,
        'w': 2, 'j_w': 2,
        'z': 3, 'j_z': 3,
        't': 4, 'j_t': 4
    }
    
    # Let's map target to 0-4 indices
    labels = np.array(df['target'].map(label_to_index).values)
    
    # Drop rows that failed to map
    valid_mask = ~pd.isna(labels)
    df = df[valid_mask]
    labels = labels[valid_mask].astype(np.int64)
    
    features = df[feature_labels].astype(np.float32).values
    num_output = 5
    print(f"Classes map: {label_to_index}")
    print(f"Number of classes: {num_output}")
    
    # Split data (80% train, 20% test) — following LogicNets
    print("Splitting data (80/20)...")
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, train_size=0.8, random_state=42
    )
    
    # Binarize data with Distributive Thermometer
    print(f"Applying Distributive Thermometer encoding with {num_bits} bits per feature...")
    thermometer = TabularThermometer(num_bits).fit(x_train)
    
    x_train_bin = thermometer.binarize(x_train).reshape(len(x_train), -1)
    x_test_bin = thermometer.binarize(x_test).reshape(len(x_test), -1)
    
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    
    # Output directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(curr_dir, "jsc-openml")
    os.makedirs(out_dir, exist_ok=True)
    
    # Save as .npy
    print(f"Saving to {out_dir}...")
    np.save(os.path.join(out_dir, "train_data.npy"), x_train_bin)
    np.save(os.path.join(out_dir, "train_labels.npy"), y_train)
    np.save(os.path.join(out_dir, "test_data.npy"), x_test_bin)
    np.save(os.path.join(out_dir, "test_labels.npy"), y_test)
    
    # Also save the thermometer thresholds for reproducibility
    np.save(os.path.join(out_dir, "thermometer_thresholds.npy"), thermometer.thresholds)
    
    # Save metadata
    metadata = {
        'num_features': features.shape[1],
        'num_bits': num_bits,
        'input_dim': x_train_bin.shape[1],
        'num_classes': int(num_output),
        'label_names': ['Gluon', 'Quark', 'W boson', 'Z boson', 'Top Quark'],
        'train_samples': x_train_bin.shape[0],
        'test_samples': x_test_bin.shape[0],
    }
    import json
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("Download and processing complete.")
    print(f"Train samples: {x_train_bin.shape[0]}, Test samples: {x_test_bin.shape[0]}")
    print(f"Original feature count: {features.shape[1]}")
    print(f"Binarized input dim: {x_train_bin.shape[1]} ({features.shape[1]} features × {num_bits} bits)")
    print(f"Class count: {num_output}")
    
    return metadata
