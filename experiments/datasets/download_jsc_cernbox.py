import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import urllib.request
import json
import h5py

from .array_dataset import TabularThermometer

curr_dir = os.path.dirname(os.path.abspath(__file__))

def download_and_process(num_bits=500):
    out_dir = os.path.join(curr_dir, "jsc-cernbox")
    os.makedirs(out_dir, exist_ok=True)
    
    raw_file_path = os.path.join(out_dir, "processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z")
    
    # 1. Download the raw file if it is not present.
    if not os.path.exists(raw_file_path):
        url = "https://cernbox.cern.ch/s/jvFd5MoWhGs1l5v/download"
        print(f"Downloading dataset from CERNBox: {url} -> {raw_file_path}")
        urllib.request.urlretrieve(url, raw_file_path)
        print("Download complete.")
    else:
        print(f"Raw dataset file already exists at {raw_file_path}.")

    # 16 High Level Features (HLF) with MMDT grooming
    feature_labels = [
        'j_zlogz', 'j_c1_b0_mmdt', 'j_c1_b1_mmdt', 'j_c1_b2_mmdt',
        'j_c2_b1_mmdt', 'j_c2_b2_mmdt', 'j_d2_b1_mmdt', 'j_d2_b2_mmdt',
        'j_d2_a1_b1_mmdt', 'j_d2_a1_b2_mmdt', 'j_m2_b1_mmdt', 'j_m2_b2_mmdt',
        'j_n2_b1_mmdt', 'j_n2_b2_mmdt', 'j_mass_mmdt', 'j_multiplicity'
    ]
    # 5 target classes (One-Hot encoded)
    output_labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
    required_columns = feature_labels + output_labels

    # 2. Open HDF5 and read only required columns (prevents OOM)
    print("Reading HDF5 data columns...")
    data_dict = {}
    with h5py.File(raw_file_path, 'r') as h5py_file:
        keys = list(h5py_file.keys())
        print(f"HDF5 table keys: {keys}")
        if "t_allpar_new" not in keys:
            raise KeyError(f"'t_allpar_new' table not found in HDF5 file. Keys: {keys}")
        dataset = h5py_file["t_allpar_new"]
        for col in required_columns:
            if col not in dataset.dtype.names:
                raise ValueError(f"Required column '{col}' is missing from the dataset.")
            print(f"  loading field: {col}...")
            data_dict[col] = dataset[col]

    # Convert structured numpy array to Pandas DataFrame
    print("Converting to Pandas DataFrame...")
    dataset_df = pd.DataFrame(data_dict)
    print(f"Total rows: {len(dataset_df)}")

    # 1. Filter out undefined signals (where sum of 5 targets is 0, representing j_undef)
    # Only keep rows where exactly one class is 1
    valid_mask = dataset_df[output_labels].sum(axis=1) == 1
    dataset_df = dataset_df[valid_mask]
    print(f"Rows after filtering undefined signals: {len(dataset_df)}")

    # 2. Drop duplicates strictly based on the 21 columns of features & labels we use
    dataset_df = dataset_df.drop_duplicates(subset=feature_labels + output_labels)
    print(f"Rows after strict feature+label deduplication: {len(dataset_df)}")

    # Extract features and labels
    features_df = dataset_df[feature_labels]
    outputs_df = dataset_df[output_labels]

    X = features_df.values.astype(np.float32)
    y_onehot = outputs_df.values.astype(np.float32)
    # Convert one-hot vectors to class index (0 to 4)
    y = np.argmax(y_onehot, axis=1)

    print(f"Features shape: {X.shape}")
    print(f"Class labels distribution: {np.bincount(y)}")

    # 3. Train/Test split (80% train, 20% test)
    print("Splitting data (80/20 train/test)...")
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )

    # 4. Fit Distributive Thermometer (no StandardScaler needed)
    print(f"Fitting DistributiveThermometer with {num_bits} bits per feature on training data...")
    thermometer = TabularThermometer(num_bits).fit(x_train)

    print("Binarizing train features...")
    num_train = x_train.shape[0]
    num_features = x_train.shape[1]
    x_train_bin = np.zeros((num_train, num_features * num_bits), dtype=np.uint8)
    for f in range(num_features):
        print(f"  binarizing feature {f+1}/{num_features}...")
        col = x_train[:, f:f+1] # (num_train, 1)
        # Directly assign boolean comparison to the preallocated uint8 array to optimize speed/memory
        x_train_bin[:, f*num_bits : (f+1)*num_bits] = col >= thermometer.thresholds[f]

    print("Binarizing test features...")
    num_test = x_test.shape[0]
    x_test_bin = np.zeros((num_test, num_features * num_bits), dtype=np.uint8)
    for f in range(num_features):
        print(f"  binarizing feature {f+1}/{num_features}...")
        col = x_test[:, f:f+1] # (num_test, 1)
        # Directly assign boolean comparison to the preallocated uint8 array to optimize speed/memory
        x_test_bin[:, f*num_bits : (f+1)*num_bits] = col >= thermometer.thresholds[f]

    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    # 5. Save outputs as .npy files
    print(f"Saving processed datasets to {out_dir}...")
    np.save(os.path.join(out_dir, "train_data.npy"), x_train_bin)
    np.save(os.path.join(out_dir, "train_labels.npy"), y_train)
    np.save(os.path.join(out_dir, "test_data.npy"), x_test_bin)
    np.save(os.path.join(out_dir, "test_labels.npy"), y_test)
    np.save(os.path.join(out_dir, "thermometer_thresholds.npy"), thermometer.thresholds)

    # Save metadata.json
    metadata = {
        'num_features': X.shape[1],
        'num_bits': num_bits,
        'input_dim': x_train_bin.shape[1],
        'num_classes': 5,
        'label_names': ['Gluon', 'Quark', 'W boson', 'Z boson', 'Top Quark'],
        'train_samples': x_train_bin.shape[0],
        'test_samples': x_test_bin.shape[0]
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Preprocessing completed successfully!")
    print(f"Binarized Input shape: {x_train_bin.shape}")
    print(f"Binarized Test shape: {x_test_bin.shape}")
