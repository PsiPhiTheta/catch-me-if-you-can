# catch-me-if-you-can
An ML project exploring alternative approaches to signature verification.

## Setup
Using Conda, you can create and activate the Python environment like so:
```bash
conda env create -f environment.yml
conda activate cmifc-dev
```

## Data
The data used in this project comes from the dataset used in the CDAR 2011 Signature Verification Competition (SigComp2011).

The testing data can be preprocessed simply by executing the `preprocessing.py` script:
```bash
cd cmiyc
python preprocessing
```

This will read all files from `data/raw/trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Forgeries` and `data/raw/trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Genuine` and save two Numpy arrays in `data/raw/`. 