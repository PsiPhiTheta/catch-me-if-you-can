# catch-me-if-you-can
An ML project exploring alternative approaches to signature verification.

## Setup
Using Conda, you can create and activate the Python environment like so:

```bash
conda env create -f environment.yml
conda activate cmiyc-dev
```

## Data
The data used in this project comes from the [dataset](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011)) used in the ICDAR 2009 Signature Verification Competition (SigComp2011). You should place the directory structure in `cmiyc/data/raw/`

The testing data can be preprocessed simply by executing the `preprocessing.py` script:
```bash
cd cmiyc
python preprocessing.py
```

This will pre-process all the dutch offline signatures from the test set and save them as two Numpy arrays, one for the genuine and one for the forgeries, in `data/clean/`. Note that estimated file size for the .pkl when processing all files is **4.8GB**.

