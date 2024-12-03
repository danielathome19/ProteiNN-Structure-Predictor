# About
ProteiNN is a Transformer network trained to predict end-to-end single-sequence protein structure by amino acid sequences. To find out more, check out the provided research paper:

* "Deep Learning for Protein Structure Prediction: Advancements in Structural Bioinformatics" (DOI: [10.1101/2023.04.26.538026](https://doi.org/10.1101/2023.04.26.538026))
* Also contained in the "PaperAndPresentation" folder is the research paper.

# Usage
Run main.py to choose from either "train", "predict", or "metrics" modes; train will retrain the model and predict will provide users the option to enter an amino acid sequence to predict the structure of (which will be output as a PDB file). Metrics mode will load the trained model and report its classification metrics (e.g., precision, recall, F1 score, etc.) on the test dataset.

**NOTE:** if you have any issues with the code not working as is, it may be due to the version of the `sidechainnet_casp12_30.pkl` file that the sidechainnet library downloads. Please contact me if you would like a copy of the version that I used to train the model.

# Bugs/Features
Bugs are tracked using the GitHub Issue Tracker.

Please use the issue tracker for the following purpose:
  * To raise a bug request; do include specific details and label it appropriately.
  * To suggest any improvements in existing features.
  * To suggest new features or structures or applications.

# License
The code is licensed under Apache License 2.0.

# Citation
If you use this code for your research, please cite this project:
```bibtex
@software{Szelogowski_ProteiNN-Structure-Predictor_2023,
 author = {Szelogowski, Daniel},
 doi = {10.1101/2023.04.26.538026},
 month = {April},
 title = {{ProteiNN-Structure-Predictor}},
 license = {Apache-2.0},
 url = {https://github.com/danielathome19/ProteiNN-Structure-Predictor},
 version = {1.0.0},
 year = {2023}
}
```

