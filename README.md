# Multi-omics clustering
Project 14, Bioinformatics course, Politecnico di Torino

This is a comprehensive user manual for the codes we produced for our project.

## DATASETS RETRIEVAL
(n.b.: the synthetic 500-samples dataset has been downloaded from the Portale della didattica)

- **Dataset retrieval (lung).ipynb**: retrieve the Lung tumors samples dataset described in the report; correct the batch effect for the gene CNV omic
- **Dataset retrieval (kidney).ipynb**: retrieve the Kidney tumors samples dataset described in the report

## OUR METHOD
- **Multi-omics clustering (synthetic).ipynb**: apply our integration and clustering method to the Synthetic samples dataset, with added Gaussian salt & pepper noise
- **Multi-omics clustering (synthetic no noise).ipynb**: apply our integration and clustering method to the Synthetic samples dataset (without added noise)
- **Multi-omics clustering (lung).ipynb**: apply our integration and clustering method to the Lung tumors samples dataset
- **Multi-omics clustering (kidney).ipynb**: apply our integration and clustering method to the Kidney tumors samples dataset
- **Visualizations.ipynb**: visualizations of the results obtained with our methods on the three datasets; in it, the autoencoders trained for each omic are automatically retrieved from our Google Drive project folder

## OTHER INTEGRATION AND CLUSTERING METHODS
- **Method MFA.ipynb**: MFA + K-means applied to the three datasets
- **Method SNF.ipynb**: SNF + spectral clustering applied to the three datasets
- **Method iClusterPlus.ipynb**: iCluster+ + K-means applied to the three datasets

## MISCELLANEOUS
- **utils.py**: some utility functions (for visualization of datasets and AE training) used in all our notebooks; this file is downloaded from our Google Drive project folder with gdown at the beginning of every notebook
- **Find protein coding genes.ipynb**: to create a text file with the indices of the features of the mRNA omic corresponding to protein coding genes (used for the pre-processing of Lung and Kidney datasets)
- **Find common meth probes illumina 27k 450k.ipynb**: to create a text file with the indices of the features of the meth omic corresponding to the methylation probes in common between the illumina methylation 27k and 450k machines (used for the pre-processing of meth of Lung and Kidney datasets)

## Other info
All the manifest & JSON files, datasets, saved trained autoencoders, the files returned by "Find protein coding genes.ipynb" and "Find common meth probes illumina 27k 450k.ipynb", and "utils.py" are stored in our Google Drive project folder, and are automatically retrieved inside the notebooks with the use of `gdown` and `wget` console commands.

**The coding and testing environment has been Google Colab**, so the libraries used and their versions are the ones pre-installed in Colab or the ones one can get with `pip install <package>`. For reference, we report the complete list of these libraries (along with their versions) here:

- `gdown == 3.6.4`
- `pandas == 1.1.5`
- `scikit-learn == 0.22.2.post1`
- `matplotlib == 3.2.2`
- `seaborn == 0.11.1`
- `numpy == 1.19.5`
- `scipy == 1.4.1`
- `tensorflow == 2.5.0`
- `prince == 0.7.1` (used in MFA)
- `snfpy == 0.2.2` (used in SNF)
- `rpy2 == 3.4.5` (used in iCluster+)
- `iClusterPlus == 1.28.0` (R Bioconductor library; used in iCluster+)
- `BiocManager == 3.8` (R library; used in iCluster+)
- `os`
- `itertools`
- `warnings`
- `utils` (our own utils.py library)
