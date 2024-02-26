# Basis module for segemntation, regression and classification tasks for MRI images

This repository is the basis moule for multiple of my projects and is used to load the data, run the defined experiments, for example for hyperparameter tuning, train multiple different networks and evaluate them.

If you use our code in your work please cite the following paper:

Albert, S.; Wichtmann, B.D.; Zhao, W.; Maurer, A.; Hesser, J.; Attenberger, U.I.; Schad, L.R.; Zöllner, F.G. Comparison of Image Normalization Methods for Multi-Site Deep Learning. Appl. Sci. 2023, 13, 8923. https://doi.org/10.3390/app13158923

More details can be found in my dissertation

Albert, Steffen. Prediction of treatment response and outcome in locally advanced rectal cancer using radiomics. Diss. 2023. https://doi.org/10.11588/heidok.00034188

## Getting Started

To do an experiment, you have to import the class from the experiments module. Test data can also be created using the create_test_files.py script, these are also used during the testing.

To try something out, different experiments can be created, which can be trained and evaluated using the Experiment class.

## Running the tests

- The test can be run using pytest and will create a test_data directory, where the created test data will be saved.
- time_seg_data_loader can be used to identify bottlenecks and profiles the different steps in the loader. For the profiles, snakeviz is used, the command line arguments for visualization are printed at the end of the script. Iti s best to run it in an interactive window for better overview
