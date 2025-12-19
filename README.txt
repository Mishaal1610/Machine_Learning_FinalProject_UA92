
Project Overview

This project trains a machine-learning model called isolation forest to detect Network anomalies in network traffic.



pre requisites libraries and utilities.

	pandas 

	numpy

	matplotlib

	joblib 

	pickle 

	pathlib 

	argparse 

	logging 

Machine Learning

	scikit-learn
	
	IsolationForest 

	StandardScaler 



Summary of the Pipeline

1)Training Data Preparation
Script-prepare_training_data.py

Running this script extracts 50000 normal or benign data and 20000 ddos data and compiles them together and
saves the dataset as training_dataset.csv. The machine learning model will be trained with this dataset due to the original dataset
being too large. 

2)Model Training
Script-train_isoforest.py

Running this script will train isoforest on its features like flow_pckts etc.
The model will be saved as isoforest_model.pkl

3)Anomaly Detection
Script-analyze_data.py

Running this script will cause isomodel to analyze the first 200,000 rows of the dataset to check for anomalies.
detected anomalies will be saved to a csv file called detected_anomalies.csv


IMPORTANT

To run the program you need to have the datasets from https://www.kaggle.com/datasets/devendra416/ddos-datasets

unbalanced_20_80_dataset.csv
final_dataset.csv

I have not attached the datasets to this upload as they are too large, have both the datasets from the link above to run the program.

Thanks

