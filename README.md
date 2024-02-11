# Stock Market Future Prediction Example

This repository provides an example program demonstrating the use of the stockmarket-future-prediction model to predict stock market directions (UP or DOWN) using time series data.

## Large File Storage (LFS) Requirements

This repository contains large data samples managed with Git Large File Storage (LFS). To work with this repository, follow these steps:

* Install Git LFS:
   sudo apt-get update
   sudo apt-get install -y git-lfs
   git lfs install 

* Clone the Repository:
   git clone https://github.com/Ruphobia/foduucom_stockmarket-future-prediction_example.git

Important Note: Git LFS tracks large files using lightweight pointer files within your repository. This helps keep your repository efficient while the actual large files are stored on a separate LFS server. 


## Overview

Utilizing the `stockmarket-future-prediction` model from foduucom, hosted on Hugging Face, this example processes minute-by-minute stock symbol data. It plots this data on an OpenCV matrix, applies the prediction model to forecast future trends, and saves the results as .jpg images. This procedure aims to offer valuable predictions on whether a stock's price is expected to rise or fall, serving as a tool for traders and financial analysts.

## Dataset and Model

- **Dataset**: The `testdata.parquet` file used in this example is sourced from [Hugging Face Datasets](https://huggingface.co/datasets/edarchimbaud/timeseries-1m-stocks), featuring time-series data for stock prices.
  
- **Model**: The predictive modeling relies on `best.pt`, available at [Hugging Face Models](https://huggingface.co/foduucom/stockmarket-future-prediction), developed by foduucom for forecasting stock market movements.

## License

This project is released under the [Unlicense](https://unlicense.org/), allowing you to freely use, modify, distribute, or do anything you wish with the code. However, it's important to note that while you can do whatever you want with the code provided in this repository, the datasets and models used (sourced from Hugging Face) have their own licenses. Users are responsible for reviewing and complying with those licenses when using or distributing the dataset and model.
