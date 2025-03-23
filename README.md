# Fraud Detection System

This repository contains the implementation of a fraud detection system using machine learning. The goal of this project is to identify potentially fraudulent transactions based on various features provided in the dataset. The project includes a user-friendly web interface to interact with the model, view transaction details, upload CSV files for batch processing, and monitor real-time alerts. Additionally, the system incorporates **data encryption** to ensure secure handling of sensitive information.

## Project Overview

Fraud detection is crucial for businesses to prevent financial losses and maintain trust. This project uses the Kaggle Credit Card Fraud Detection Dataset to build a model capable of predicting fraudulent transactions. The web application provides a dashboard for real-time monitoring, detailed analysis of transactions, and timely alerts. 

The system allows users to **upload CSV files** containing transaction data, which are processed through a **data pipeline** to predict fraudulent transactions. All sensitive data is **encrypted** both during file upload and processing to ensure security and privacy.

## Dataset

The dataset used for this project is the Kaggle Credit Card Fraud Detection Dataset, which includes the following features:
- **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset.
- **V1-V28**: Anonymized features derived from PCA to ensure privacy and reduce dimensionality.
- **Amount**: Transaction amount.
- **Class**: 1 for fraudulent transactions, 0 for non-fraudulent transactions.

### Data Preparation

- Initial exploration and cleaning of the dataset.
- Handling missing values, encoding categorical variables, and scaling numerical features.
- Explanation of PCA and its impact on feature extraction.
- **Encryption**: All uploaded transaction data is encrypted using **Fernet encryption** to ensure data security and privacy.

### Model Development

- Building and training a fraud detection model using algorithms such as **Logistic Regression**, **Random Forest**, or **Gradient Boosting**.
- Evaluation of model performance using metrics like **accuracy**, **precision**, **recall**, and **F1 score**.

### Data Pipeline

- A pipeline that processes uploaded CSV files by first encrypting the data, performing necessary **data preprocessing**, passing it through the fraud detection model, and returning the results.
- **Real-time processing** of transactions via the web interface, with results displayed immediately.

### User Interface

- Users can upload CSV files containing transaction details (amount, anonymized features) for fraud detection.
- Real-time alerts for suspected fraudulent transactions.
- Visualizations for detailed transaction analysis.
- Users can **download encrypted results** of the predictions as a CSV file for record-keeping.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone git@github.com:KeabetsweMotloung/Fraud-detection-System.git
    cd fraud-detection-system
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    cd Server and run 'python server.py'
    ```

**Note**: Ensure you are using Python 3.x and have all necessary system dependencies installed.

## Usage

1. **Data Upload**: Upload a CSV file containing transaction data via the web interface. The data will be encrypted, processed, and passed through the fraud detection model.
2. **Model Prediction**: Receive predictions on whether each transaction is fraudulent or not, along with real-time alerts.
3. **Results**: View and download the results in an encrypted CSV file.

## Results

The model demonstrates the ability to identify fraudulent transactions with high accuracy. The web interface allows for easy interaction, real-time monitoring, and detailed transaction analysis. The system ensures **data privacy** and **security** by encrypting all uploaded and processed data.

## Contributing

Contributions are welcome! Please follow the guidelines for code style and testing when submitting a Pull Request or opening an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for more details. The MIT License allows for commercial and non-commercial use, modification, distribution, and private use of the software.

## Acknowledgements

- Special thanks to Kaggle for providing the dataset.
- Inspired by various online resources and communities dedicated to data science and machine learning.
- **Data Security**: Thanks to the **cryptography** library for providing encryption tools to protect sensitive data.
