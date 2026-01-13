# Toxic Comment Classification with LSTM and Streamlit

This project implements a deep learning model using Long Short-Term Memory (LSTM) networks to classify comments as toxic or non-toxic. The model is trained on a dataset of online comments and deployed as an interactive web application using Streamlit.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [File Structure](#file-structure)

## Project Overview
Online platforms often struggle with toxic content. This project aims to build a robust system that can automatically detect and flag toxic comments, contributing to a healthier online environment. The core of the solution is an LSTM-based neural network trained on a large corpus of labeled comments.

## Features
- **Text Preprocessing**: Includes lowercasing, punctuation removal, stop word removal, and stemming.
- **Tokenizer**: Converts text into numerical sequences suitable for neural networks.
- **LSTM Model**: A sequential model with an Embedding layer, LSTM layer, Dropout, and a Dense sigmoid output layer for binary classification.
- **Streamlit Web Application**: An easy-to-use interface to input comments and get real-time toxicity predictions.
- **Ngrok Integration**: Allows public access to the locally hosted Streamlit application.

## Dataset
The model is trained on the `train.csv` dataset, which contains a collection of online comments labeled as either toxic (1) or non-toxic (0).

## Model Architecture
The neural network consists of:
- **Input Layer**: Takes sequences of length `MAX_LEN` (150).
- **Embedding Layer**: Converts word indices into dense vectors of size 128 for `MAX_WORDS` (10000) unique words.
- **LSTM Layer**: A recurrent layer with 128 units to capture sequential dependencies in text.
- **Dropout Layer**: With a rate of 0.5 to prevent overfitting.
- **Dense Layer**: A single output neuron with a sigmoid activation function for binary classification.

## Setup and Installation
To set up and run this project locally, follow these steps:

1.  **Clone the repository** (if applicable, or download the project files):
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2.  **Install the required Python packages**:
    ```bash
    pip install streamlit pyngrok tensorflow nltk scikit-learn pandas numpy
    ```

3.  **Download NLTK stopwords**:
    ```python
    import nltk
    nltk.download("stopwords")
    ```

4.  **Prepare the dataset**:
    - Ensure you have `train.csv` in your project directory. This dataset is used for training the model.

5.  **Obtain an ngrok Authtoken**:
    - Go to [ngrok.com](https://ngrok.com/) and sign up for a free account.
    - Get your authtoken from the ngrok dashboard.
    - Add your authtoken to ngrok configuration (replace `<YOUR_NGROK_AUTH_TOKEN>`):
    ```bash
    !ngrok config add-authtoken <YOUR_NGROK_AUTH_TOKEN>
    ```

## Usage

1.  **Train the model**:
    Run the Python cells in the notebook (or script) that perform data preprocessing, model definition, and training.
    This will save `toxicity_model.keras` and `tokenizer.pkl` files.

2.  **Run the Streamlit application**:
    The `app.py` script contains the Streamlit application code. Execute it in your terminal:
    ```bash
    streamlit run app.py
    ```
    *Note: In a Colab environment, this is often run in the background like so: `!streamlit run app.py &>/dev/null &`*

3.  **Access the Streamlit app via ngrok**:
    Once the Streamlit app is running, use ngrok to expose it to the internet. The following Python code will print the public URL:
    ```python
    from pyngrok import ngrok

    public_url = ngrok.connect(8501) # Streamlit usually runs on port 8501
    print("Streamlit App URL:", public_url)
    ```
    Open the provided `public_url` in your web browser to interact with the toxicity detection application.

## File Structure
- `train.csv`: The dataset used for training the model.
- `app.py`: The Streamlit web application code.
- `toxicity_model.keras`: The saved Keras model.
- `tokenizer.pkl`: The saved Keras tokenizer object.
- `README.md`: This file.

Colab paid products - Cancel contracts here

