# Fake News Detection using NLP & Machine Learning

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Scikit--learn%20%7C%20NLTK%20%7C%20Pandas-orange.svg)


This repository contains the code for a machine learning project focused on detecting fake news articles using Natural Language Processing (NLP). The project demonstrates an end-to-end workflow, including data preprocessing, feature engineering, model training, and performance evaluation by comparing a Linear Support Vector Classifier (LinearSVC) and a Multinomial Naive Bayes model.

## Table of Contents

- [About The Project](#about-the-project)
- [Key Features](#key-features)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Results and Analysis](#-results-and-analysis)

## About The Project

In the age of digital information, the proliferation of fake news poses a significant challenge. This project addresses this problem by building a robust classification system to distinguish between real and fake news articles. It employs a comprehensive NLP pipeline to process and clean the text data, which is then converted into numerical features using TF-IDF. Finally, two supervised learning models are trained and evaluated to determine the most effective approach for this classification task.

## Key Features

- **End-to-End NLP Pipeline:** Implements a full text preprocessing workflow including tokenization, lowercasing, stopword removal, and lemmatization with POS tagging.
- **TF-IDF Feature Extraction:** Converts raw text into meaningful numerical representations that capture word importance.
- **Model Comparison:** Trains and evaluates two popular text classification models: LinearSVC (SVM) and Multinomial Naive Bayes.
- **Performance Evaluation:** Provides a detailed analysis of model performance using metrics like Accuracy, Precision, Recall, and F1-Score.
- **Data Visualization:** Utilizes confusion matrices, classification reports, and word clouds for insightful analysis and clear presentation of results.

## 📊 Dataset

The project uses the "Fake and Real News Dataset" from Kaggle.

- **Source:** [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Description:** The dataset consists of two separate CSV files:
  - `True.csv`: 21,417 real news articles.
  - `Fake.csv`: 23,481 fake news articles.
- **Attributes:**
  - `title`: The headline of the news article.
  - `text`: The full text content of the article.
  - `subject`: The subject category of the news.
  - `date`: The publication date.

## 💻 Tech Stack

- **Language:** Python 3
- **Core Libraries:**
  - **Data Manipulation:** `pandas`, `numpy`
  - **Machine Learning:** `scikit-learn`
  - **NLP:** `nltk`, `spacy`
  - **Visualization:** `matplotlib`, `seaborn`, `wordcloud`

## 📁 Project Structure
~~~
.
├── Fake_News_Detection.ipynb       # Main Jupyter Notebook with all the code
└── Dataset/
├── Fake.csv                    # Dataset of fake news articles
└── True.csv                    # Dataset of real news articles
~~~
## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/fake-news-detection.git](https://github.com/your-username/fake-news-detection.git)
    cd fake-news-detection
    ```
2.  **Install the required Python packages:**
    ```sh
    pip install pandas numpy scikit-learn nltk spacy matplotlib seaborn wordcloud jupyter
    ```
3.  **Download NLTK and SpaCy data models:**
    Open a Python interpreter and run the following commands:
    ```python
    import nltk
    import spacy
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    spacy.cli.download("en_core_web_sm")
    ```
4.  **Download the Dataset:**
    - Download the data from the [Kaggle link](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
    - Create a `Dataset` folder in the project's root directory.
    - Place `True.csv` and `Fake.csv` inside the `Dataset` folder.

## Usage

1.  Launch the Jupyter Notebook server:
    ```sh
    jupyter notebook
    ```
2.  Open the `Fake_News_Detection.ipynb` file in your browser.
3.  **Important:** Verify that the file paths for the dataset in the notebook (`data1_path` and `data2_path` variables) correctly point to the location of your CSV files.
4.  Run the cells sequentially to execute the entire workflow.

## 📈 Results and Analysis

The final evaluation showed that the **Linear Support Vector Classifier (LinearSVC) model significantly outperformed the Multinomial Naive Bayes model**, achieving near-perfect classification scores.

#### Performance Metrics

| Metric | LinearSVC (SVM) | Multinomial Naive Bayes |
| :--- | :---: | :---: |
| **Accuracy** | 99.37% | 92.86% |
| **Precision** | 0.99 | 0.93 |
| **Recall** | 0.99 | 0.93 |
| **F1-Score** | 0.99 | 0.93 |

#### Analysis

- **LinearSVC:** The confusion matrix and high precision/recall scores indicate that the SVM model is extremely effective at correctly identifying both real and fake news, with a very low number of false positives and false negatives.
- **Multinomial Naive Bayes:** While still performing well with over 92% accuracy, this model is noticeably less precise than the SVM, making more misclassifications.
- **Conclusion:** For this dataset and text representation, the LinearSVC model is the superior choice for building a reliable fake news detector.
