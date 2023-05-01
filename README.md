# Sephora Skincare Reviews Sentiment Analysis

This project performs sentiment analysis on Sephora skincare reviews using a pre-trained DistilBERT model fine-tuned on a custom dataset.

## Introduction

Sentiment analysis is a natural language processing technique used to determine the sentiment expressed in a piece of text. In this project, we use a pre-trained DistilBERT model from Hugging Face to perform sentiment analysis on Sephora skincare product reviews. The model is fine-tuned on a custom dataset to improve performance on the specific domain.

## Dataset

The dataset used in this project is available on Kaggle: [Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews?resource=download&select=reviews_0_250.csv).

The dataset contains various columns, such as product_id, user_id, rating, and review_text. For our sentiment analysis task, we focus on the `review_text` column.

## Dependencies

To run this project, you will need the following libraries:

- Python 3.7 or later
- PyTorch 1.9.0 or later
- TorchText 0.6.0
- Transformers 4.0.0 or later
- Pandas
- CUDA (if using a GPU)

## Installation

1. Clone this repository:
  ```bash
  git clone https://github.com/yourusername/sephora-skincare-reviews-sentiment-analysis.git
  ```

2. Change to the project directory:
  ```bash
  cd sephora-skincare-reviews-sentiment-analysis
  ```

3. Install the required libraries:
  ```bash
  pip install -r requirements.txt
  ```

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews?resource=download&select=reviews_0_250.csv) and place it in the `data` folder.


## Usage - (as Python Project in local env)

1. Preprocess the dataset by running `preprocess_data.py`:

```bash
python preprocess_data.py
```

This script will split the dataset into training, validation, and test sets.
2. Fine-tune the DistilBERT model using the training data by running `train.py`:
```bash
python train.py
```

3. Evaluate the model on the test data by running `evaluate.py`:
```bash
python evaluate.py
```

4. To perform sentiment analysis on new data, use the `predict_sentiment` function in the `predict.py` script.

## Usage - (as Notebook .ipynb project)

1. Download or open the .ipynb file in a notebook of your choice

2. Run each cell in a sequential order

3. To perform sentiment analysis on new data, use the `predict_sentiment` function from the dedicated cell at the end

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
