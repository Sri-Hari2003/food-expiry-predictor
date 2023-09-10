# Fruit and Vegetable Dataset for Shelf Life

This repository contains a dataset for fruit and vegetable images along with code for a machine learning model and an expiration date prediction system.

## Dataset Description

The dataset used in this project is available on Kaggle and can be found at the following link: [Fruit and Vegetable Dataset for Shelf Life](https://www.kaggle.com/datasets/soorajkavumpadi/fruit-and-vegetable-dataset-for-shelf-life).

The dataset is organized into categories, with each category representing a different type of fruit or vegetable. Each category further contains subdirectories for fresh and expired items. The dataset is structured as follows:

```
dataset/
    ├── category_1/
    │   ├── Fresh/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── Expired/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    ├── category_2/
    │   ├── Fresh/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── Expired/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    ├── ...
```

## Machine Learning Model
The repository includes Python code for training a robust machine learning model to classify fresh and expired fruits and vegetables. The model is a convolutional neural network (CNN) implemented using TensorFlow and Keras. We provide a pre-trained model ready for use.

### Pre-trained Model
cnnmodel.h5: A pre-trained CNN model for classifying fresh and expired items. This model has been trained on the provided dataset and can be used for your own classification tasks.
### Usage Example
You can utilize the pre-trained model to classify fresh and expired items in your own images. We offer code examples and guidance in the repository on how to use the model effectively.

## Expiration Date Prediction
In addition to classification, the repository also includes Python code for predicting the expiration date of a product based on its category and the purchase date. This prediction is made using a dataset that provides information about the average shelf life for different categories of products.

### Dataset
dataset.xlsx: The dataset used for expiration date prediction. This dataset offers insights into the average shelf life (in days) for various product categories.
Expiration Date Prediction Function
We provide a Python function to predict the expiration date of a product. You can input the product's category and purchase date to get an estimate of its remaining shelf life or determine if it has already expired.

## Usage

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Sri-Hari2003/food-expiry-predictor
   ```

2. Install the required dependencies (ensure you have TensorFlow, scikit-learn, fuzzywuzzy, pandas, and other necessary packages installed).

3. Follow the code examples provided in the repository to use the machine learning model for classification and the expiration date prediction system.

## License

This project is licensed under the [Apache License](LICENSE).

