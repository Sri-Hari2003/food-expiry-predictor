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

The repository includes Python code for training a machine learning model to classify fresh and expired fruits and vegetables. The model is a convolutional neural network (CNN) implemented using TensorFlow and Keras.

- `cnnmodel.h5`: A pre-trained CNN model for classifying fresh and expired items.

## Expiration Date Prediction

The repository also includes Python code for predicting the expiration date of a product based on its category and the purchase date. The prediction is made using a dataset that contains information about average shelf life for different categories of products.

- `dataset.xlsx`: The dataset used for expiration date prediction.

## Usage

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Sri-Hari2003/Fruit-and-Vegetable-Shelf-Life-Dataset.git
   ```

2. Install the required dependencies (ensure you have TensorFlow, scikit-learn, fuzzywuzzy, pandas, and other necessary packages installed).

3. Follow the code examples provided in the repository to use the machine learning model for classification and the expiration date prediction system.

## License

This project is licensed under the [Apache License](LICENSE).

