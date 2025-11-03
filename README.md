# Garbage Classification with Deep Learning

### Machine Learning Assignment Project (Part B)

The project is about the usage and evaluation of two deep learning models for a 12-class garbage classification problem. The main objective is to develop a model that is efficient and can correctly identify the category of waste just by looking at the image.

This GitHub repository holds the Jupyter Notebook (`mru.ipynb`) wherein all the data preprocessing, model construction, training, and testing are documented with code.

---

## 1. Dataset Source

The dataset adopted is the **Garbage Classification** (12 classes) dataset from Kaggle.

* **Source:** [https://www.kaggle.com/datasets/mostafaabla/garbage-classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
* **Total Images:** 15,515
* **Classes (12):** * [**ACTION:** Run `print(image_dataset.class_names)` in your notebook and list the 12 classes here.]
    * e.g., 'battery', 'biological', 'brown-glass', etc.

#### Data Preprocessing
The data was preprocessed through `tf.keras.utils.image_dataset_from_directory` with the defined parameters:
* **Image Size:** Resized all images to $128 \times 128$ pixels.
* **Labels:** One-hot encoded vectors (`label_mode='categorical'`) were assigned.
* **Splits:** The dataset was divided into 80% Training, 10% Validation, and 10% Test sets.
* **Performance:** The `tf.data` pipelines were enhanced by using `.cache()` and `.prefetch()`.

---

## 2. Methods (Models)

The project is about the implementation and comparison of two deep learning models as per the assignment's requirement.

### Model 1: Custom CNN (from the scratch)
* **Description:** [**ACTION:** Write 2-3 sentences describing your custom CNN. Mention that it contains 3 convolutional blocks (Conv2D + MaxPooling2D) followed by a `Flatten` and a `Dense` classifier. This is the 'ANN/CNN' model.]
* **Norma
