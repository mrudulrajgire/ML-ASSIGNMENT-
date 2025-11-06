# Garbage Classification with Deep Learning

### Machine Learning Assignment Project (Part B)

The main purpose of the project is to develop, evaluate, and compare two deep learning models for the problem of classifying garbage into 12 categories. The project aims to build a model that is not only efficient but also accurate enough to determine the waste category merely by its image.

The repository on GitHub contains the Jupyter Notebook (`mru.ipynb`), which includes coding and layout for the whole process from data preprocessing, model building, training, and testing.

---

## 1. Dataset Source

The dataset adopted is the **Garbage Classification** (12 classes) dataset from Kaggle.

* **Source:** [https://www.kaggle.com/datasets/mostafaabla/garbage-classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
* **Total Images:** 15,515
* **Classes (12):** `battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass`.

#### Data Preprocessing
The training data underwent preprocessing using the function `tf.keras.utils.image_dataset_from_directory` with the specified characteristics:
* **Image Size:** All images were resized to $128 \times 128$ pixels.
* **Labels:** Vectors of one-hot encoding were assigned (`label_mode='categorical'`).
* **Splits:** The original dataset was partitioned into 80% Training, 10% Validation, and 10% Test sets.
* **Performance:** Performance of the `tf.data` pipelines was improved using `.cache()` and `.prefetch()`.

---

## 2. Methods (Models)

This undertaking is fundamentally about the execution and evaluation of two deep learning algorithms in alignment with the requirements of the assignment.

### Model 1: Custom CNN (from the scratch)
* **Description:** This model was developed from the ground up to fulfill the `CNN`/`ANN` requirement. The network is made up of three convolutional blocks (a `Conv2D` layer and then `MaxPooling2D`) with 32, 64, and 128 filters, in that order. The feature extraction stage is succeeded by a `Flatten` layer and a dense classifier (`ANN`) consisting of 128 units and a `Dropout` layer (0.5 rate) for regularization.
* **Normalization:** The pixel values were transformed into the `[0, 1]` interval by dividing them by 255.0.

### Model 2: Transfer Learning (MobileNetV2)
* **Description:** The powerful **MobileNetV2** with its pre-trained weights on ImageNet is utilized in this model as a frozen feature extractor. The convolutional base was 'frozen' in order to reap the benefits of its learned features and a classifier head was added on top. The head includes a `GlobalAveragePooling2D` layer (to keep fewer parameters), a `Dropout` layer (0.2 rate), and a final `Dense` layer with 12 `softmax` outputs for our unique classes.
* **Normalization:** The mobilenet_v2 model needed to be specifically preprocessed. The function `mobilenet_v2.preprocess_input`, which maps pixel values to the `[-1, 1]` range, was employed.

---

## 3. Steps to Run the Code

This project is meant to be executed in a **Google Colab** setting.

1.  **Open in Colab:** First, upload the `mru.ipynb` file to your Google Colab account.
2.  **Enable GPU:** To speed up the training process, change the runtime by going to `Runtime > Change runtime type` and selecting `T4 GPU`.
3.  **Run All Cells:** After that, click `Runtime > Run all`.
    * The `kagglehub` library is used in the notebook, which automatically downloads the appropriate dataset.
    * The code will handle data preprocessing, model building for both, training, and finally, evaluation printout.
---

## 4. Experiments & Results

The **test set (10% of the data)** that was set aside for this purpose was used for training and evaluating both models. The main criteria for comparison were Test Accuracy and Weighted F1-Score.

**[ACTION: Run the final cells of your notebook and copy the results table here.]**

| Model | Test Accuracy | Weighted F1-Score |
| :--- | :---: | :---: |
| 1. Custom CNN | **[Your_CNN_Accuracy]** | **[Your_CNN_F1_Score]** |
| 2. Transfer Learning | **[Your_TL_Accuracy]** | **[Your_TL_F1_Score]** |

<br>

**[ACTION: Take screenshots of your two training history plots. Upload them to your GitHub repository and change the file names below to match.]**

**Training History (Custom CNN):**
![Custom CNN Training History](custom_cnn_history.png)

**Training History (Transfer Learning):**
![Transfer Learning History](tl_history.png)

---

## 5. Conclusion

**[ACTION: Write 2-4 sentences summarizing your findings based on the results table.]**

*Example:*
All metrics were surpassed by the Transfer Learning model (MobileNetV2) and the accuracy of **[Your_TL_Accuracy]%** was attained. This is a clear indication of the advantages of transfer learning as the MobileNetV2 base was previously trained on millions of actual images thus giving a much better starting point for feature extraction compared to our basic CNN. Future work may be done in such a way that the top layers of the MobileNetV2 model are gradually unfreeze and cause even more increase in accuracy.

---

## 6. References
* **Dataset:** [Garbage Classification (12 classes) on Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
* **Libraries:** [TensorFlow](https://www.tensorflow.org) & [Keras](https://keras.io/)
* **Model:** [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
