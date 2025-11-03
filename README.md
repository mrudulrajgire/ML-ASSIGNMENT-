# Garbage Classification with Deep Learning

### Machine Learning Assignment Project (Part B)

The project is about the usage and evaluation of two deep learning models for a 12-class garbage classification problem. The main objective is to develop a model that is efficient and can correctly identify the category of waste just by looking at the image.

This GitHub repository holds the Jupyter Notebook (`mru.ipynb`) wherein all the data preprocessing, model construction, training, and testing are documented with code.

---

## 1. Dataset Source

The dataset adopted is the **Garbage Classification** (12 classes) dataset from Kaggle.

* **Source:** [https://www.kaggle.com/datasets/mostafaabla/garbage-classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
* **Total Images:** 15,515
* **Classes (12):** `battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass`.

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
* **Description:** This model was built from scratch to satisfy the `CNN`/`ANN` requirement. The architecture consists of three convolutional blocks (a `Conv2D` layer followed by `MaxPooling2D`) with 32, 64, and 128 filters, respectively. This feature extraction base is followed by a `Flatten` layer and a dense classifier (`ANN`) with 128 units and a `Dropout` layer (0.5 rate) for regularization.
* **Normalization:** Pixel values were scaled to the `[0, 1]` range by dividing by 255.0.

### Model 2: Transfer Learning (MobileNetV2)
* **Description:** This model uses the powerful **MobileNetV2** architecture, pre-trained on ImageNet, as a frozen feature extractor. We 'froze' the convolutional base to leverage its learned features and added a new classifier head on top. This head consists of a `GlobalAveragePooling2D` layer (to reduce parameters), a `Dropout` layer (0.2 rate), and a final `Dense` layer with 12 `softmax` outputs for our specific classes.
* **Normalization:** This model required specific preprocessing. We used the `mobilenet_v2.preprocess_input` function, which scales pixel values to the `[-1, 1]` range.

---

## 3. Steps to Run the Code

This project is designed to be run in a **Google Colab** environment.

1.  **Open in Colab:** Upload the `mru.ipynb` file to Google Colab.
2.  **Enable GPU:** For faster training, go to `Runtime > Change runtime type` and select `T4 GPU`.
3.  **Run All Cells:** Click `Runtime > Run all`.
    * The notebook uses the `kagglehub` library, which automatically downloads the correct dataset.
    * The script will preprocess the data, build both models, train them, and print the final evaluation.

---

## 4. Experiments & Results

Both models were trained and evaluated on the held-out **test set (10% of the data)**. The primary metrics used for comparison are Test Accuracy and Weighted F1-Score.

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
The Transfer Learning model (MobileNetV2) significantly outperformed the Custom CNN in all metrics, achieving **[Your_TL_Accuracy]%** accuracy. This demonstrates the power of transfer learning, as the MobileNetV2 base was already trained on millions of real-world images, providing a much stronger starting point for feature extraction than our simple CNN. For future work, fine-tuning the top layers of the MobileNetV2 model could potentially increase accuracy even further.

---

## 6. References
* **Dataset:** [Garbage Classification (12 classes) on Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
* **Libraries:** [TensorFlow](https://www.tensorflow.org) & [Keras](https://keras.io/)
* **Model:** [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
