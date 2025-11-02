# Garbage Classification using Deep Learning

### Project for Machine Learning Assignment (Section B)

This project applies and compares two deep learning models for a 12-class garbage classification task. The goal is to build an efficient model capable of accurately sorting waste into its correct category based on an image.

This repository includes the Jupyter Notebook (`mru.ipynb`) with all data preprocessing, model building, training, and evaluation code.

---

## 1. Dataset Source

The dataset used is the **Garbage Classification** (12 classes) dataset from Kaggle.

* **Source:** [https://www.kaggle.com/datasets/mostafaabla/garbage-classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
* **Total Images:** 15,515
* **Classes (12):** * [**ACTION:** Run `print(image_dataset.class_names)` in your notebook and list the 12 classes here.]
    * e.g., 'battery', 'biological', 'brown-glass', etc.

#### Data Preprocessing
The data was loaded using `tf.keras.utils.image_dataset_from_directory` with the following specifications:
* **Image Size:** All images were resized to $128 \times 128$ pixels.
* **Labels:** Loaded as one-hot encoded vectors (`label_mode='categorical'`).
* **Splits:** The dataset was split into 80% Training, 10% Validation, and 10% Test sets.
* **Performance:** The `tf.data` pipelines were optimized using `.cache()` and `.prefetch()`.

---

## 2. Methods (Models)

This project implements and compares two deep learning models as required by the assignment.

### Model 1: Custom CNN (from scratch)
* **Description:** [**ACTION:** Write 2-3 sentences describing your custom CNN. Mention it has 3 convolutional blocks (Conv2D + MaxPooling2D) followed by a `Flatten` and a `Dense` classifier. This serves as the 'ANN/CNN' model.]
* **Normalization:** Pixel values were scaled to the `[0, 1]` range by dividing by 255.0.

### Model 2: Transfer Learning (MobileNetV2)
* **Description:** [**ACTION:** Write 2-3 sentences describing this model. Explain that you used the pre-trained MobileNetV2 model (with its ImageNet weights) as a "frozen" feature extractor. You then added a new classification head (a `GlobalAveragePooling2D` and a `Dense` layer) which was trained on the garbage data.]
* **Normalization:** This model used the specific `mobilenet_v2.preprocess_input` function, which scales pixels to the `[-1, 1]` range.

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

[**ACTION:** This is the most important part. Copy the **final results table** from your notebook (the one from the last cell) and paste it here. It should look like this:]

| Model | Test Accuracy | Weighted F1-Score |
| :--- | :---: | :---: |
| 1. Custom CNN | [Your_CNN_Accuracy] | [Your_CNN_F1_Score] |
| 2. Transfer Learning | [Your_TL_Accuracy] | [Your_TL_F1_Score] |

<br>

[**ACTION:** Take screenshots of your two training history plots (Accuracy/Loss vs. Epochs) from your notebook. Upload them to your GitHub repository and link them here.]

**Training History (Custom CNN):**
![Custom CNN Training History](custom_cnn_history.png)

**Training History (Transfer Learning):**
![Transfer Learning History](tl_history.png)

---

## 5. Conclusion

[**ACTION:** Write 2-4 sentences summarizing your findings.]

* Start by stating which model performed better. (e.g., "The Transfer Learning model (MobileNetV2) significantly outperformed the Custom CNN in all metrics.")
* Explain *why* you think it performed better. (e.g., "This demonstrates the power of transfer learning, as the MobileNetV2 base was already trained on millions of real-world images...")
* Add one final thought. (e.g., "For future work, fine-tuning the top layers of the MobileNetV2 model could potentially increase accuracy even further.")

---

## 6. References
* **Dataset:** [Garbage Classification (12 classes) on Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
* **Libraries:** [TensorFlow](https://www.tensorflow.org) & [Keras](https://keras.io/)
* **Model:** [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
