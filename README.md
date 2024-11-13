# Face Mask Detection

This project is designed to detect whether a person is wearing a face mask or not using deep learning techniques. It utilizes a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify images into two categories: "with mask" and "without mask". The dataset used for training the model is the [Face Mask Dataset](https://www.kaggle.com/omkargurav/face-mask-dataset) from Kaggle.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data Analysis](#data-analysis)
- [Model](#model)
- [Evaluation](#evaluation)
- [Predictive System](#predictive-system)
- [License](#license)

## Installation

To run this project, you need to install the following dependencies:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib
- Pillow
- Scikit-learn
- Kaggle

You can install the required libraries using `pip`:

```bash
pip install tensorflow keras numpy opencv-python matplotlib pillow scikit-learn kaggle
```

## Usage

### 1. Download the Dataset

The dataset used for training the model is available on Kaggle. You can download it directly using the Kaggle API. To do so, you need to authenticate with your Kaggle account and use the following code to download and extract the dataset:

```python
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
dataset_name = "omkargurav/face-mask-dataset"
api.dataset_download_files(dataset_name, path='.', unzip=True)
```

### 2. Data Analysis

The dataset consists of images categorized into two folders: `with_mask` and `without_mask`. Each image is classified into one of these categories, and labels are assigned accordingly.

```python
with_mask_files = os.listdir('./data/with_mask')
without_mask_files = os.listdir('./data/without_mask')
```

### 3. Image Preprocessing

Images are resized to 128x128 pixels and converted to RGB format for consistency across the dataset.

```python
image = Image.open('./data/with_mask/' + img_file).resize((128, 128)).convert('RGB')
```

### 4. Model Training

The model is built using a Convolutional Neural Network (CNN) with layers such as `Conv2D`, `MaxPooling2D`, and `Dense`. The model is compiled using the Adam optimizer and sparse categorical crossentropy loss function.

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
history = model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=5)
```

### 5. Model Evaluation

The trained model is evaluated on a test dataset to measure accuracy.

```python
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print('Test Accuracy = ', accuracy)
```

### 6. Plotting Loss and Accuracy

Training and validation loss, along with accuracy, are plotted to visualize the performance of the model.

```python
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

plt.plot(history.history['acc'], label='train accuracy')
plt.plot(history.history['val_acc'], label='validation accuracy')
plt.legend()
plt.show()
```

### 7. Predictive System

To predict if a person in a given image is wearing a mask, you can provide the image path as input to the following function:

```python
input_image_path = input('Path of the image to be predicted: ')
input_image = cv2.imread(input_image_path)
input_image_resized = cv2.resize(input_image, (128, 128))
input_image_scaled = input_image_resized / 255
input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])
input_prediction = model.predict(input_image_reshaped)
```

The result will display whether the person is wearing a mask or not.

## License

This project is open source and available under the MIT License. Feel free to fork, modify, and contribute!
```

### Instructions:

1. Copy the above content.
2. Open your `README.md` file in your project directory (or create one if it doesn't exist).
3. Paste the content into the file.
4. Save the file.

This will give a detailed and organized structure to your project, guiding users through installation, usage, and understanding how the project works.
