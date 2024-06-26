# Cat vs. Dog Image Classification

This project aims to classify images of cats and dogs using a Support Vector Machine (SVM) with a Principal Component Analysis (PCA) for dimensionality reduction. The project includes data preprocessing, model training, evaluation, and evaluation results.

## Project Structure

The project consists of the following files and directories:

- `Task3.py`: Python script for training and evaluating the SVM model.
- `confusion_matrix.png`: Image file for the confusion matrix.
- `classification_report.txt`: Text file for the classification report.
- `README.md`: This file.

## Requirements

- Python 3.6+
- numpy
- scikit-learn
- matplotlib
- tqdm
- joblib
- cv2 (OpenCV)
- seaborn

You can install the required libraries using pip:

```sh
pip install numpy scikit-learn matplotlib tqdm joblib opencv-python seaborn
```

## Code Overview

The script `Task3.py` performs the following steps:

1. **Setup and Paths**: Define paths for saving the confusion matrix, classification report, and the model.

    ```python
    folder_path = "E:\\ML Intern\\task 3 csv files"
    os.makedirs(folder_path, exist_ok=True)

    confusion_image_path = os.path.join(folder_path, 'confusion matrix.png')
    classification_file_path = os.path.join(folder_path, 'classification_report.txt')
    model_file_path = os.path.join(folder_path, "svm_model.pkl")
    ```

2. **Load and Preprocess Data**: Load the images, resize them, normalize, and flatten for feature extraction. Label the images (dog = 1, cat = 0).

    ```python
    train_dir = os.path.join(dataset_dir, "train")
    image_size = (50, 50)
    # Processing code for images...
    ```

3. **Train-Test Split**: Split the data into training and testing sets.

    ```python
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=42)
    ```

4. **PCA and SVM Pipeline**: Set up a pipeline with PCA and SVM, and perform a grid search to find the best parameters.

    ```python
    pca = PCA(n_components=0.8, random_state=42)
    svm = SVC()
    pipeline = Pipeline([('pca', pca), ('svm', svm)])
    param_grid = {
        'pca__n_components': [2, 1, 0.9, 0.8],
        'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=4)
    grid_search.fit(X_train, y_train)
    ```

5. **Model Evaluation**: Evaluate the best model on the test set and save the results.

    ```python
    best_pipeline = grid_search.best_estimator_
    accuracy = best_pipeline.score(X_test, y_test)
    y_pred = best_pipeline.predict(X_test)

    classification_rep = classification_report(y_test, y_pred, target_names=['Cat', 'Dog'])
    with open(classification_file_path, 'w') as file:
        file.write(classification_rep)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(confusion_image_path)
    plt.show()
    ```


## Running the Code

1. Ensure that the training and testing image directories are correctly set up as specified in the script.
2. Run the script `Task3.py`.

    ```sh
    python Task3.py
    ```

3. The script will display the confusion matrix, and print the classification report along with the trained model.

## Acknowledgements

This project is a part of a machine learning internship task for image classification.


