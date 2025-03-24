"""
Face Mask Detection System combining traditional ML and deep learning approaches
This script contains two independent implementations:
1. Traditional approach using HOG features with SVM/Neural Network
2. Deep learning approach using CNN
"""
###########################Importing Necessary Libraries#################################
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
from PIL import Image
from scipy.stats import uniform, randint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ---------------------------- Common Utility Functions ----------------------------

def is_valid_image(image_path):
    """Checking if the image file is valid and not corrupted"""
    try:
        img = Image.open(image_path)
        img.verify()  # Verifying the file integrity
        return True
    except Exception as e:
        print(f"Found invalid image: {image_path}. Error: {str(e)}")
        return False

def convert_to_jpeg(image_path):
    """Converting unsupported image formats to JPEG format"""
    try:
        img = Image.open(image_path)
        new_path = os.path.splitext(image_path)[0] + ".jpeg"
        img.convert('RGB').save(new_path, "JPEG")
        os.remove(image_path)
        return new_path
    except Exception as e:
        print(f"Failed converting image {image_path}: {str(e)}")
        return None

# ---------------------------- Part A: Traditional ML Approach ----------------------------

def apply_clahe(image):
    """Applying Contrast Limited Adaptive Histogram Equalization to enhance image"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel_clahe = clahe.apply(l_channel)
    lab_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

def preprocess_image_ml(image_path, target_size=(64, 64)):
    """Preprocessing images for traditional ML approach"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        enhanced_img = apply_clahe(img_array)
        return enhanced_img
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def extract_features(image):
    """Extracting HOG features from the image"""
    return hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)

def load_ml_dataset(data_path):
    """Loading dataset for traditional ML methods"""
    images, labels = [], []
    print("Loading dataset for traditional ML approach...")
    
    for label, folder in enumerate(['with_mask', 'without_mask']):
        folder_path = os.path.join(data_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found!")
            continue
            
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            if not is_valid_image(img_path):
                continue
                
            processed_img = preprocess_image_ml(img_path)
            if processed_img is not None:
                images.append(processed_img)
                labels.append(label)
                
    print(f"Loaded {len(images)} images for ML approach")
    return np.array(images), np.array(labels)
#--------------------------Hyperparameter tuning for SVM------------------------
def train_svm_with_tuning(X_train, y_train):
    """Tuning and training SVM classifier with randomized search"""
    print("Tuning SVM hyperparameters...")
    param_dist = {
        'C': uniform(0.1, 10),
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svm = SVC()
    search = RandomizedSearchCV(
        svm, param_dist, n_iter=10, cv=3, 
        scoring='accuracy', random_state=42
    )
    search.fit(X_train, y_train)
    print(f"Best SVM params: {search.best_params_}")
    return search.best_estimator_
#-----------------------Hyperparameter tuning for Neural Network-----------------
def train_nn_with_tuning(X_train, y_train):
    """Tuning and training Neural Network classifier"""
    print("Tuning Neural Network hyperparameters...")
    param_dist = {
        'hidden_layer_sizes': [(100,), (200,)],
        'activation': ['relu', 'tanh'],
        'alpha': uniform(0.0001, 0.1),
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [500]
    }
    nn = MLPClassifier(random_state=42)
    search = RandomizedSearchCV(
        nn, param_dist, n_iter=10, cv=3,
        scoring='accuracy', random_state=42
    )
    search.fit(X_train, y_train)
    print(f"Best NN params: {search.best_params_}")
    return search.best_estimator_

def run_traditional_ml():
    """Running the traditional ML pipeline"""
    print("\n" + "="*50)
    print("Running Traditional ML Approach")
    print("="*50 + "\n")
    
    data_path = os.path.join(os.path.dirname(__file__), 'dataset')
    images, labels = load_ml_dataset(data_path)
    
    # Extracting HOG features
    print("Extracting HOG features...")
    features = np.array([extract_features(img) for img in images])
    
    # Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Training and evaluating SVM
    print("\nTraining SVM classifier...")
    svm_model = train_svm_with_tuning(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred)*100:.2f}%")
    
    # Training and evaluating Neural Network
    print("\nTraining Neural Network classifier...")
    nn_model = train_nn_with_tuning(X_train, y_train)
    nn_pred = nn_model.predict(X_test)
    print(f"NN Accuracy: {accuracy_score(y_test, nn_pred)*100:.2f}%")

# ---------------------------- Part B: Deep Learning Approach ----------------------------

def preprocess_image_dl(image_path, target_size=(150, 150)):
    """Preprocessing images for deep learning approach"""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalizing to [0,1]
        return img_array
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def load_dl_dataset(data_path):
    """Loading dataset for deep learning approach"""
    images, labels = [], []
    print("Loading dataset for deep learning approach...")
    
    for label, folder in enumerate(['with_mask', 'without_mask']):
        folder_path = os.path.join(data_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found!")
            continue
            
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            
            # Handling unsupported formats
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                converted_path = convert_to_jpeg(img_path)
                if converted_path:
                    img_path = converted_path
                else:
                    continue
                    
            if not is_valid_image(img_path):
                continue
                
            processed_img = preprocess_image_dl(img_path)
            if processed_img is not None:
                images.append(processed_img)
                labels.append(label)
    
    if not images:
        raise ValueError("No valid images found in dataset!")
    
    print(f"Loaded {len(images)} images for DL approach")
    return np.array(images), np.array(labels)

def build_cnn_model(input_shape=(150, 150, 3)):
    """Building CNN model architecture"""
    print("Constructing CNN model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def run_deep_learning():
    """Running the deep learning pipeline"""
    print("\n" + "="*50)
    print("Running Deep Learning (CNN) Approach")
    print("="*50 + "\n")
    
    data_path = os.path.join(os.path.dirname(__file__), 'dataset')
    try:
        images, labels = load_dl_dataset(data_path)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    # Building CNN model
    print("\nBuilding CNN model architecture...")
    model = build_cnn_model()
    
    # Display model summary
    print("\nCNN Model Summary:")
    model.summary()
    
    # Training CNN
    print("\nTraining CNN model:")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1  # Show progress bars
    )
    
    # Evaluating CNN
    print("\nEvaluating CNN on test set:")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nCNN Test Accuracy: {test_acc*100:.2f}%")
    print(f"CNN Test Loss: {test_loss:.4f}")
    
    # Making predictions
    print("\nGenerating CNN predictions:")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Detailed performance report
    print("\nCNN Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Without Mask', 'With Mask']
    ))
    
    # Confusion matrix
    print("\nCNN Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("          Predicted")
    print("Actual   No Mask   Mask")
    print(f"No Mask    {cm[0,0]}        {cm[0,1]}")
    print(f"Mask       {cm[1,0]}        {cm[1,1]}")
    
    # Training history visualization (optional)
    print("\nCNN Training History:")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")

# ---------------------------- Main Execution ----------------------------

if __name__ == "__main__":
    # Running both approaches independently
    run_traditional_ml()
    run_deep_learning()