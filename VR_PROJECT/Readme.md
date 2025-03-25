# Team members: R Lakshman IMT2022090
#               Shreyas S  IMT2022078
#           Tanay Nagarkar IMT2022083 
#  VR mini Project-1: Face Mask Detection, Classification, and Segmentation
The below parts implements a **face mask detection system** using two independent approaches:  
1. **Traditional Machine Learning** - HOG features with SVM/Neural Network  
2. **Deep Learning** - Convolutional Neural Networks (CNN)  

---

## Requirements  
To run this project firstly download the project the download the dataset files from the link given then place the dataset and MSFD folders inside the VR_PROJECT,Then run the python files of all parts, install the required dependencies:  
```bash
pip install opencv-python numpy tensorflow scikit-learn scikit-image pillow scipy torch torchvision torchaudio

```

---

## Dataset  
The dataset was sourced from:  
[Face Mask Detection Dataset](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)  
Please make sure to download and paste datasets inside the VR_PROJECT folder cause its not possible to upload it on github
It consists of images categorized into two classes:  
- **with_mask**  
- **without_mask**  

These images are preprocessed and used for training both traditional ML and deep learning models.  

---

## Methodology  

### **Part A: Traditional Machine Learning Approach**  
This approach relies on **HOG (Histogram of Oriented Gradients)** features and machine learning classifiers.  

#### **Steps**  
1. **Preprocessing**  
   - Converted images to RGB and resized them.  
   - Applied **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to enhance image contrast.  
   
2. **Feature Extraction**  
   - Extracted **HOG features** from images to capture edge and shape information.  

3. **Training ML Models**  
   - **SVM (Support Vector Machine)**: Hyperparameter tuning was done using **RandomizedSearchCV**.  
   - **Neural Network (MLPClassifier)**: Tuned with different activation functions and hidden layers.  

4. **Hyperparameter Tuning**  
   - For **SVM**, we used **Randomized Grid Search CV** with a custom grid, and the best parameters were selected.  
   - For **Neural Networks**, we also used **Randomized Grid Search CV** with a custom grid to find the optimal parameters.  

5. **Evaluation**  
   - **Accuracy Score**  
   - **Confusion Matrix**  
   - **Classification Report**  

---

## **Results (Traditional ML Approach)**  

| Approach | Accuracy |  
|----------|---------|  
| **SVM (HOG Features)** | **87.55%** |  
| **Neural Network (MLPClassifier)** | **92.06%** |  

---

### **Part B: Deep Learning Approach**  
This approach uses **Convolutional Neural Networks (CNNs)** to classify masked and unmasked faces.  

#### **Steps**  
1. **Preprocessing**  
   - Converted images to RGB and resized them to `(150, 150)`.  
   - Normalized pixel values to **[0,1]** for better training stability.  

2. **Model Architecture**  
   - **3 Convolutional Layers** with **ReLU activation** and **MaxPooling**.  
   - **Flatten Layer** followed by a **Dense Layer with Dropout** to prevent overfitting.  
   - **Final Output Layer** with **Sigmoid Activation** for binary classification.  

3. **Dataset Splitting**  
   - The dataset was split into **80% training** and **20% testing**.  
   - The model was evaluated using **both testing and validation sets**.  

4. **Training**  
   - Optimized using **Adam optimizer** with `binary_crossentropy` as the loss function.  
   - Trained for **20 epochs** with batch size **32**.  

5. **Evaluation**  
   - **Accuracy Score**  
   - **Classification Report**  
   - **Confusion Matrix**  

---

## **Results (CNN Approach)**  

**CNN Test Accuracy:** **94.45%**  
**CNN Test Loss:** **0.1561**  

### **CNN Classification Report**  

```
              precision    recall  f1-score   support

Without Mask       0.94      0.96      0.95       460
   With Mask       0.95      0.93      0.94       369

    accuracy                           0.94       829
   macro avg       0.94      0.94      0.94       829
weighted avg       0.94      0.94      0.94       829
```

### **CNN Confusion Matrix**  

```
          Predicted
Actual   No Mask   Mask
No Mask    441        19
Mask       27        342
```

### **CNN Training History**  
- **Final Training Accuracy:** **97.89%**  
- **Final Validation Accuracy:** **94.45%**  

---

## Conclusion  
- The **traditional ML approach** works well but is dependent on feature extraction.  
- The **deep learning approach (CNN)** outperforms traditional methods in accuracy and robustness.  
- CNN is recommended for **real-world applications** due to its high accuracy and adaptability.  


# Part C: Region Segmentation Using Traditional Techniques

## Dataset  
We used the dataset from this repository: [MFSD](https://github.com/sadjadrz/MFSD).  
Please make sure to download and paste datasets inside the VR_PROJECT folder cause its not possible to upload it on github
- **Input Images:** `crop_image` folder  
- **Ground Truth:** `crop_image_segmentation` folder  

## Objective  
The main goal was to segment out masks using various traditional image processing techniques.  

## Methodology  

### 1. **GrabCut for Background Removal**  
We applied **GrabCut**, a traditional non-deep learning-based segmentation method, to remove the background pixels and retain the foreground object.

### 2. **Skin Color Detection & Removal**  
We converted images into multiple color spaces:
   - **HSV (Hue, Saturation, Value)**
   - **YCrCb (Luma, Chroma Red, Chroma Blue)**
   - **Lab (Lightness, a*, b*)**  
We detected human skin pixels and blacked them out, as they are not relevant for mask detection.

### 3. **Eye and Glasses Detection using Haarcascade**  
We used OpenCV’s Haar Cascade method with the pre-trained classifier:
   - `"haarcascade_eye_tree_eyeglasses.xml"`  
Detected **eyes and glasses** were blacked out to remove distractions.

### 4. **Hair Removal Strategy**  
To handle hair interference, we blacked out the **top one-third** of each image, which proved effective across most samples.

### 5. **Binary Thresholding for Mask Segmentation**  
We applied **adaptive binary thresholding** to generate final masks, where:
   - **White** represents the mask  
   - **Black** represents the background and removed regions  

## Evaluation Metric: Intersection Over Union (IoU)  
We used **IoU (Intersection Over Union)** to measure segmentation accuracy:

$$ \text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} $$

### **Results**  
- The **average IoU score** obtained for this dataset was **~0.3**.

## Conclusion  
This approach provided a basic mask segmentation method using traditional techniques. However, the low IoU score suggests room for improvement, possibly using deep learning-based methods.


## **Part D: Deep Learning-Based Segmentation using U-Net**

This section implements **segmentation of face masks** using a **U-Net-based deep learning model**. The model is trained to **segment masks from facial images**, using a dataset containing images and their corresponding segmentation masks.

---

### **Dataset**  
The dataset consists of:  
- **Input Images:** Located in the `face_crop` folder  
- **Ground Truth Masks:** Located in the `face_crop_segmentation` folder  

Each image in the dataset has a corresponding segmentation mask that indicates the mask region.

---

### **Preprocessing**  
1. **Image Loading & Resizing**  
   - The images are resized to `(128, 128)` for uniform input size.  
   - Both images and masks are converted to tensors for training.  

2. **Normalization**  
   - Pixel values are normalized to **[0,1]** for better stability in training.  

3. **Dataset Splitting**  
   - The dataset is split into **80% training** and **20% testing**.  

---

### **Model Architecture (U-Net)**  
The segmentation model is based on **U-Net**, a popular deep learning architecture for image segmentation.  

#### **U-Net Components:**  
- **Encoder (Downsampling Path)**  
  - Consists of multiple **Double Convolution** blocks with **Batch Normalization** and **ReLU activation**.  
  - **MaxPooling** is used to reduce spatial dimensions.  

- **Bottleneck Layer**  
  - The deepest part of the network with **1024 feature channels**.  

- **Decoder (Upsampling Path)**  
  - Uses **Transposed Convolutions** to upsample feature maps.  
  - Skip connections are used to **retain spatial information** from earlier layers.  

- **Final Output Layer**  
  - A **1×1 convolution** with **Sigmoid activation** produces a binary segmentation mask.  

---

### **Training**  
1. **Loss Function**  
   - **Binary Cross Entropy with Logits Loss** is used for mask segmentation.  

2. **Optimizer**  
   - The model is trained using the **AdamW optimizer** with:  
     - Learning rate: `1e-3`  
     - Weight decay: `1e-4`  

3. **Learning Rate Scheduler**  
   - A **Cosine Annealing Scheduler** is used for adaptive learning rate adjustments.  

4. **Mixed Precision Training**  
   - **Automatic Mixed Precision (AMP)** is utilized for efficient training using **PyTorch GradScaler**.  

5. **Number of Epochs**  
   - The model is trained for **10 epochs** with a **batch size of 64**.  

---

### **Evaluation Metric: Intersection Over Union (IoU)**  
The model is evaluated using the **IoU (Intersection Over Union) score**, which measures segmentation accuracy:

$$ \text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} $$

---

### **Results (U-Net Approach)**  

**Average Training IoU:** **0.9545**  
**Average Test IoU:** **0.9574**  

### **IoU Score Progression Per Epoch**  
- **Epoch 1** - Train Loss: 0.3751, Train IoU: 0.4879
- **Epoch 2** - Train Loss: 0.1330, Train IoU: 0.8767
- **Epoch 3** - Train Loss: 0.0927, Train IoU: 0.9093
- **Epoch 4** - Train Loss: 0.0769, Train IoU: 0.9242
- **Epoch 5** - Train Loss: 0.0690, Train IoU: 0.9335
- **Epoch 6** - Train Loss: 0.0637, Train IoU: 0.9408
- **Epoch 7** - Train Loss: 0.0601, Train IoU: 0.9462
- **Epoch 8** - Train Loss: 0.0575, Train IoU: 0.9504
- **Epoch 9** - Train Loss: 0.0560, Train IoU: 0.9530
- **Epoch 10** - Train Loss: 0.0551, Train IoU: 0.9545


### **Key Findings**  
- **Unet model** performed better than these other models:  
  - **Unet average test IoU score:** **0.9574**  
  - **Traditional methods average IoU score:** **0.3**  
  - **CNN Final Validation Accuracy:** **94.45%**  
