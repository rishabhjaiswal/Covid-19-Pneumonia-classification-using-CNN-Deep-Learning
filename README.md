🧠 COVID-19 Image Classification using Pretrained Deep Learning Models
This project focuses on training and testing pretrained deep learning models (ResNet50 and VGG16) to classify chest X-ray images into multiple categories (Normal, Pneumonia, COVID-19).
It applies transfer learning, data augmentation, and fine-tuning techniques to achieve high diagnostic accuracy.

🚀 Project Overview
The goal of this project is to implement and compare ResNet50 and VGG16 convolutional neural networks for medical image classification.
Using Google Colab and Kaggle datasets, the models were trained to detect respiratory diseases from X-ray images.
Highlights:
• 📊 Transfer learning with pretrained ResNet50 and VGG16
• ⚙️ Data preprocessing and augmentation using ImageDataGenerator
• 🧩 Optimization with Adam optimizer and Categorical Crossentropy loss
• 🔍 Evaluation through precision, recall, F1-score, and confusion matrix

🧰 Libraries Used
• TensorFlow / Keras
• Matplotlib
• NumPy
• Seaborn
• Scikit-learn
• OS, Warnings

📊 Data & Preprocessing
• Data sourced from Kaggle (Chest X-Ray dataset)
• Images resized to 320×320, normalized, and converted to RGB
• Training/validation split: 80% / 20%
• Applied augmentation techniques:
• Random shear, zoom, and shift
• Normalization and centering
• Mean-zero scaling for pixel intensity

⚙️ How to Run
This project is designed to run on Google Colab.
1. Open the notebook:
image_classification.ipynb

2. Run the first three cells to set up your environment.
3. After running Cell #3, insert your kaggle.json file.
• To get it:
Go to Kaggle → Account → API → Create New API Token
(this will download kaggle.json)
• Upload this file to your Colab environment.
• (A copy of kaggle.json is also included in the project ZIP file.)
4. Continue running the remaining cells to:
• Load dataset from Kaggle
• Train the models
• Evaluate and compare results

📈 Results Summary
Model Training Accuracy Test Accuracy Key Notes
ResNet50 94% 96% Best performing model with high recall
VGG16 90% 92% Smooth convergence, minimal overfitting
ResNet50 outperformed VGG16 with slightly higher accuracy and recall, making it better suited for medical image diagnostics.

🔮 Future Work
• Extend to larger and more balanced datasets
• Experiment with deeper architectures and longer training epochs
• Deploy as a web-based or edge AI application
