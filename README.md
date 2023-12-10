# Image Classification Model Training and Testing

Note: The code written in this repo is from the view of a Google Colab and Google Drive user who can run this code using this tool.
## High-level description/presentation of the project
The rise of Machine Learning (ML) and Computer Vision (CV) has transformed multiple sectors, leveraging visual data interpretation. This revolution finds application in diverse fields such as autonomous vehicles, medical imaging, and facial recognition. Convolutional Neural Networks (CNNs) play a pivotal role in processing image data, excelling in tasks like object recognition. Yet, challenges persist, including adaptability to variations, limited data for training, and model interpretability, necessitating robust CNN-based CV systems. Our project aims to build a resilient CV model using CNN for precise classification of colorectal and prostate cancer tissues, as well as animal face recognition. The project involves dataset processing, model architecture design, training, and analysis, striving to enhance CNN-based visual analysis and advance the field of CV, offering a solid base for future research and practical applications. To achieve this, we have selected three datasets, each characterized by distinct image formats and classes. The project is divided into two tasks, the first involving training a CNN model on Dataset 1, and the second applying pre-trained CNN encoders and ImageNet CNN encoders to Dataset 2 and Dataset 3 for feature extraction. The data will be processed, cleaned, and split into training and testing sets, with performance evaluation based on several metrics, including accuracy and confusion matrices. The team will conduct a comprehensive analysis, utilizing t-SNE visualization and implementing both supervised and unsupervised learning techniques to assess and compare results. Our study is expected to validate deep learning models, facilitating knowledge transfer across applications and providing practical insights for image classification and deep learning applications.

## Requirements to run your Python code (libraries, etc)
The parts of this project are dissected into several Jupyter Notebooks. For them to run successfully, you would need to run them individually on Google Colab. The reason the project is dissected is due to limited resources and processing power which would result in collapsing or crashing the entire project. You would have to install Numpy, Pandas, Matplotlib, PyTorch, and Sci-Kit Learn to successfully run the code as available. Please ensure that the datasets are on your Google Drive with the appropriate path name to bring the best possible results.


## Instruction on how to train/validate your model
If you are using Google Colab, then the code for training the model is already there. It would certainly make it faster with a higher processing unit like GPU to train the model which is on the Task 1 Jupyter Notebook. All the data is imported using the PyTorch and Scikit-Learn libraries. 

## Instructions on how to run the pre-trained model on the provided sample test dataset
As part of running the provided sample test dataset, using Google Colab would help if you used Google Drive where the sample dataset is uploaded and an appropriate path is added. Consequently, the pre-trained model would function easily on Google Colab.

## Your source code package in PyTorch


## Description on how to obtain the Dataset from an available download link
Here are the three dataset links for download. It would really help if you could upload them to your personal Google Drive:
Dataset 1: Colorectal Cancer Classification 
Link: https://1drv.ms/u/s!AilzKc-njjP7mN0NOZvxl0TPAUxmig?e=K0TpeX

Dataset 2: Prostate Cancer Classification
Link: https://1drv.ms/u/s!AilzKc-njjP7mN0M_LjB5xeAydDsrA?e=0obzsx

Dataset 3: Animal Faces Classification
Link: https://1drv.ms/u/s!AilzKc-njjP7mN0LqoRZvUYONY9sbQ?e=wxWbip
