# Emotion recognition with facial landmarks

This repository shows the work done for our final project in the course 'Applied Machine Learning in Python'.

Our job was to tackle the FER-2013 dataset available on Kaggle. 

First we extracted facial landmarks in the [Emotion Recognition](EmotionRecognition.ipynb) notebook.
Secondly the data was processed and 15 facial features were calculated in the [Facial Features Calculation](FacialFeaturesCalculation.ipynb) notebook.
Next we trained a linear multiclass model on the produced data in the0 [Linear Multiclass Classification](LinearMulticlass.ipynb) notebook.
Lastly we set up two CNN's and processed the FER-2013 dataset directly in the [Convolutional Neural Networks](CNN.ipynb) notebook. 

## Prerequisits
Use the applied_ml environment from the course linked as a subrepo. 
Additionally use 
```
conda install conda-forge::imutils
conda install anaconda::cmake
conda install conda-forge::dlib
conda install conda-forge::opencv
conda install conda-forge::tensorflow
```

## Results

Our models achieve the following performance on :

| Model name         | Accuracy         
| ------------------ |---------------- |
| Multi Linear Classification   |     47%        |
| CNN Model A   |      47,3%        |    
| CNN Model B   |     47,0%         |


