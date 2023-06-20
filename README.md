# Breast-Cancer-Survival-Prediction-Using-Random-Forest

The purpose of this project is to predict the breast cancer survival prediction whether the patient affected with Malignant or Benign.

## Objective : 

The objective of this project a real model to predict the breast cancer survival prediction.

## Abstract : 

Predicting person tumor (Malignant or Benign) based upon his tumour features i.e ,. its radius , area , smoothness , texture, perimeter using Random Forest algorithm we will predict the accuracy of the tumour features.

## Methodology : 


### Data Collection: 

   Cancer data collection begins by identifying people with cancer who have been diagnosed or received medical care in hospitals, outpatient clinics, radiology departments, doctors' offices, laboratories, surgical centers

### Data Preprocessing: 

  Some preprocessing methods, such as error correction, resolving data inconsistency, noise removal, filling null values in the collected cancer data.

### Model Development:  

   Breast cancer detection methods that have been employed utilizing ML algorithms include random forest model to implement the prediction.

### Feature Engineering: 

   Feature engineering involves selecting the relevant features that will be used to build the predictive model. It help us to identify the cancer level for their treatment.

### Model Evaluation: 

   The results show that the random forest model has better prediction performance when the data set is divided by 8 : 2. The recall rate of this model was also 1, indicating that the random forest model also correctly predicted all malignant breast cancer.



## Requirements : 

### HARDWARE REQUIREMENTS : 

    ARM64 or x64 processor; Quad-core or better recommended. 
    Minimum of 4 GB of RAM. 
    Hard disk space: Minimum of 850 MB up to 210 GB of available space
    Video card that supports a minimum display resolution of WXGA
### SOFTWARE REQUIREMENTS:

      Visual Studio Code

## Project Architecture : 

![image](https://github.com/priya672003/Breast-cancer-survival-prediction-using-random-forest/assets/81132849/2ce27d8e-09a4-4cfa-980b-b168f916e52f)

## Program : 

```python3
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn.datasets
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import  RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve, RocCurveDisplay
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
print(breast_cancer_dataset)
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)
data_frame.head()
data_frame['label'] = breast_cancer_dataset.target
data_frame.shape
data_frame.info()
data_frame.isnull().sum()
data_frame.describe()
print(data_frame.corr())
```

