# Breast-Cancer-Survival-Prediction-Using-Random-Forest

The purpose of this project is to predict the breast cancer survival prediction whether the patient affected with Malignant or Benign.

## Objective : 

The objective of this project a real model to predict the breast cancer survival prediction.

## Abstract : 

Predicting person tumor (Malignant or Benign) based upon his tumour features i.e ,. its radius , area , smoothness , texture, perimeter using Random Forest algorithm we will predict the accuracy of the tumour features.

## Methodology : 


### Data Collection: 

 -   Cancer data collection begins by identifying people with cancer who have been diagnosed or received medical care in hospitals, outpatient clinics, radiology departments, doctors' offices, laboratories, surgical centers 

### Data Preprocessing: 

 - Some preprocessing methods, such as error correction, resolving data inconsistency, noise removal, filling null values in the collected cancer data.

### Model Development:  

  - Breast cancer detection methods that have been employed utilizing ML algorithms include random forest model to implement the prediction.

### Feature Engineering: 

  - Feature engineering involves selecting the relevant features that will be used to build the predictive model. It help us to identify the cancer level for their treatment.

### Model Evaluation: 

   - The results show that the random forest model has better prediction performance when the data set is divided by 8 : 2. The recall rate of this model was also 1, indicating that the random forest model also correctly predicted all malignant breast cancer.



## Requirements : 

### HARDWARE REQUIREMENTS : 

   - ARM64 or x64 processor; Quad-core or better recommended. 
   - Minimum of 4 GB of RAM. 
   - Hard disk space: Minimum of 850 MB up to 210 GB of available space
   - Video card that supports a minimum display resolution of WXGA
### SOFTWARE REQUIREMENTS:

     - Visual Studio Code

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
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X.shape, X_train.shape, X_test.shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(30,)),
                          keras.layers.Dense(20, activation='relu'),
                          keras.layers.Dense(2, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
categorical_columns = X_train.select_dtypes(include='object').columns
print(categorical_columns)
numeric_columns = X_train.select_dtypes(exclude='object').columns
print(numeric_columns)
numeric_features = Pipeline([
    ('handlingmissingvalues',SimpleImputer(strategy='median')),
    ('scaling',StandardScaler(with_mean=True))
])
print(numeric_features)
categorical_features = Pipeline([
    ('handlingmissingvalues',SimpleImputer(strategy='most_frequent')),
    ('encoding', OneHotEncoder()),
    ('scaling', StandardScaler(with_mean=False))
])

print(categorical_features)
processing = ColumnTransformer([
    ('numeric', numeric_features, numeric_columns),
    ('categorical', categorical_features, categorical_columns)
])

processing
def prepare_model(algorithm):
    model = Pipeline(steps= [
        ('processing',processing),
        ('pca', TruncatedSVD(n_components=3, random_state=12)),
        ('modeling', algorithm)
    ])
    model.fit(X_train, y_train)
    return model
def prepare_confusion_matrix(algo, model):
    print(algo)
    plt.figure(figsize=(12,8))
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    plt.show()
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 

def prepare_classification_report(algo, model):
    print(algo+' Report :')
    pred = model.predict(X_test)
    print(classification_report(y_test, pred))

def prepare_roc_curve(algo, model):
    print(algo)
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, thresholds = roc_curve(y_test,  y_pred_proba)
    roc_auc = auc(fpr, tpr)
    curve = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    curve.plot()
    plt.show()
algorithms = [ ('Random Forest calssifier', RandomForestClassifier()) ]
trained_models = []
model_and_score = {}
print(model_and_score)
for index, tup in enumerate(trained_models):
    prepare_confusion_matrix(tup[0], tup[1])
for index, tup in enumerate(trained_models):
    prepare_roc_curve(tup[0], tup[1])
plt.figure(figsize=(20,10))
corr = data_frame.corr()
sns.heatmap(corr, annot=True, cmap="pink")
input_data=eval(input())
# change the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardizing the input data
input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
print(prediction)

prediction_label = [np.argmax(prediction)]
print(prediction_label)

if(prediction_label[0] == 1):
  print('The tumor is Malignant')

else:
  print('The tumor is Benign')

```

## Output : 

![image](https://github.com/priya672003/Breast-cancer-survival-prediction-using-random-forest/assets/81132849/be38a5d2-9094-4790-94bd-ce934f385f18)


![image](https://github.com/priya672003/Breast-cancer-survival-prediction-using-random-forest/assets/81132849/0fd93bf6-5711-47ff-9d1a-4e1f948a6dc7)



## Result : 

   The results of our study demonstrate the utility of the Random Forest algorithm in breastcancer survival prediction. The high accuracy, sensitivity, and specificity, along with the interpretability and robustness of the model, make it a valuable tool for clinicians and researchers in the field of breast cancer.


## Conclusion : 

In conclusion, accurate survival prediction for breast cancer is crucial in providing patients with appropriate treatment and support. By analyzing various factors such as tumor size, grade, lymph node involvement, and biomarkers, medical professionals can estimate the likelihood of survival and make informed decisions regarding treatment plans.


## References : 

- Huang, M. L., Hung, C. S., Chiang, C. T., & Wu, T. H. (2012). Breast cancer diagnosis using a hybrid machine learning approach. IEEE Transactions on
Computational Biology and Bioinformatics, 9(6), 1859-1869.

- Al-Masni, M. A., Al-Ahmadi, T., Al-Enezi, R., & Al-Jaroodi, J. (2019). A comparative study of machine learning algorithms for breast cancer detection and
diagnosis. International Journal of Advanced Computer Science and Applications, 10(6), 275-281.

-  S. M. Prabha and S. Sumathi. "Breast Cancer Prediction Using K-Nearest Neighbor and Decision Tree Algorithms." International Journal of Pure and Applied
Mathematics, vol. 119, no. 12, 2018, pp. 1565-1572.

-  Sisodia DS, Bhattacharjee D, Panigrahi BK. A Decision Tree Classifier for Breast Cancer Detection. In: 2018 International Conference on Advances in Computing, Communication Control and Networking (ICACCCN). IEEE; 2018. p. 266-269.

 - Fong, A., & Alobaidli, S. (2017). Predicting breast cancer using machine learning
techniques: A systematic review. In Journal of Cancer Therapy (Vol. 8, No. 10,



