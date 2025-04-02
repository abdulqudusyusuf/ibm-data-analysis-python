# House Sales in King County, USA

## Table of Content
- [Project Overview](#project-overview)
- [The Dataset](#prerequisites)
- [Step 1](#step-1)
- [Step 2](#step-2)
- [Step 3](#step-3)

### Project Overview
As a Data Analyst for a Real Estate Investment Trust (REIT) that aims to invest in residential real estate. The primary task is to determine the market price of houses based on a set of features. By analyzing and predicting housing prices, I will assist the REIT in making informed investment decisions. The project revolves around a dataset that contains house sale prices for King County, which includes Seattle. The dataset covers the period between May 2014 and May 2015. The original dataset was obtained from a reliable source and has been slightly modified for the purposes of this project.

The project is divided into ten questions, and I will complete using Python programming language. Throughout the project, you will work with various attributes or features of the houses, such as square footage, number of bedrooms, number of floors, and more. By analyzing these features and their relationship with the sale prices, I will build models to predict the market price of houses based on given attributes.

### The Dataset
This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015
![house1](https://github.com/user-attachments/assets/4a06a3ca-d5ca-4999-8098-9e61d2e7eb41)

```python
#After executing the below command restart the kernel and run all cells.
!pip3 install scikit-learn --upgrade --user

```
You will require the following libraries:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
%matplotlib inline

```
#### Importing Data Sets
Load the csv:

```python
file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)

```
```python
df.head()

```
![house2](https://github.com/user-attachments/assets/7e759596-1531-408c-b6a7-e0aafea0c35d)

### Question 1: 
Display the data types of each column using the function dtypes

```python
print(df.dtypes)

```

```python
Unnamed: 0         int64
id                 int64
date              object
price            float64
bedrooms         float64
bathrooms        float64
sqft_living        int64
sqft_lot           int64
floors           float64
waterfront         int64
view               int64
condition          int64
grade              int64
sqft_above         int64
sqft_basement      int64
yr_built           int64
yr_renovated       int64
zipcode            int64
lat              float64
long             float64
sqft_living15      int64
sqft_lot15         int64
dtype: object

```

We use the method describe to obtain a statistical summary of the data frame

```python
df.describe()

```
![house3](https://github.com/user-attachments/assets/00430e6b-1802-4200-bd41-d60eeaa68e90)

#### Data Wrangling
### Question 2:
Drop the columns "id" and "Unnamed: 0" from axis 1 using the method drop(), then use the method describe() to obtain a statistical summary of the data. Take a screenshot and submit it, make sure the inplace parameter is set to True

```python
df.drop("id", axis = 1, inplace = True)
df.drop("Unnamed: 0", axis = 1, inplace = True)

df.describe()

```

![house4](https://github.com/user-attachments/assets/1b53e49e-5255-47f5-8d5a-313cb1201280)

We can see we have missing values for the columns   `bedrooms` and `bathrooms`

```python
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sunumber of NaN values for the column bedrooms : 13
number of NaN values for the column bathrooms : 10

```

number of NaN values for the column bedrooms : 13
number of NaN values for the column bathrooms : 10

We can replace the missing values of the column 'bedrooms' with the mean of the column 'bedrooms' using the method replace(). Don't forget to set the inplace parameter to True


```python
mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean,inplace= True)

```
We also replace the missing values of the column 'bathrooms' with the mean of the column 'bathrooms' using the method replace(). Don't forget to set the inplace parameter top True

```python
mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)

```

```python
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

```

number of NaN values for the column bedrooms : 0
number of NaN values for the column bathrooms : 0

#### Exploratory Data Analysis
### Question 3:

Use the method `value_counts` to count the number of houses with unique floor values, use the method `.to_frame()` to convert it to a dataframe.

```python
df['floors'].value_counts().to_frame()

```
![house5](https://github.com/user-attachments/assets/8bb886e5-9510-446f-bba4-8676cbdadb50)

### Question 4
Use the function `boxplot` in the seaborn library to determine whether houses with a waterfront view or without a waterfront view have more price outliers.
```python
sns.boxplot(x="waterfront",y="price",data=df)

```
<AxesSubplot:xlabel=’waterfront’, ylabel=’price’>

![house6](https://github.com/user-attachments/assets/0371d51a-2021-48f6-8194-674e09848d29)

### Question 5
Use the function `regplot` in the seaborn library to determine if the feature `sqft_above` is negatively or positively correlated with price.
```python
sns.regplot(x="sqft_above",y="price",data=df,color='green',ci=None)
sns.regplot

```
![house7](https://github.com/user-attachments/assets/343dad47-c94e-48b9-a182-e69c6a66b121)

```python
df.corr()['price'].sort_values()
```
![house8](https://github.com/user-attachments/assets/89a4603c-6d57-410c-90b1-906bbcbe2f5f)


#### Model Development
We can Fit a linear regression model using the longitude feature 'long' and caculate the R^2.

```python
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(x,y)
```
0.00046769430149007363

### Question 6
Fit a linear regression model to predict the `'price'` using the feature `'sqft_living'` then calculate the R^2. Take a screenshot of your code and the value of the R^2.

```python
X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm
lm.fit(X,Y)
lm.score(X, Y)

```
0.49285321790379316

### Question 7
Fit a linear regression model to predict the `'price'` using the list of features:

```python
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]  
```

```python
# Then calculate the R^2
X = df[features]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
lm.score(X, Y)
```
0.6576788153088797

Create a list of tuples, the first element in the tuple contains the name of the estimator:

`'scale'`

`'polynomial'`

`'model'`

The second element in the tuple contains the model constructor

`StandardScaler()`

`PolynomialFeatures(include_bias=False)`

`LinearRegression()`

```python
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

```

### Question 8
Use the list to create a pipeline object to predict the `‘price’`, fit the object using the features in the list features, and calculate the R^2.

```python
pipe=Pipeline(Input)
pipe
X = df[features]
Y = df['price']
pipe.fit(X,Y)
pipe.score(X,Y)

```
0.7513413380708591

#### Model Evaluation and Refinement

```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")

```

done

We will split the data into training and testing sets:

```python
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

```

number of test samples: 3242
number of training samples: 18371

### Question 9
Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R² using the test data.

```python
from sklearn.linear_model import Ridge

```

```python
RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train,y_train)
RidgeModel.score(x_train,y_train)
```
0.6478759163939111

### Question 10
Perform a second order polynomial transform on both the training data and testing data. Create and fit a Ridge regression object using the training data, set the regularisation parameter to 0.1, and calculate the R² utilising the test data provided. Take a screenshot of your code and the R².

```python
pr = PolynomialFeatures(degree = 2)
x_train_pr = pr.fit_transform(x_train[features])
x_test_pr = pr.fit_transform(x_test[features])

RidgeModel1 = Ridge(alpha = 0.1) 
RidgeModel1.fit(x_train_pr, y_train)
RidgeModel1.score(x_test_pr, y_test)

```
0.7002744271145418


This project is *originally based on the* **IBM Data Analysis with Python** *course on Coursera.* *And the dataset was sourced from:*  
[*House Sales in King County, USA*](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)  


I hope these insights have provided valuable information.

Thank you













