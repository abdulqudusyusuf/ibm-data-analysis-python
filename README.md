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
