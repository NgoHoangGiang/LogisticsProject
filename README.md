# Analyzing NHG's business situation and proposing a logistics strategy:

## 1. Problem Statements:

### About NHG:
- The company provides food and beverage products in the Northern provinces of Vietnam.
- Headquartered in Hanoi, with many offices throughout the North.
### Director David Ngo's challenges:
- Difficulty in monitoring the company's business situation (output, revenue, profit,...) in a rapidly growing market.
- Often receive inaccurate or exaggerated information from branch managers when asking about the business situation over the phone.
- Receiving different Excel files from managers, making it difficult to see the business situation as a whole.
### Director David Ngo's wishes:
- Want to have a simple, clear view of the company's business activities.
- Need an overview that illustrates the truth from the data.
- Desire to use actual data to make decisions to increase revenue and profits.
### Solution:
- Use Power BI to visualize data, helping David Ngo easily grasp the company's business situation.
- Focus on key indicators such as output, revenue, and profit to come up with appropriate Logistics strategies for the company.
- Build a linear regression model to predict the company's business situation.

## 2. Data Mining:

### Information about data columns:
- Information about time: ‘DATE_KEY’ và ‘MONTH_KEY’;
- Information about locations, stores and distribution centers: 'OUTLET_CODE', 'LOCATION', 'OUTLET_CITY', 'OUTLET_DISTRICT', 'OUTLET_WARD', 'URBAN_RURAL' 'OUTLET_LATITUDE', 'OUTLET_LONGITUDE', 'DC_CODE', 'DC_CITY', 'DC_DISTRICT', 'DC_WARD', 'DC_LATITUDE', 'DC_LONGITUDE';
- Information about types of business: 'BUSINESSTYPEL1', 'BUSINESSTYPEL2', 'BUSINESSTYPEL3';
- Information about product: 'DIVISION', 'SUB_CATEGORY', 'BRAND', 'BRANDY', 'VARIANT', 'STD_SKU';
- Information about financial: 'REVENUE', 'SALES_QTY', 'PROFIT';
- Information about promotion cost: 'TOTAL_PROMOTION_COST', 'POSM_PROMOTION_COST', 'FG_PROMOTION_COST', 'DISCOUNT_COST', 'FG_PROMOTION_QTY'
### AIMS Web:
- **Purpose:** To unlock business insights previously invisible to the sales team, for decision support and data automation, reducing manual data collection time;
- **Stakeholders:** Sales Director; Marketing team; Customer service team; Data and analysis team; IT.
- **Final result:** Automated dashboards provide quick and up-to-date sales insights to support data-driven decision-making.
- **Criteria for success:** Dashboard explores store business insights with existing data; The logistics team can make better decisions based on the company's business situation.

## 3. Data processing:

### With Python:
**- Import required libraries:**

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error, r2_score

from scipy import stats


**- Reading data:**

df = pd.read_csv('north_size.csv')

df.head()
![image](https://github.com/user-attachments/assets/7611bcb0-3c9f-4d81-8977-b01ad157931b)

df.info()

![image](https://github.com/user-attachments/assets/2c01a432-74d4-4442-9384-d6589c452d01)

df.shape

![image](https://github.com/user-attachments/assets/618df394-3496-44d4-83d3-289d2b4006f4)


**- Checking for missing values ​​in the dataset:**

df.isnull().sum()

![image](https://github.com/user-attachments/assets/abf37213-c27a-4fe7-934e-ec1dfb05e03e)


**- Checking for duplicated values ​​in the dataset:**

df.duplicated().sum()

![image](https://github.com/user-attachments/assets/55b61948-b1b8-41da-a0b8-de852e3dc942)


**- Checking for inconsistent values ​​in the dataset:**

df.select_dtypes(include=[np.number]).lt(0).sum()

![image](https://github.com/user-attachments/assets/8b4dcad4-0c6b-4eb1-a266-40dfee6f6a56)


**- Replace the value in the 'LOCATION' column of the dataset:**

df['LOCATION'] = df['LOCATION'].replace('Đường Chợ','Đường chợ')


**- Export .csv file after cleaning the dataset:**

df.to_csv('Dulieudetrucquanhoa.csv',encoding='utf-8', index=False)

## 4. Building and analyzing dashboard:

## 5. Predict business situation through linear regression model:

**- Convert qualitative variables to quantitative variables:**

- from sklearn.preprocessing import LabelEncoder

- le = LabelEncoder()

- df['URBAN_RURAL'] = le.fit_transform(df['URBAN_RURAL'])

![image](https://github.com/user-attachments/assets/23f9c720-a4eb-4bc3-b64a-aa829ceb1d51)

Helps the algorithm process the variable 'URBAN_RURAL' after entering the model

Helps the model capture the influence of the variable 'URBAN_RURAL' on the target variable 'PROFIT'

Reduce errors, increase model accuracy

**- Eliminate outliers in the dataset:**

![image](https://github.com/user-attachments/assets/b248b196-420b-415e-af9c-f9518958e586)

Outlier numbers can greatly affect the regression coefficient, leading to an inaccurate model.

Eliminating outliers helps the model more accurately reflect the relationship between variables while improving model performance.

Helps the model to be less affected by abnormal values, increasing model stability.

The model will have better prediction ability when it is not affected by data points that are too different.

**- Building a multivariate linear regression model:**

![image](https://github.com/user-attachments/assets/6805181e-32bd-4e93-824f-dd69e291d9b3)



**- Correlation matrix between dependent variable and independent variable:**


**- Chia tỷ lệ dữ liệu:**

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

**- Tách dữ liệu để Train và Test:**

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

from sklearn.linear_model import Ridge

pipeline = Pipeline(steps=[('regressor', Ridge())])  # Linear Regression model


## Insights của dự án: 

