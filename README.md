# Analyzing NHG's business situation and proposing a logistics strategy:

## Problem Statements:

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

## Data Mining:

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

## Data processing:

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


**- Đọc dữ liệu:**

df = pd.read_csv('north_size.csv')

df.head()

df.info()

df.shape


**- Kiểm tra giá trị missing trong bộ dữ liệu:**

df.isnull().sum()


**- Kiểm tra giá trị duplicate trong bộ dữ liệu:**

df.duplicated().sum()


**- Kiểm tra các giá trị không phù hợp trong bộ dữ liệu:**

df.select_dtypes(include=[np.number]).lt(0).sum()


**- Thay thế giá trị trong cột 'LOCATION' của bộ dữ liệu:**

df['LOCATION'] = df['LOCATION'].replace('Đường Chợ','Đường chợ')


**- Xuất file .csv sau khi làm sạch bộ dữ liệu:**

df.to_csv('Dulieudetrucquanhoa.csv',encoding='utf-8', index=False)

### Với Power BI:

### Xây dựng mô hình hồi quy trên Python:
**- Mã hóa biến định tính sang biến định lượng:**

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['URBAN_RURAL'] = le.fit_transform(df['URBAN_RURAL'])


**- Loại bỏ số ngoại lai:**

Q1 = df.select_dtypes(include=[np.number]).quantile(0.25)

Q3 = df.select_dtypes(include=[np.number]).quantile(0.75)

IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR


df = df[~((df.select_dtypes(include=[np.number]) < lower_bound) | (df.select_dtypes(include=[np.number]) > upper_bound)).any(axis=1)]


**- Trực quan hóa ma trận tương quan giữa các biến:**
![image](https://github.com/user-attachments/assets/a63c92e7-a471-49c4-8dc1-1eca483700f7)

**- Xây dựng biến cho mô hình hồi quy:**

X = df[['REVENUE', 'SALES_QTY', 'TOTAL_PROMOTION_COST', 'DISCOUNT_COST', 'URBAN_RURAL']]

y = df['PROFIT']


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

