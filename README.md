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

- Helps the algorithm process the variable 'URBAN_RURAL' after entering the model

- Helps the model capture the influence of the variable 'URBAN_RURAL' on the target variable 'PROFIT'

- Reduce errors, increase model accuracy

![image](https://github.com/user-attachments/assets/23f9c720-a4eb-4bc3-b64a-aa829ceb1d51)


**- Eliminate outliers in the dataset:**

- Outlier numbers can greatly affect the regression coefficient, leading to an inaccurate model.

- Eliminating outliers helps the model more accurately reflect the relationship between variables while improving model performance.

- Helps the model to be less affected by abnormal values, increasing model stability.

- The model will have better prediction ability when it is not affected by data points that are too different.

![image](https://github.com/user-attachments/assets/b248b196-420b-415e-af9c-f9518958e586)


**- Building a multivariate linear regression model:**

- X is a set of input variables (Independent Variables) including columns: 'REVENUE'; 'SALES_QTY'; 'TOTAL_PROMOTION_COST'; 'DISCOUNT_COST'; 'URBAN_RURAL'

- y is the target variable (Dependent Variable): 'PROFIT'

![image](https://github.com/user-attachments/assets/6805181e-32bd-4e93-824f-dd69e291d9b3)


**- Correlation matrix between dependent variable and independent variable:**

![image](https://github.com/user-attachments/assets/b68456c2-9fa4-4854-b357-c89b98e9b39f)

**- Normalize and split the dataset:**

- Using StandardScaler from the sklearn.preprocessing library to normalize the data set X. This method helps give features on the same scale (mean = 0, standard deviation = 1). This is especially important when Machine Learning algorithms approach data scaling, such as Linear Regression, SVM, KNN,...

- Useing train_test_split from sklearn.model_selection library to split the data set into 2 parts: training set (70%) and testing set (30%). The parameter random_state = 1 ensures the reproducibility of data splitting.

![image](https://github.com/user-attachments/assets/414689d2-3335-4846-9bf6-e5fe19a8dca3)

- Build a Ridge regression model to limit multicollinearity in the model

- Find the best value for the Alpha parameter by testing different values ​​through cross-validation

- Determine the optimal Alpha value of 0.1, helping to optimize the Ridge model to get the best prediction results.

![image](https://github.com/user-attachments/assets/09f4b666-db72-48aa-af15-9480e8c95837)

**- Model predictions:**

![image](https://github.com/user-attachments/assets/3599c68e-811c-4a31-bb6e-67393e01f20b)


**- Checking model's fitting:**

![image](https://github.com/user-attachments/assets/473d8110-dcb0-4133-9a80-f8c09ed5959f)

**- Plot of Residuals vs. Fitted values:**

![image](https://github.com/user-attachments/assets/8ee5c066-be59-498c-ad94-ad40f5a6a23d)

**- Plot of Distribution Residuals:**

![image](https://github.com/user-attachments/assets/d2c66d28-7a92-4951-994e-2dc39d229750)

**- Plot of Actual vs. Predicted values:**

![image](https://github.com/user-attachments/assets/d5269f08-dfa2-4929-9ccf-efec3edaeef1)

## 6. Conclusion: 

### Insights from the dashboard and recommendations for the Logistics department:

- Distribution by province/city: Hanoi city is the main area generating revenue, accounting for the majority of profits in the region. From there, it is proposed to increase warehouse and transportation resources in Hanoi to quickly meet customer needs. At the same time, optimize transportation routes between cities to reduce delivery time.

- Prioritize shipping for effective sales channels: Off Premise sales channel accounts for the entire store's revenue. Therefore, it is necessary to focus maximum logistics resources on this channel. At the same time, design the fastest and most cost-effective route to serve Off Premise retail channels throughout the region.

- "Off-road" sales location: is the top priority: "Off-street" is the sales location that brings in nearly 85% of total revenue throughout the region. Therefore, focus on building effective delivery routes, ensuring goods are always available at "Off-road" sales points. The amount allocated to other locations can be adjusted to avoid wasting resources.

- Warehouse management based on key products: "Kokomi Noodles", "Omachi Base" and "Nam Ngu" are products that need to be prioritized for storage because of high consumer demand from customers. Therefore, warehouses should maintain large inventories of these key products. It is necessary to monitor commodity consumption trends to avoid shortages and at the same time reduce inventory of products with lower demand.

- Strengthen logistics capacity at the end of the month: The company's revenue tends to increase at the end of the month. Therefore, it is necessary to plan to increase distribution of goods at this time, ensuring that goods are always ready for peak sales at the end of the month.

### Rigde regression model:

- Intercept: The intercept is 1820.24, showing that when all independent variables have a value of 0, the predicted value of the dependent variable is 1820.24. This demonstrates that there is a fixed level of influence before considering the impact of other factors.

- Beta Coefficients (Coefficients):

+ 1805.66: The Revenue variable has a strong positive influence on the Profit variable.

+ -302.88: The Sales_Qty variable has a negative (negative) influence on the Profit variable, meaning that when the value of this variable increases, the value of the dependent variable tends to decrease.

+ 36.83: The Total_Promotion_Cost and Discount_Cost variables have a positive impact on the Profit variable but are not as strong as the Revenue variable.

+ -2.34: The Urban_rural variable has an opposite impact on the Profit variable, but the level of influence is smaller than the Sales_Qty variable.

- MSE, MAE, R² results: MSE (Mean Squared Error) and MAE (Mean Absolute Error) have quite large values ​​(MSE = 241712.39, MAE = 241712.39), showing that the model has a relatively large error. However, the R² value = 0.92 is quite high, showing that the Ridge regression model explains 92% of the variation in the data, the remaining 8% is explained by other factors that cannot be included in the model. image. Overall, the model has quite good accuracy with high R², but attention should be paid to data points with large residuals.

- Plot of Residuals vs. Fitted values: + This chart tests the model's assumptions about residuals. The residuals do not appear to be randomly distributed, there is some tendency to follow a conical pattern, which may imply that the model has heteroscedasticity problems. This is a sign to consider adjusting the model or applying measures to overcome this phenomenon.

- Plot of Distribution Residuals: This chart shows the distribution of residuals around the value 0. The residuals are most concentrated around the value 0, proving that the model has relative predictive ability. accurate with the majority of forecasts not deviating too much. However, there are some large residuals on either side, which could be a sign of outliers or a not-quite-fit model.

- Plot of Actual vs. Predicted values: This chart shows the close correlation between actual value and predicted value. The red line (45 degree line) represents the model's perfection, and points distributed quite close to this line show that the model's predictions are quite accurate.
