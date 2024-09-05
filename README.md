# Dự án Logistics:
## Xác định vấn đề:

  NHG là một công ty vận chuyển hàng hóa sản phẩm trên toàn quốc. Hiện tại, NHG có trụ sở chính tại thành phố Hà Nội và sau đó họ có rất nhiều văn phòng trên khắp 3 miền cả nước. 

  David Ngo - giám đốc của NHG đang phải đối mặt với nhiều thách thức trong việc quản lý và tối ưu hóa hoạt động logistic. Mặc dù công ty có trụ sở chính tại Hà Nội và mạng lưới phân phối rộng khắp 3 miền của Việt Nam, ông David gặp khó khăn trong việc theo dõi và quản lý các hoạt động vận chuyển hàng hóa tại khu vực miền Bắc, nơi NHG phân phối sản phẩm cho các cửa hàng.

  Thứ nhất, thông tin và hành vi mua sắm của khách hàng (bao gồm ngày bán, địa điểm, sản phẩm, kênh bán hàng,...) được lưu trữ trong các tệp Excel rời rạc tại các cửa hàng trên khắp miền Bắc. Sự phân tán này gây khó khăn lớn trong việc tổng hợp và phân tích dữ liệu một cách chính xác và nhanh chóng. Thứ hai, ông David đang phải dựa vào các báo cáo từ các nhà quản lý chi nhánh để hiểu về hoạt động kinh doanh. Tuy nhiên, các thông tin này thường không đáng tin cậy do các nhà quản lý có xu hướng "tô hồng" tình hình thực tế, dẫn đến việc ông David không có cái nhìn chính xác về tình hình kinh doanh. Thứ ba, việc phải xử lý 69 tệp Excel riêng lẻ gây ra sự chậm trễ trong việc ra quyết định và dẫn đến sự mơ hồ trong việc nhận định các xu hướng bán hàng. Điều này làm cho ông David không thể có được cái nhìn toàn diện về hoạt động logistic và doanh số bán hàng.

  Điều này làm cho ông ấy thất vọng, ông ấy không thể nhìn bức tranh một cách hoàn chỉnh nhất. David Ngo chỉ quan tâm đến việc có được những thông tin chi tiết đơn giản, dễ hiểu nhưng những gì mà các nhà quản lý đưa ra là quá nhiều và anh ấy không thể tiêu thụ nhiều số lượng như vậy (A picture is worth a thousand words). Thứ nhất, cần phải triển khai một hệ thống quản lý dữ liệu tập trung, nơi tất cả các thông tin bán hàng từ các chi nhánh và cửa hàng có thể được nhập vào một cơ sở dữ liệu duy nhất. Điều này sẽ cho phép ông Bhavin có được cái nhìn toàn diện và kịp thời về hoạt động kinh doanh của công ty. Thứ hai, triển khai Power BI hoặc các công cụ phân tích dữ liệu trực quan khác để tự động hóa quá trình tổng hợp và phân tích dữ liệu. Power BI sẽ giúp ông Bhavin dễ dàng truy cập và xem các biểu đồ, báo cáo, giúp ông hiểu rõ hơn về các xu hướng bán hàng, hiệu suất của từng khu vực, và các yếu tố ảnh hưởng đến doanh thu. Thứ ba, cài đặt hệ thống báo cáo định kỳ tự động để gửi các thông tin chi tiết về hoạt động logistic và doanh số bán hàng. Điều này sẽ giúp ông Bhavin có thể theo dõi tình hình kinh doanh một cách liên tục mà không cần phụ thuộc vào các báo cáo thủ công từ các nhà quản lý.

## Khám phá dữ liệu:

### Với Python:
- Nhập thư viện cần thiết:
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
- Đọc dữ liệu:
df = pd.read_csv('north_size.csv')
df.head()
df.info()
df.shape
- Kiểm tra giá trị missing trong bộ dữ liệu:
df.isnull().sum()
- Kiểm tra giá trị duplicate trong bộ dữ liệu:
df.duplicated().sum()
- Kiểm tra các giá trị không phù hợp trong bộ dữ liệu:
df.select_dtypes(include=[np.number]).lt(0).sum()
- Thay thế giá trị trong cột 'LOCATION' của bộ dữ liệu:
df['LOCATION'] = df['LOCATION'].replace('Đường Chợ','Đường chợ')
- Xuất file .csv sau khi làm sạch bộ dữ liệu:
df.to_csv('Dulieudetrucquanhoa.csv',encoding='utf-8', index=False)

### Với Power BI:
### Xây dựng mô hình hồi quy trên Python:
## Insights của dự án: 

