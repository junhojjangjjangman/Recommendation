import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import surprise
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
import os

from surprise.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import joblib
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

electronics_data = pd.read_csv("C:/Users/15/Desktop/DataSet/[Dataset]_Module11_(Recommendation).csv")
names = ['userId', 'productId','Rating','timestamp']
##읽어온 csv컬럼값을 0번째 행으로 받기
col_list = electronics_data.columns.tolist() #읽어온 csv파일 컬럼 값을 list형태로 바꿔줌
electronics_data.columns = names #기존 읽어온 csv파일 컬럼 값을 names의 값들로 바꿔줌
df = pd.DataFrame([col_list], columns=names) #새로운 pandas를 만들때 컬럼 값을 names로 0번째 행에 기존 컬럼값으로 추가
df = pd.concat([df, electronics_data], ignore_index=True)# 나머지 데이터들 이어 붙이기
df = df.astype({'Rating':'float64','timestamp':'int64'})
# print(df.head())
# print(df.head(20))

# print(df.shape)
df1 = df.head(1048576)

# print(df.dtypes)
# print(df1.info())
#print(df1['Rating'].describe()) # 전체 데이터 통계 같은거 ? 표시
# print("Minimum rating is: ",df1['Rating'].min()) #min값 확인
# print("Maximum rating is: ",df1['Rating'].max()) #max값 확인

#print("Number of missing values across columns: \n", df1.isnull().sum()) # 누락된 값 확인


print(df1.groupby('Rating')['Rating'].sum())
rating_groups = df1.groupby('Rating')['Rating'].count()
rating_groups.plot(kind='bar', figsize=(10, 6))
plt.xticks(rotation=0)
plt.show() #막대그래프 그리기

print("Total data ")
print("-"*50)
print("\nTotal no of ratings :", df1['Rating'].count())
print("Total No of Users   :", df1['userId'].unique())
print("Total No of products  :", df1['productId'].unique())


if 'timestamp' in df1.columns:
    # df1.drop(['timestamp'], axis= 1)
    del df1['timestamp']

# print(df1.columns) #삭제됐는지 확인
no_of_rated_products_per_user = df1.groupby('userId')['Rating'].count().sort_values(ascending=False)
print(no_of_rated_products_per_user.head())

quantiles = no_of_rated_products_per_user.quantile(np.arange(0,1.01,0.01), interpolation='higher')

plt.figure(figsize=(10,10))
plt.title("Quantiles and their Values")
quantiles.plot()
# 차이가 0.05인 분위수를 찾습니다.
plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], c='orange', label="quantiles with 0.05 intervals")

# 차이가 0.25인 분위수도 구해 보겠습니다.
plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25], c='m', label = "quantiles with 0.25 intervals")
plt.ylabel('No of ratings by user')
plt.xlabel('Value at the quantile')
plt.legend(loc='best')
# plt.show()

print('\n No of rated product more than 50 per user : {}'.format(sum(no_of_rated_products_per_user >= 50)) )
print('\n No of rated product more than 50 per user : {}\n'.format(sum(no_of_rated_products_per_user >= 60)) )

new_df=df1.groupby("productId").filter(lambda x:x['Rating'].count() >=50)

no_of_ratings_per_product = new_df.groupby('productId')['Rating'].count().sort_values(ascending=False)

fig = plt.figure(figsize=plt.figaspect(.5))
ax = plt.gca()
plt.plot(no_of_ratings_per_product.values)
plt.title('# RATINGS per Product')
plt.xlabel('Product')
plt.ylabel('No of ratings per product')
ax.set_xticklabels([])
sum_Product =new_df.groupby('productId')['Rating'].sum().sort_values(ascending=False)
AVG_Product_Rating = new_df.groupby('productId')['Rating'].mean().sort_values(ascending=False)
# plt.show()

print(AVG_Product_Rating.head())
print(no_of_ratings_per_product.head())


PRRDF = pd.concat([AVG_Product_Rating, no_of_ratings_per_product], axis=1)# 나머지 데이터들 이어 붙이기
print(PRRDF.head())

# print(PRRDF['Rating'].max())