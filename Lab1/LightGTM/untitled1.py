import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.color_palette("husl", 10)
from scipy import stats
from scipy.stats import skew
from scipy.special import boxcox1p  # 数据转换
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
pd.set_option('display.max_columns', None)  #能看到所有列
pd.set_option('display.float_format', lambda x: '%.3f' % x)

train = pd.read_csv('D:\WeChat Files\WeChat Files\wxid_gardxi0vlbi822\FileStorage\File/2023-10/train.csv')
train_ID = train['Id']
train.drop("Id", axis = 1, inplace = True)

#Data Analysis
train.head()
train.describe()
sns.regplot(x = 'GrLivArea', y = 'SalePrice', color = 'black', data = train)
plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



# Transformation
f, ax = plt.subplots(figsize=(16,8))
sns.distplot(train['SalePrice'], color="red")
ax.set(xlabel="SalePrice")
ax.set(title="Distribution of SalePrice Feature")
plt.show()
         
train['SalePrice'] = np.log1p(train['SalePrice'])
f, ax = plt.subplots(figsize=(16,8))
sns.distplot(train['SalePrice'], color="black")
ax.set(ylabel="Price Frequencies")
ax.set(xlabel="SalePrice")
ax.set(title="Distribution of SalePrice Feature")
plt.show()

f, ax = plt.subplots(figsize=(16,8))
sns.distplot(train['LotFrontage'], color="red")
ax.set(xlabel="LotFrontage")
ax.set(title="Distribution of Lot Frontage Feature")
plt.show()

train['LotFrontage'] = np.log1p(train['LotFrontage'])
f, ax = plt.subplots(figsize=(16,8))
sns.distplot(train['LotFrontage'], color="black")
ax.set(xlabel="LotFrontage")
ax.set(title="Distribution of LotFrontage Feature")
plt.show()


f, ax = plt.subplots(figsize=(16,8))
sns.distplot(train['LotArea'], color="red")
ax.set(xlabel="LotArea")
ax.set(title="Distribution of Lot Area Feature")
plt.show()

train['LotArea'] = np.log1p(train['LotArea'])
f, ax = plt.subplots(figsize=(16,8))
sns.distplot(train['LotArea'], color="black")
ax.set(xlabel="LotArea")
ax.set(title="Distribution of LotArea Feature")
plt.show()

f, ax = plt.subplots(figsize=(16,8))
sns.distplot(train['1stFlrSF'], color="red")
ax.set(xlabel="1stFlrSF")
ax.set(title="1stFlrSF")
plt.show()

train['1stFlrSF'] = np.log1p(train['1stFlrSF'])
f, ax = plt.subplots(figsize=(16,8))
sns.distplot(train['1stFlrSF'], color="black")
ax.set(xlabel="1stFlrSF")
ax.set(title="Distribution of 1stFlrSF Feature")
plt.show()

f, ax = plt.subplots(figsize=(16,8))
sns.distplot(train['GrLivArea'], color="red")
ax.set(xlabel="GrLivArea")
ax.set(title="GrLivArea")
plt.show()

train['GrLivArea'] = np.log1p(train['GrLivArea'])
f, ax = plt.subplots(figsize=(16,8))
sns.distplot(train['GrLivArea'], color="black")
ax.set(xlabel="GrLivArea")
ax.set(title="Distribution of GrLivArea Feature")
plt.show()
# Missing value
missing_values = train.isnull().sum()
missing_columns = missing_values[missing_values > 0]
print(missing_columns)
train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
train["Alley"] = train["Alley"].fillna("Not Available")
train["MasVnrType"] = train["MasVnrType"].fillna("Not Available")
train["MasVnrArea"] = train["MasVnrArea"].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    train[col] = train[col].fillna('Not Available')
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
train["FireplaceQu"] = train["FireplaceQu"].fillna("Not Available")
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    train[col] = train[col].fillna('Not Available')
train["PoolQC"] = train["PoolQC"].fillna("Not Available")
train["Fence"] = train["Fence"].fillna("Not Available")
train["MiscFeature"] =train["MiscFeature"].fillna("Not Available")
train['GarageYrBlt']=train['GarageYrBlt'].fillna(0)

train['MSSubClass'] = train['MSSubClass'].apply(str)
train['OverallCond'] = train['OverallCond'].astype(str)
train['YrSold'] = train['YrSold'].astype(str)
train['MoSold'] = train['MoSold'].astype(str)

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
le = LabelEncoder()
for i in cols:
    train[i] = le.fit_transform(train[[i]])
    
numeric_feats = train.dtypes[train.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for i in skewed_features:
    train[i] = boxcox1p(train[i], lam)

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a DataFrame with the independent variables
X = train[['GrLivArea', 'LotFrontage', 'LotArea','1stFlrSF','SalePrice',]]
X['Intercept'] = 1
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)
    
#Heat map
n_train = train.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(14, 14))
sns.heatmap(n_train.corr(), annot=True, annot_kws={"size": 6}, cmap='coolwarm', linewidths=.5, square=True)

correlation_matrix = n_train.corr()
cols = correlation_matrix.nlargest(16, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
plt.figure(figsize=(12, 12))
heatmap = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.title('Top 16 Correlation Matrix')# 选出15组与sale price相关性高的数据
plt.show()





train['SaleType'] = train['SaleType'].fillna(train['SaleType'].mode()[0])
train['MSSubClass'] = train['MSSubClass'].fillna("Not Available")
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    train[col] = train[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    train[col] = train[col].fillna(0)
train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])
train["Functional"] = train["Functional"].fillna("Typ")
train['KitchenQual'] = train['KitchenQual'].fillna(train['KitchenQual'].mode()[0])
train['Exterior1st'] = train['Exterior1st'].fillna(train['Exterior1st'].mode()[0])
train['Exterior2nd'] = train['Exterior2nd'].fillna(train['Exterior2nd'].mode()[0])

train = pd.get_dummies(train, columns=['MasVnrType', 'GarageType', 'MiscFeature'])

categorical_features = ['MSZoning', 'LandContour', 'Utilities', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                        'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                        'Foundation', 'Heating', 'Electrical', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']
# train = pd.get_dummies(train, columns=categorical_features)

#######LGBM
data = pd.read_csv("D:\WeChat Files\WeChat Files\wxid_gardxi0vlbi822\FileStorage\File/2023-10/train.csv")
# 选择与 Sale Price 相关性很高的 15 列数据
selected_columns = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'Fireplaces', 'GarageYrBlt', 'BsmtFinSF1', 'LotFrontage', 'SalePrice']
data = data[selected_columns]
# 数据预处理
data = data.fillna(0)  # 处理缺失值
# 数据划分为训练集和测试集
X = data.drop(columns=['SalePrice'])  # 特征
y = data['SalePrice']  # 目标变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 创建和训练LightGBM回归模型
params = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'max_depth':10,
    'n_estimators': 1000
}# 初始模型训练
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train)
# 预测
y_pred = model.predict(X_train)
# 评估模型性能
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2) Score: {r2}")
# 超参数调整与网格搜索
model = lgb.LGBMRegressor()
# 定义超参数搜索空间
param_grid = {
    'learning_rate': [0.04, 0.05, 0.055, 0.06, 0.065, 0.07, 0.1, 0.2],
    'max_depth': [5, 7, 8, 9, 10, 11, 12, 14, 15, 16],
    'num_leaves': [18, 19, 20, 21, 30, 40]
}
# 使用Grid Search进行超参数优化
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print("Grid search finished.")
# 打印最佳参数
best_params = grid_search.best_params_
print("best_params:", best_params)
# 使用最佳参数的模型进行训练
best_model = lgb.LGBMRegressor(**best_params)
best_model.fit(X_train, y_train)
# 预测并评估模型性能
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2) Score: {r2}")
# 可视化特征重要性
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train)
feature_importance = model.feature_importances_
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names, feature_importance))
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
top_n = 5  # 前5个重要特征
top_features = [x[0] for x in sorted_feature_importance[:top_n]]
top_feature_importance = [x[1] for x in sorted_feature_importance[:top_n]]
plt.figure(figsize=(10, 6))
plt.barh(top_features, top_feature_importance)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Top 5 Feature Importance')
plt.gca().invert_yaxis()
plt.show()
# 打印每个特征的重要性
for feature, importance in zip(feature_names, feature_importance):
    print(f"Feature: {feature}, Importance: {importance}")
# 创建柱状图
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # 反转y轴，以按重要性降序排列
plt.show()
# 获取最重要的特征
most_important_feature = sorted_feature_importance[0][0]
# 绘制外样本预测与最重要特征的散点图
plt.figure(figsize=(8, 6))
plt.scatter(X_test[most_important_feature], y_pred, c='blue', label='Predicted')
plt.xlabel(most_important_feature)
plt.ylabel('Predicted Sale Price')
plt.title(f'Predicted Sale Price vs. {most_important_feature}')
plt.legend()
plt.show()




#####LGBM
data = pd.read_csv('D:\WeChat Files\WeChat Files\wxid_gardxi0vlbi822\FileStorage\File/2023-10/train.csv')
# 选择与 Sale Price 相关性很高的 15 列数据
selected_columns = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'Fireplaces', 'GarageYrBlt', 'BsmtFinSF1', 'LotFrontage', 'SalePrice']
data = data[selected_columns]
# 数据预处理
data = data.fillna(0) 
X = train.drop(columns=['SalePrice'])
y = train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
params = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 1000
}
model = LGBMRegressor(**params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2) Score: {r2}")
#grid
model = lgb.LGBMRegressor()
param_grid = {
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [5,10,15],
    'min_child_samples': [10, 20, 30]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
X = train.drop(columns=['SalePrice'])
y = train['SalePrice']
grid_search.fit(X, y)
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# 使用最佳参数的模型进行训练
best_model = lgb.LGBMRegressor(**best_params)
best_model.fit(X, y)

# 在训练数据上进行交叉验证
cross_val_scores = cross_val_score(best_model, X, y, cv=3, scoring='neg_mean_squared_error')
mse_mean = -cross_val_scores.mean()
print("Mean Cross-Validation MSE:", mse_mean)

y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
print(f"Root Mean Square Error (RMSE): {rmse}")

feature_importance = best_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
top_10_features = feature_importance_df.head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=top_10_features)
plt.title('Top 10 Feature Importance')
plt.show()

most_important_feature = sorted_feature_importance[0][0]
# 绘制外样本预测与最重要特征的散点图
plt.figure(figsize=(8, 6))
plt.scatter(X_test[most_important_feature], y_pred, c='blue', label='Predicted')
plt.xlabel(most_important_feature)
plt.ylabel('Predicted Sale Price')
plt.title(f'Predicted Sale Price vs. {most_important_feature}')
plt.legend()
plt.show()
