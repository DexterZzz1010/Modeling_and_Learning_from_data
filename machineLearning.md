```python
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# scaler = StandardScaler()
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

from sklearn.ensemble import IsolationForest
model = IsolationForest(contamination=0.02)  # contamination 参数控制被认为是离群值的比例

# 使用 fit_predict 方法进行离群值检测
outliers = model.fit_predict(X)  # 假设 features 是你的特征数据

# 找出被标记为离群值的样本的索引
outlier_indices = np.where(outliers == -1)

# 剔除离群值
X = np.delete(X, outlier_indices, axis=0)
y = np.delete(y, outlier_indices, axis=0)
```

```python
param_grid = {
    "alpha": [1e0, 1e-3],
    "kernel": ["rbf"],  # Gaussian kernel = radial basis function
    "gamma": np.logspace(-2, 2, 3)
}
kr = GridSearchCV(KernelRidge(), param_grid=param_grid)
kr.fit(Xtrain,ytrain);
kr.best_params_
```

```python
# heatmap

plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()
```

```
sns.pairplot(df)
plt.show()
```



