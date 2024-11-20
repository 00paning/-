import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_excel(r"C:\Users\pszpszpsz\Desktop\california.xlsx")
from sklearn.model_selection import train_test_split, KFold

X = df.drop(['price'],axis=1)
y = df[('price')]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from hyperopt import fmin, tpe, hp
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 定义超参数搜索空间
parameter_space_rf = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 200, 300]),     # 决策树数量
    'max_depth': hp.choice('max_depth', [5, 10, 20, None]),             # 树的最大深度
    'min_samples_split': hp.uniform('min_samples_split', 0.01, 0.5),    # 分裂所需最小样本比例
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01, 0.5)       # 叶节点最小样本比例
}

# 定义目标函数
def objective(params):
    # 使用超参数创建随机森林回归模型
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=42
    )

    # 在训练集上拟合模型
    model.fit(X_train, y_train)

    # 在测试集上预测
    y_pred = model.predict(X_test)

    # 计算均方误差（MSE）
    mse = mean_squared_error(y_test, y_pred)

    # 返回MSE，Hyperopt会最小化该目标值
    return mse

# 运行超参数优化
best_params = fmin(
    fn=objective,                   # 优化的目标函数
    space=parameter_space_rf,        # 搜索空间
    algo=tpe.suggest,                # 贝叶斯优化算法
    max_evals=100                    # 最大评估次数
)

# 显示最优超参数组合
print("Best hyperparameters:", best_params)

# 使用最佳超参数组合重新训练模型
best_model_regression = RandomForestRegressor(
    n_estimators=[50, 100, 200, 300][best_params['n_estimators']],
    max_depth=[5, 10, 20, None][best_params['max_depth']],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42
)

# 在训练集上训练模型
best_model_regression.fit(X_train, y_train)
from sklearn import metrics

# 预测
y_pred_train = best_model_regression.predict(X_train)
y_pred_test = best_model_regression.predict(X_test)

y_pred_train_list = y_pred_train.tolist()
y_pred_test_list = y_pred_test.tolist()

# 计算训练集的指标
mse_train = metrics.mean_squared_error(y_train, y_pred_train_list)
rmse_train = np.sqrt(mse_train)
mae_train = metrics.mean_absolute_error(y_train, y_pred_train_list)
r2_train = metrics.r2_score(y_train, y_pred_train_list)

# 计算测试集的指标
mse_test = metrics.mean_squared_error(y_test, y_pred_test_list)
rmse_test = np.sqrt(mse_test)
mae_test = metrics.mean_absolute_error(y_test, y_pred_test_list)
r2_test = metrics.r2_score(y_test, y_pred_test_list)

print("训练集评价指标:")
print("均方误差 (MSE):", mse_train)
print("均方根误差 (RMSE):", rmse_train)
print("平均绝对误差 (MAE):", mae_train)
print("拟合优度 (R-squared):", r2_train)

print("\n测试集评价指标:")
print("均方误差 (MSE):", mse_test)
print("均方根误差 (RMSE):", rmse_test)
print("平均绝对误差 (MAE):", mae_test)
print("拟合优度 (R-squared):", r2_test)
