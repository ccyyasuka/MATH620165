from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import f1_score
import time
# 加载数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
start_time = time.time()
# 创建带特征缩放的SVM分类器管道
clf = make_pipeline(StandardScaler(), SVC())
# clf = make_pipeline(SVC())
# 设置要调优的参数
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [1, 0.1, 0.01, 0.001],
    'svc__kernel': ['rbf', 'poly', 'sigmoid']
}

# 使用网格搜索进行参数调优
grid = GridSearchCV(clf, param_grid, refit=True, verbose=3)

# 训练模型
grid.fit(X_train, y_train)

# 找到并显示最佳参数
best_params = grid.best_params_
print("Best parameters found: ", best_params)

# 使用最佳参数的模型进行预测
y_pred = grid.predict(X_test)


# 训练模型
# clf.fit(X_train, y_train)


# # 使用最佳参数的模型进行预测
# y_pred = clf.predict(X_test)

end_time = time.time()  # 记录程序结束运行时间
print('cost %f second' % (end_time - start_time))
# 评估模型
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
f1 = f1_score(y_test, y_pred, average='macro')
print("F1-Score:", f1)