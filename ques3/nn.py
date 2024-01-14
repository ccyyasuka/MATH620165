import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score
import time
class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        # X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.softmax(X)

        return X


# # load IRIS dataset
# dataset = pd.read_csv('dataset/iris.csv')

# # transform species to numerics
# dataset.loc[dataset.species == 'Iris-setosa', 'species'] = 0
# dataset.loc[dataset.species == 'Iris-versicolor', 'species'] = 1
# dataset.loc[dataset.species == 'Iris-virginica', 'species'] = 2


# train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:4]].values,
#                                                     dataset.species.values, test_size=0.8)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
start_time = time.time()
# wrap up with Variable in pytorch
train_X = Variable(torch.Tensor(train_X).float())
test_X = Variable(torch.Tensor(test_X).float())
train_y = Variable(torch.Tensor(train_y).long())
test_y = Variable(torch.Tensor(test_y).long())


net = Net()

criterion = nn.CrossEntropyLoss()  # cross entropy loss

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(10001):
    optimizer.zero_grad()
    out = net(train_X)
    loss = criterion(out, train_y)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print('number of epoch')
        print(epoch)
        print("loss")
        print(loss.data.item())

predict_out = net(test_X)
_, predict_y = torch.max(predict_out, 1)
end_time = time.time()  # 记录程序结束运行时间
print('cost %f second' % (end_time - start_time))
# print('prediction accuracy')
# print(accuracy_score(test_y.data, predict_y.data))
# print('macro precision')
# print(precision_score(test_y.data, predict_y.data, average='macro'))
# print('micro precision')
# print(precision_score(test_y.data, predict_y.data, average='micro'))
# # print 'macro recall', recall_score(test_y.data, predict_y.data, average='macro')
# # print 'micro recall', recall_score(test_y.data, predict_y.data, average='micro')
# print(recall_score(test_y.data, predict_y.data, average='macro'))
# print(recall_score(test_y.data, predict_y.data, average='micro'))
accuracy = accuracy_score(test_y.data, predict_y.data)
cm = confusion_matrix(test_y.data, predict_y.data)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
f1 = f1_score(test_y.data, predict_y.data, average='macro')
print("F1-Score:", f1)