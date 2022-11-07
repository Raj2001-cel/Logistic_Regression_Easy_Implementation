import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def accuracy(y_pred, y_test):
    count  = 0.0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            count =  count + 1
    return count/len(y_test)

class LogisticRegression():

    def __init__(self, lr, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        print(X.shape)
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = []
        for pred in y_pred:
            if pred <= 0.5:
                class_pred.append(0)
            else:
                class_pred.append(1)
        return class_pred



X = [[10],[25],[32],[35],[55],[58],[60],[70],[65],[30],[90],[50]]
X = np.array(X)

y = [1,1,1,1,0,0,0,0,0,1,0,0]
X_train = X[0:len(X)-2]
y_train = y[0:len(y)-2]

X_test =  X[len(X)-2:]
y_test = y[len(y)-2:]


clf = LogisticRegression(lr=0.001)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print(y_pred,y_test)
acc = accuracy(y_pred, y_test)
print(acc)