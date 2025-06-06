from sklearn.linear_model import LinearRegression
import numpy as np

F,N = map(int,input().split())
X_train = []
y_train = []

for _ in range(F):
    *features,target = map(float,input().split())
    X_train.append(features)
    y_train.append(target)
    
X_train = np.array(X_train)
y_train = np.array(y_train)

model = LinearRegression()
model.fit(X_train, y_train)

X_test =[]
for _ in range(N):
    features = list(map(float, input().split()))
    X_test.append(features)
    
X_test = np.array(X_test)
predictions = model.predict(X_test)
for prediction in predictions:
    print(f"{prediction :.2f}:")