from preprocessing import MakeOneHot
from load_data import load_mnist
from nn.model import Model
from nn.layer import FC, Softmax, ReLU, Dropout
from nn.optimizer import Adam, SGD
from nn.loss import cross_entropy
from nn.metrix import accuracy


(X_train, y_train), (X_test, y_test) = load_mnist()
X_train = X_train.reshape((X_train.shape[0], -1)) / 255
X_test = X_test.reshape((X_test.shape[0], -1)) / 255

transformer = MakeOneHot()
y_train = transformer.fit_transform(y_train)
y_test = transformer.transform(y_test)

model = Model()
model.add(FC(500, input_shape=784))
model.add(ReLU())
model.add(Dropout(0.5))
model.add(FC(150))
model.add(ReLU())
model.add(Dropout(0.5))
model.add(FC(50))
model.add(ReLU())
model.add(Dropout(0.5))
model.add(FC(10))
model.add(Softmax())

model.compile(Adam(eta=0.01), cross_entropy, accuracy)

model.fit(X_train, y_train, max_iter=10, batch_size=2000)

print("train acc: {:.2f}%".format(model.score(X_train, y_train)))
print("test acc: {:.2f}%".format(model.score(X_test, y_test)))
