# Code for creating a spiral dataset from CS231n
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchmetrics
from sklearn.model_selection import train_test_split
from torch import nn
import Helper_functions as hp


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
plt.show()

# Turn data into tensors

X = torch.from_numpy(X).type(torch.float) # features as float32
y = torch.from_numpy(y).type(torch.LongTensor) # labels need to be of type long

# Create train and test splits
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size=0.2,
                                                 random_state=RANDOM_SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

X_train,X_test,y_train,y_test = X_train.to(device),X_test.to(device),y_train.to(device),y_test.to(device)

print(X_train.shape)
print(y_train[:10])

torch.manual_seed(42)
torch.cuda.manual_seed(42)

class modelV0(nn.Module):
    def __init__(self):
        super(modelV0, self).__init__()
        self.seq = nn.Sequential(
                    nn.Linear(in_features=2,out_features=16),
                    nn.ReLU(),
                    nn.Linear(in_features=16,out_features=32),
                    nn.ReLU(),
                    nn.Linear(in_features=32,out_features=3)
                )
    def forward(self,x):
        return self.seq(x)


my_model = modelV0().to(device)
logits = my_model(X_train)
print(f"Logits: {logits[:10]}")
print(f"Probs: {torch.softmax(logits[:10],dim=1)}")
print(f"Pred: {torch.argmax(logits[:10],dim=1)}")

loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(params=my_model.parameters(),
                        lr=0.1)
acc_fn = torchmetrics.Accuracy(task="multiclass",num_classes=3).to(device)
epochs=10000

for epoch in range(epochs):
    my_model.train()

    logits = my_model(X_train)
    prob = torch.softmax(logits,dim=1)
    pred = torch.argmax(prob,dim=1)

    acc_fn.update(pred,y_train)
    acc = acc_fn.compute()

    loss = loss_fn(prob,y_train)
    optim.zero_grad()
    loss.backward()
    optim.step()

    my_model.eval()

    with torch.inference_mode():

        logits = my_model(X_test)
        prob = torch.softmax(logits,dim=1)
        pred = torch.argmax(prob,dim=1)

        acc_fn.update(pred,y_test)
        val_acc = acc_fn.compute()

        val_loss = loss_fn(prob,y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Acc: {acc:.2f}% Loss: {loss:.4f} | Acc_val: {val_acc:.2f}% Loss_val: {val_loss:.4f}")


plt.figure(figsize=(10,7))

plt.subplot(1,2,1)
hp.plot_decision_boundary(my_model,X_train,y_train)
plt.title("Train")

plt.subplot(1,2,2)
hp.plot_decision_boundary(my_model,X_test,y_test)
plt.title("Test")

plt.show()