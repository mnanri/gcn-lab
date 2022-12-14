import gzip
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

train_size = 60000
test_size = 10000
batch_size = 100
epoch_num = 20

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.conv1 = nn.Conv2d(1, 32, 3) # 26x26x32
    self.conv2 = nn.Conv2d(32, 64, 3) # 24x24x64
    self.pool = nn.MaxPool2d(2, 2) # 12x12x64

    self.dropout1 = nn.Dropout2d()
    self.fc1 = nn.Linear(12 * 12 * 64, 128)

    self.dropout2 = nn.Dropout2d()
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.pool(x)

    x = x.view(-1, 12 * 12 * 64)
    # print("after x.view: ",x.size()) # torch.Size([100, 9216])
    x = self.dropout1(x)
    # print("after x.dropout1: ",x.size()) # torch.Size([100, 64, 12, 12])
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    # print("after x.dropout2: ",x.size()) # torch.Size([100, 128])
    x = self.fc2(x)
    # print("after x.fc2: ",x.size()) # torch.Size([100, 10])
    return x

def load_dataset(data_size, img_file_name, label_file_name):
  with gzip.open('./mnist/'+ img_file_name, 'rb') as f:
    imgs = np.frombuffer(f.read(), np.uint8, offset=16)

  imgs = imgs.reshape(data_size, 1, 28, 28)
  imgs = imgs.astype(np.float32)
  imgs /= 255

  for i in range(int(3*data_size/5)):
    tmp_img = np.zeros((28,28))
    if i%3 == 0:
      for j in range(28):
        for k in range(28):
          tmp_img[27-k][j] = imgs[i][0][j][k]
    elif i%3 == 1:
      for j in range(28):
        for k in range(28):
          tmp_img[k][27-j] = imgs[i][0][j][k]
    else:
      for j in range(28):
        for k in range(28):
          tmp_img[27-j][27-k] = imgs[i][0][j][k]
    imgs[i][0] = tmp_img

  with gzip.open('./mnist/'+ label_file_name, 'rb') as f:
    labels = np.frombuffer(f.read(), np.uint8, offset=8)

  X = torch.tensor(imgs, dtype=torch.float32)
  y = torch.tensor(labels, dtype=torch.int64)
  data_set = torch.utils.data.TensorDataset(X, y)

  print("Complete!")
  return data_set


def main():
  # load data
  train_set = load_dataset(train_size, 'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
  test_set = load_dataset(test_size, 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

  print("train set size: >>> ", len(train_set))
  print("test set size: >>>", len(test_set))

  # set device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # set model
  model = Net().to(device)

  # set model as train mode
  model.train()

  # get data
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

  # set optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  # set loss function
  criterion = nn.CrossEntropyLoss()

  # define struct for history of loss and accuracy
  history = {
    "epoch": [],
    "train_loss": [],
    "test_loss": [],
    "test_acc": []
  }

  print("==========Start Training==========")
  start = time.time()
  # learning loop part by epoch
  for epoch in range(epoch_num):
    train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
      inputs, labels = inputs.to(device), labels.to()

      # initialize optimizer
      optimizer.zero_grad()

      # forward
      outputs = model(inputs)
      loss = criterion(outputs, labels)

      # backward
      loss.backward()

      # update weights
      optimizer.step()

      # add loss
      train_loss += loss.item()
      if i % 10 == 9:
        progress_bar = '['+('='*((i+1)//10))+(' '*((train_size//100-(i+1))//10))+']'
        print('\repoch: {:d} loss: {:.3f}  {}'.format(epoch + 1, loss.item(), progress_bar), end="  ")

    print('\repoch: {:d} loss: {:.3f}'.format(epoch + 1, train_loss / (train_size / batch_size)), end="  ")
    history["epoch"].append(epoch+1)
    history["train_loss"].append(train_loss / (train_size / batch_size))

    correct = 0
    total = 0
    batch_num = 0
    loss = 0
    with torch.no_grad():
      for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        batch_num += 1

    history["test_acc"].append(correct/total)
    history["test_loss"].append(loss/batch_num)
    endstr = ' '*max(1,(train_size//1000-39))+"\n"
    print('Test Accuracy: {:.2f} %%'.format(100 * float(correct/total)), end='  ')
    print(f'Test Loss: {loss/batch_num:.3f}',end=endstr)

    if epoch == 0:
      print("Time for 1 epoch: {:.2f} sec".format(time.time()-start))

  end = time.time()
  print("==========Finish Training==========")
  print("Total time: {:.2f} sec".format(end-start))

  '''
  fig1, ax1 = plt.subplots()
  ax1.set_title('Loss [train=blue, test=orange]')
  ax1.set_xlabel('epoch')
  ax1.set_ylabel('loss')
  ax1.grid()
  ax1.plot(history['epoch'], history["train_loss"], color="orange",label="train")
  ax1.plot(history['epoch'], history["test_loss"], color="blue",label="test")
  fig1.tight_layout()
  fig1.savefig('./mnist/loss_cnn.png')
  fig1.show()

  fig2, ax2 = plt.subplots()
  ax2.set_title('Accuracy')
  ax2.set_xlabel('epoch')
  ax2.set_ylabel('accuracy')
  ax2.grid()
  ax2.plot(history['epoch'], history["test_acc"], color="black")
  fig2.tight_layout()
  fig2.savefig('./mnist/accuracy_cnn.png')
  fig2.show()
  '''

  # output final result
  correct = 0
  total = 0

  with torch.no_grad():
    for inputs, labels in test_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)))

main()
