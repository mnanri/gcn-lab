import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

train_size = 60000
test_size = 10000
batch_size = 100
epoch_num = 100

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.conv1 = nn.Conv2d(1, 32, 3) # 28x28x32 -> 26x26x32
    self.conv2 = nn.Conv2d(32, 64, 3) # 26x26x64 -> 24x24x64
    self.pool = nn.MaxPool2d(2, 2) # 24x24x64 -> 12x12x64

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

    x = self.dropout1(x)
    x = x.view(-1, 12 * 12 * 64)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    return x

def load_dataset(data_size, img_file_name, label_file_name):
  with gzip.open('./minst/'+ img_file_name, 'rb') as f:
    imgs = np.frombuffer(f.read(), np.uint8, offset=16)

  imgs = imgs.reshape(data_size,28,28)
  imgs = imgs.astype(np.float32)
  imgs /= 255

  with gzip.open('./minst/'+ label_file_name, 'rb') as f:
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

  print("==========Finish Training==========")

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
