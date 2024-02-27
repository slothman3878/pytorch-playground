import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

# 3 fully connected layers with sigmoid activation
class Network1(nn.Module):
  def __init__(self):
    super(Network1, self).__init__()
    self.fc1 = nn.Linear(784, 200)
    self.fc2 = nn.Linear(200, 100)
    self.fc3 = nn.Linear(100, 10)
    
  def forward(self, x):
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = nn.functional.sigmoid(x)
    x = self.fc2(x)
    x = nn.functional.sigmoid(x)
    x = self.fc3(x)
    x = nn.functional.sigmoid(x)
    return x
  
# 3 fully connected layers with ReLU activation
class Network2(nn.Module):
  def __init__(self):
    super(Network2, self).__init__()
    self.fc1 = nn.Linear(784, 200)
    self.fc2 = nn.Linear(200, 100)
    self.fc3 = nn.Linear(100, 10)
    
  def forward(self, x):
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = nn.functional.relu(x)
    x = self.fc2(x)
    x = nn.functional.relu(x)
    x = self.fc3(x)
    x = nn.functional.relu(x)
    return x
  
# 1 convlutional layer
# 2 fully connected layer
class Network3(nn.Module):
  def __init__(self, dropout=0.2):
    super(Network3, self).__init__()
    self.conv = nn.Conv2d(1, 20, 5) # convolutional layer
    self.fc1 = nn.Linear(20 * 12 * 12, 100) # fully connected layer 1
    self.fc2 = nn.Linear(100, 10) # fully connected layer 2
    self.dropout = nn.Dropout(p=dropout)
    
  def forward(self, x):
    x = self.conv(x)
    x = nn.functional.relu(x)
    x = nn.functional.max_pool2d(x, 2)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = self.dropout(x)
    x = nn.functional.relu(x)
    x = self.fc2(x)
    x = nn.functional.log_softmax(x, dim=1)
    return x