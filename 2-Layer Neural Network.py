import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import os
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

USE_PCA = True

if USE_PCA:
    from sklearn.decomposition import PCA
    NUM_COMPONENTS = 2
    pca = PCA(n_components=NUM_COMPONENTS)

class FruitDataset(Dataset):

    def __init__(self, data_dir, split):

        assert(split == 'train' or split == 'test')

        self.split = split
        self.data_dir = os.path.join(data_dir, 'train' if self.split == 'train' else 'test')
        self.data_types = ['Carambula', 'Lychee', 'Pear']

        if self.split == 'train':
            self.data_num_per_class = 490
        else:
            self.data_num_per_class = 166

        self.images = []
        self.labels = []
        self.labels_map = {0: 'Carambula', 1: 'Lychee', 2: 'Pear'}

        for label, type_name in enumerate(self.data_types):
            for i in range(self.data_num_per_class):
                fname = os.path.join(self.data_dir, type_name, '{}_{}_{}.png'.format(type_name, self.split, i))
                image = np.array(Image.open(fname), dtype=np.float32)[..., 0] / 255.
                self.images.append(image)
                self.labels.append(label)
        
        self.images = np.array(self.images)

        if USE_PCA:
            self.images_pca = self.get_PCA_features()

    def get_PCA_features(self):
        self.images_reshape = self.images.reshape(self.images.shape[0], -1)
        if self.split == 'train':
            return pca.fit_transform(self.images_reshape)
        else:
            return pca.transform(self.images_reshape)

    # working for indexing
    def __getitem__(self, index):
        if USE_PCA:
            return self.images_pca[index], self.labels[index]
        else:
            return self.images[index], self.labels[index]
    # return the length of our dataset
    def __len__(self):
        
        return len(self.images)

training_data = FruitDataset('D:\學校課程\機器學習\HW2\Data', 'train')
test_data = FruitDataset('D:\學校課程\機器學習\HW2\Data', 'test')

image_pca_features, label = training_data[0]
image = training_data.images[0]
# plt.show()
# plt.imshow(image, cmap='gray', vmin=0, vmax=1)
# print(training_data.labels_map[label])

# Create data loaders.
BATCH_SIZE = 32

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

#Creat 3 layer NeuralNetwork

def linear(in_dim,out_dim):
    k = 1/in_dim
    transform_matrix = np.random.uniform(-np.sqrt(k), np.sqrt(k), (out_dim,in_dim))
    return transform_matrix

def bias(in_dim,out_dim):
    k = k = 1/in_dim
    bias = np.random.uniform(-np.sqrt(k), np.sqrt(k), out_dim)
    bias_matrix = bias.reshape(1, out_dim)
    return bias_matrix
    


class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim_1,hidden_dim_2, output_dim):
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.output_dim = output_dim
        # 初始化權重和偏差
        self.w1 = linear(self.input_dim, self.hidden_dim_1) #512*2
        self.bias1 = bias(self.input_dim, self.hidden_dim_1) #1*512
        self.w2 = linear(self.hidden_dim_1, self.hidden_dim_2) #512*512
        self.bias2 = bias(self.hidden_dim_1, self.hidden_dim_2) #1*512
        self.w3 = linear(self.hidden_dim_2, output_dim) #3*512

    def relu(self,x):    
        return np.maximum(0, x)
    
    def relu_der(self,x):
        return (x > 0).astype(int)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True) 

    def forward(self, X):
        # 前向傳輸計算預測值
        self.z1 = np.dot(X, self.w1.T) # 第一層的輸出 (32*2, 2*512)=32*512
        self.a1 = self.relu(self.bias1+self.z1) # 第一層的activation输出 32*512
        self.z2 = np.dot(self.a1, self.w2.T) # 第二層的输出 (32*512, 512*3)=32*3
        self.a2 = self.relu(self.bias2+self.z2) # 第二層的activation输出 32*3
        self.z3 = np.dot(self.a2, self.w3.T) #第三層的輸出
        self.a3 = self.softmax(self.z3) #第四層的softmax輸出
        return self.z3

    def backward(self, X, y, learning_rate, batch_size):
        # 反向傳播
        
        error = self.a3 - y

        # delta3 = error
        d_weights3 = np.dot(error.T, self.a2)/batch_size
        delta2 = np.dot(error, self.w3)*self.relu_der(self.a2)
        d_weights2 = np.dot(delta2.T,self.a1)/batch_size
        d_bias2 = np.sum(delta2, axis=0, keepdims=True) / batch_size
        delta1 = np.dot(delta2, self.w2.T)*self.relu_der(self.a1)
        d_weights1 = np.dot(delta1.T, X)/batch_size
        d_bias1 = np.sum(delta1, axis=0, keepdims=True) / batch_size
        

        # 更新權重及bias
        self.w3 -= learning_rate * d_weights3
        self.w2 -= learning_rate * d_weights2
        self.bias2 -= learning_rate * d_bias2
        self.w1 -= learning_rate * d_weights1
        self.bias1 -= learning_rate * d_bias1
        return self.w1, self.w2, self.w3, self.bias1, self.bias2
    
ThreeNNmodel=NeuralNetwork(2,512,512,3)

loss_fn = nn.CrossEntropyLoss()

def train(dataloader, wt1, wt2, wt3, loss_fn, b1, b2):
        num_batches = len(dataloader)
        size = len(dataloader.dataset)
        epoch_loss = 0
        correct = 0
        weight1 = wt1
        weight2 = wt2
        weight3 = wt3
        bias1 = b1
        bias2 = b2
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            if not USE_PCA:
                # batch_size, H, W -> batch_size, H*W
                X = X.reshape(X.shape[0], -1)
                
            X_np = X.cpu().numpy()

            # Compute prediction error
            pred = ThreeNNmodel.forward(X_np)
            one_hot_y=np.eye(3)[y]

            # Backpropagation
            weight1, weight2, weight3, bias1, bias2= ThreeNNmodel.backward(X_np, one_hot_y, 1e-3, X_np.shape[0])
            pred_tensor = torch.tensor(pred)

            loss = loss_fn(pred_tensor, y)
            epoch_loss += loss.item()
            pred_maxindex = pred_tensor.argmax(dim=1, keepdim=True)
            correct += pred_maxindex.eq(y.view_as(pred_maxindex)).sum().item()

        avg_epoch_loss = epoch_loss / num_batches
        avg_acc = correct / size

        return avg_epoch_loss, avg_acc

def test(dataloader, wt1, wt2, wt3, loss_fn, b1, b2):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    epoch_loss = 0
    correct = 0
    weight1 = wt1
    weight2 = wt2
    bias1 = b1
    bias2=b2

    for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            if not USE_PCA:
                X = X.reshape(X.shape[0], -1)
           
            X_np = X.cpu().numpy()
            
            # Compute prediction error
            pred = ThreeNNmodel.forward(X_np)
            pred_tensor = torch.tensor(pred)
            
            loss = loss_fn(pred_tensor, y)
            epoch_loss += loss.item()
            pred_maxindex = pred_tensor.argmax(dim=1, keepdim=True)
            correct += pred_maxindex.eq(y.view_as(pred_maxindex)).sum().item()

    avg_epoch_loss = epoch_loss / num_batches
    avg_acc = correct / size

    return avg_epoch_loss, avg_acc

epochs = 20
lost_list = []
for epoch in range(epochs):
    train_loss, train_acc = train(train_dataloader, ThreeNNmodel.w1, ThreeNNmodel.w2, ThreeNNmodel.w3, loss_fn, ThreeNNmodel.bias1, ThreeNNmodel.bias2)
    test_loss, test_acc = test(test_dataloader, ThreeNNmodel.w1, ThreeNNmodel.w2, ThreeNNmodel.w3, loss_fn, ThreeNNmodel.bias1, ThreeNNmodel.bias2)
    print(f"Epoch {epoch + 1:2d}: Train_Loss = {train_loss:.4f} Train_Acc = {train_acc:.2f} Test_Loss = {test_loss:.4f} Test_Acc = {test_acc:.2f}")
    lost_list.append(train_loss)

#plot loss curve

print("Done!")
epoch = np.arange(0, epochs)  # 產生數字1到20的一維陣列
epoch_mat = epoch.reshape(epochs, 1)
plt.plot(epoch_mat, lost_list, 'b-', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#plot decision region

all_train_data = DataLoader(training_data, shuffle=True)

list_train = []
label_train = []
for X, y in all_train_data:
    list_train.append(X)
    one_hot_y=np.eye(3)[y]
    label_train.append(one_hot_y)
train_array = np.concatenate(list_train, axis=0)
label_array = np.concatenate(label_train, axis=0)
label_mat1 = label_array.reshape(-1,3)
# print(train_array)
# print(label_mat)
# print(label_mat.shape)

all_test_data = DataLoader(test_data, shuffle=True)

list_test = []
label_test = []
for X, y in all_test_data:
    list_test.append(X)
    one_hot_y=np.eye(3)[y]
    label_test.append(one_hot_y)
test_array = np.concatenate(list_test, axis=0)
label_array = np.concatenate(label_test, axis=0)
label_mat2 = label_array.reshape(-1,3)

def decision_region(model, X, y):

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max()+1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max()+1
        xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        Z = model.forward(np.c_[xx1.ravel(), xx2.ravel()])
        pred_label = np.argmax(Z, axis=1)
        Z = pred_label.reshape(xx1.shape)
        cmap_cont = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap_cont)
        
        labels = np.unique(np.argmax(y, axis=1))   
        markers = ['o', 's', '^']
        colors = ['r', 'g', 'b']
        names = ['Carambula', 'Lychee', 'Pear']
        for i, label in enumerate(labels):
            plt.scatter(X[np.argmax(y, axis=1) == label, 0], X[np.argmax(y, axis=1) == label, 1],
                        c=colors[i], marker=markers[i], label=names[i], alpha=0.8, s = 20)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Decision Boundary')
        plt.legend()
        plt.show()

d1=decision_region(ThreeNNmodel, train_array, label_mat1)
d2=decision_region(ThreeNNmodel, test_array, label_mat2)
