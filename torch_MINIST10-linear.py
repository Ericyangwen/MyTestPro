import torch
from torchvision import transforms
from torchvision import datasets  #抽象类，只可以继承不可以实例化???
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

'''数据前期准备'''
batch_size = 64
#数据需要按照该方式预处理
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307, ),(0.3081, ))])

'''训练集以及测试集数据整理'''
train_dataset = datasets.MNIST(root= "./",train=True,download= True,transform=transform)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_dataset = datasets.MNIST(root="./",train=False,download=True,transform=transform)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

'''模型设置,除了自己设计的网络外，基本上都需要继承torch.nn.Module'''
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = torch.nn.Linear(784,512)
        self.l2 = torch.nn.Linear(512,256)
        self.l3 = torch.nn.Linear(256,128)
        self.l4 = torch.nn.Linear(128,64)
        self.l5 = torch.nn.Linear(64,10)
    
    def forward(self,x):
        x = x.view(-1,784)  #-1代表自动运算维度
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

model = Net()    
#device = torch.device("cuda:0" if torch.cuda.is_available)

'''损失函数与优化器选择'''
loss_func = torch.nn.CrossEntropyLoss()
opti = optim.SGD(model.parameters(),lr = 0.01,momentum=0.5)

'''训练部分'''
def train(epoch):
    train_loss = 0
    for bach_index,(input,target) in enumerate(train_loader,0):
        ''''''
        output = model(input)
        ''''''
        tempLoss = loss_func(output,target)
        opti.zero_grad()
        tempLoss.backward()
        opti.step()

        train_loss+=tempLoss.item()
        if bach_index % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, bach_index + 1, train_loss / 3000))
            train_loss = 0

'''测试部分'''
def test():
    correct = 0
    total = 0.0
    with torch.no_grad():
        for data in test_loader:
            images,labels = data
            output = model(images)
            _,predicted = torch.max(output.data,dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('Accuracy on test set: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()



