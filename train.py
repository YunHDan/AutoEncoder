import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import os


class dataset(torch.utils.data.Dataset):
    '''训练数据集'''

    def __init__(self):
        smis = ['C=C', 'C(=O)C', 'CNC', 'CCC(=O)C',
                'C1=CC=CC=C1', 'C#CC', 'O=C=O', 'CCCO',
                'N#N', 'C=CC=CO', 'NC(=O)C', 'OCCOCC']
        '''创建分子图片数据集，保存在Data文件夹中'''
        for i in range(len(smis)):
            mol = Chem.MolFromSmiles(smis[i])
            img = Draw.MolToImage(mol, size=(50, 50))
            img.save(f'Data/img{i}.png')
        traindata = []
        for i in range(len(smis)):
            '''将图片转成50x50x3 张量'''
            traindata.append(np.array(Image.open(f'Data/img{i}.png')))
        self.traindata = torch.tensor(np.array(traindata), dtype=torch.float32)
        self.n = len(smis)

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        return self.traindata[item]


class Autuencoder(nn.Module):
    '''自编码器'''

    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(5, 5), stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(5, 5), stride=1, padding=0),
                                    nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(5, 5), stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(5, 5), stride=1, padding=0),
                                    nn.Flatten(start_dim=2, end_dim=3),
                                    nn.Linear(1156, 1000),
                                    nn.ReLU(),
                                    nn.Linear(1000, 800),
                                    nn.ReLU())
        self.decode = nn.Sequential(nn.Linear(800, 2000),
                                    nn.ReLU(),
                                    nn.Linear(2000, 2500))

    def forward(self, input):
        out = self.encode(input)
        out = self.decode(out)
        b, c, _ = out.shape
        out = out.view(b, c, 50, 50)
        return out


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "zero"
    epochs = 10000
    batch_size = 1
    #   你可以调节epochs和batch大小，然后运行train和test，对比不同结果，感悟batches与epoches的关系
    dataloder = DataLoader(dataset(), shuffle=True, batch_size=batch_size)  # 加载数据
    auto = Autuencoder().cuda()
    optim = torch.optim.Adam(params=auto.parameters())
    Loss = nn.MSELoss()  # 损失函数
    for i in range(epochs):
        for data in dataloder:
            data = data.permute(0, 3, 1, 2).cuda()
            yp = auto(data)
            loss = Loss(yp, data)
            optim.zero_grad()
            loss.backward()
            optim.step()
    torch.save(auto, 'autoencoder.pkl')  # 保存模型