import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import numpy as np
import warnings


class Food_Dataset(Dataset):
    def __init__(self, anno_path, folder, dtype='tensor'):
        assert dtype in ['numpy', 'tensor', 'Tensor', 'img'], 'data type not supported'
        super(Food_Dataset, self).__init__()
        self.folder = folder
        self.dtype = dtype
        with open(anno_path, mode='rt', encoding='utf8') as f:
            self.annotation = f.readlines()

    def __getitem__(self, index):
        img_path, label = self.annotation[index].split()
        label = torch.tensor(int(label), dtype=torch.long)
        img = Image.open(self.folder+img_path).convert('RGB')
        if self.dtype == 'img':
            return img, label
        if self.dtype == 'numpy':
            return img, label
        transformer = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        img = transformer(img)
        img = img[:3]
        return img, label

    def __len__(self):
        return len(self.annotation)



'''
data shape: 696*696*3
'''
def get_trainloader(batch_size, shuffle=True, num_workers=4, drop_last=True):
    train_path = r'./data/ISIA_Food500/retrieval_dict/train.txt'
    folder_path = r'./data/ISIA_Food500/images/'

    loader =  DataLoader(
        Food_Dataset(train_path, folder_path, 'tensor'), 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        )
    return loader


def get_testloader(batch_size, shuffle=False, num_workers=4, drop_last=True):
    test_path = r'./data/ISIA_Food500/retrieval_dict/test.txt'
    folder_path = r'./data/ISIA_Food500/images/'

    loader =  DataLoader(
        Food_Dataset(test_path, folder_path, 'tensor'), 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
        )
    return loader


def check_img():
    train_path = r'./data/ISIA_Food500/retrieval_dict/test.txt'
    folder = r'./data/ISIA_Food500/images/'
    annotation = []
    corrupted_images = []
    warnings.filterwarnings('error')
    with open(train_path, 'r', encoding='utf8') as f:
        annotation = f.readlines()
    warnings.filterwarnings('error', category=UserWarning)

    for line in annotation:
        img_path, label = line.split()
        try:
            img = Image.open(folder+img_path).convert('RGB')
            img.close()
            
        except:
            print(img_path)
            corrupted_images.append(img_path)

    with open(r'./data/ISIA_Food500/retrieval_dict/corrupt.txt', 'w', encoding='utf8') as f:
        for line in corrupted_images:
            f.write(line+'\n')



if __name__ == '__main__':
    check_img()
