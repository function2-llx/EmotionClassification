import torch
import torchvision
import vectorizer
from torch.utils.data import Dataset, DataLoader

label_len = 8
# batch_size = 8
# uniform_size = 100

class EmotionDataset(Dataset):
    def __init__(self, filename, transform, uniform_size=None):
        self.data = []
        self.uniform_size = uniform_size
        
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                text, label = [list(map(int, x.split(' '))) for x in line.split('\t')]
                        
                # label = [label.index(max(label))]
                # tot += len(text)

                if uniform_size != None and len(text) > uniform_size:
                    import random
                    # generate a random subsequence of 'text'
                    p = sorted(random.sample([i for i in range(len(text))], uniform_size))  
                    text = [text[i] for i in p]

                self.data.append((text, label))
                
            # print('text average len:', tot // len(lines))

        self.transform = transform

    def __len__(self):
       return len(self.data) 

    def __getitem__(self, idx):
        return self.transform((*self.data[idx], self.uniform_size))
            
def prepare(batch_size, uniform_size):
    compose =  torchvision.transforms.Compose([
        vectorizer.Vectorizer()
    ])

    train_set = EmotionDataset('data/data.train', compose, uniform_size=uniform_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # uniformed_train_set = EmotionDataset('data/data.train', compose, uniform_size=500)

    verify_set = EmotionDataset('data/data.verify', compose, uniform_size=uniform_size)
    # verify_loader = DataLoader(verify_set, batch_size=1)
    
    test_set = EmotionDataset('data/data.test', compose, uniform_size=uniform_size)
    # test_loader = DataLoader(test_set, batch_size=1)

    return train_set, train_loader, verify_set, test_set
