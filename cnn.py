import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from vectorizer import vec_len as in_channels
from data import label_len

batch_size = 64
uniform_size = 306
conv_size = 3
pool_size = 2
h_size = 128

epoch_lim = 10

c = [256, 66]

assert (uniform_size - conv_size + 1) % pool_size == 0
assert ((uniform_size - conv_size + 1) // pool_size - conv_size + 1) % pool_size == 0

class CNN(nn.Module):
    def __init__(self, c1, c2):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, conv_size),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(c1, c2, conv_size),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )

        self.fc = nn.Sequential(
            nn.Linear(((uniform_size - conv_size + 1) // pool_size - conv_size + 1) // pool_size * c2, h_size),
            nn.ReLU(),
            nn.Linear(h_size, label_len),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x = self.layer0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        # print(x.size())
        x = self.fc(x)
        return x


cnn = CNN(*c)

import weight

# cnn.apply(weight.weight_init)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=1e-4)

def evalutate(text):
    # print(text.size())
    output = cnn(text.view(1, *text.size()))

    return output, output[0].topk(1)[1]

def test(test_set):
    with torch.no_grad():
        n = len(test_set)
        cnt = 0

        labels = []
        outputs = []

        # for text, label in test_set:
        for i, data in enumerate(test_set.data):
            text_raw, label_all = data
            # print(label_all)
            text, label = test_set.transform((text_raw, label_all, uniform_size))
            output, predict = evalutate(text)
            # print(nn.functional.softmax(output))
            labels.append(label)
            outputs.append(output)
            if label == predict:
                cnt += 1

            from sys import stderr
            stderr.write('\rtesting: %f             ' % (i * 100 / n))

        return criterion(torch.cat(outputs), torch.cat(labels)), cnt * 100 / n

save_path = 'models/'

def save(name='cnn.pt'):
    import os
    try:
        os.mkdir(save_path)
    except FileExistsError:
        pass
    
    torch.save(cnn.state_dict(), os.path.join(save_path, 'cnn.pt'))
    print('save success')

def load():
    import os
    try:
        cnn.load_state_dict(torch.load(os.path.join(save_path, 'cnn.pt')))
        # print('load success')
    except Exception as e:
        print(e)
        print('load fail')

def train_all(train_loader, verify_set, test_set):
    epoch = 0
    n = len(train_loader)
    min_loss, max_accur = test(verify_set)
    last_save = 0

    while True:  # loop over the dataset multiple times
        tot_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn(inputs)
            # print(outputs, labels)
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            optimizer.step()

            from sys import stderr

            stderr.write('\rtraining: %f        ' % (i * 100 / n))

            # print statistics
            tot_loss += loss.item()
        print('average loss:', tot_loss / n)
        verify_loss, accur = test(verify_set)

        print('verify loss: %f, accuracy: %f' % (verify_loss, accur))

        last_save += 1

        if max_accur < accur:
            max_accur = accur
            save()
            last_save = 0

        # if verify_loss < min_loss:
        #     min_loss = verify_loss
        #     save()
        #     last_save = 0

            # if last_save > 1:


        if last_save == epoch_lim:
            # test(test_set)
            break

        # if verify_loss > 2 * min_loss:
        #     break
        
        epoch += 1

# return accuracy, fscore, corr
def judge(test_set):
    from sklearn.metrics import f1_score
    from scipy.stats import pearsonr
    import numpy as np
    n = len(test_set)
    accur = 0
    f1 = 0
    corr = 0
    y_pred = []
    y_true = []
    label_cnt = np.zeros(8)
    output_cnt = np.zeros(8)
    with torch.no_grad():
        for i, data in enumerate(test_set.data):
            text_raw, label_all = data
            # print(label_all)
            text, label = test_set.transform((text_raw, label_all, uniform_size))
            y_true.append(label.item())
            label_cnt[label] += 1
            # if label == 3:
            #     fennu += 1
            output, predict = evalutate(text)
            output = nn.functional.softmax(output, dim=1).view(-1)
            y_pred.append(predict.item())
            if predict == label:
                accur += 1
            output_cnt[predict] += 1
            corr += pearsonr(label_all, output)[0]
            from sys import stderr
            stderr.write('\rjudging: %f        ' % (i * 100 / n))


        accur = accur / n * 100
        f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='macro')
        print('\raccuracy:', accur)
        print('F-score:', f1)
        print('corr:', corr / n)
        # print(list(map(lambda x: x / n, label_cnt)))
        # print(list(map(lambda x: x / n, output_cnt)))

    # print('fennu:', fennu / n)

# def get_Fscore(test_set):
#     with torch.no_grad():
#         for text, label in test_set:
#             output, pos = 
