#!/usr/local/bin/python3.6

import torch.nn as nn
import torch

hidden_size = 72
save_path = 'models/'
batch_size = 64
uniform_size = 128
num_layers = 1

epoch_lim = 10

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=hidden_size, num_layers=num_layers):
        super(RNN, self).__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, x = self.gru(x)
        x = x[0]
        x = self.fc(x)
        return x

from word_embedding import vec_len as input_size
from data import label_len as output_size

import torch.optim as optim


rnn = RNN(input_size, output_size, hidden_size=hidden_size, num_layers=num_layers)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=1e-3)

# single
def evalutate(text):
    output = rnn(text.view(1, *text.size()))
    return output, output[0].topk(1)[1]

    # with torch.no_grad():
    #     hidden = rnn.initHidden(1)
    #     for word in text:
    #         output, hidden = rnn(word.view(-1, *word.size()), hidden)

    #     return output, output[0].topk(1)[1]

def test(test_set):
    with torch.no_grad():
        n = len(test_set)
        cnt = 0

        labels = []
        outputs = []

        # for text, label in test_set:
        for i, data in enumerate(test_set):
            text, label = data
            output, predict = evalutate(text)
            labels.append(label)
            outputs.append(output)
            if label == predict:
                cnt += 1

            from sys import stderr
            stderr.write('\rtesting: %f             ' % (i * 100 / n))

        return criterion(torch.cat(outputs), torch.cat(labels)), cnt * 100 / n

def save():
    import os
    try:
        os.mkdir(save_path)
    except FileExistsError:
        pass
    
    torch.save(rnn.state_dict(), os.path.join(save_path, 'rnn.pt'))
    print('save success')

def load(name='rnn.pt'):
    import os
    try:
        rnn.load_state_dict(torch.load(os.path.join(save_path, 'rnn.pt')))
        # print('load success')
    except Exception as e:
        print(e)
        print('load fail')

def train_all(train_loader, verify_set, test_set):
    # iter_n = 100
    n = len(train_loader)
    min_loss, accur = test(verify_set)
    last_save = 0
    while True:
        tot_loss = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # text, label = data
            optimizer.zero_grad()
            # output, loss = train(text, label)
            outputs = rnn(inputs)
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()


            from sys import stderr
            stderr.write('\rtraining: %f        ' % (i * 100 / n))

        print('average loss:', tot_loss / n)
        # accur = test(verify_set)
        verify_loss, accur = test(verify_set)
        print('verify loss: %f, accuracy: %f' % (verify_loss, accur))
        last_save += 1
        if verify_loss < min_loss:
            min_loss = verify_loss
            save()
            last_save = 0

        if last_save == epoch_lim:
            break
        # if verify_loss > 1.5 * min_loss:
        #     break


def judge(test_set):
    from sklearn.metrics import f1_score
    from scipy.stats import pearsonr
    import numpy as np
    n = len(test_set)
    accur = 0
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