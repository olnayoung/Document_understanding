import torch
import torch.nn as nn
import torch.optim as optim
import glob
import cv2
import numpy as np
import os
import time
import argparse

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from BERTgridNet import NetForBertgrid


class TrainOrTest():
    def __init__(self, lossname):
        self.class_num = args.class_num
        self.batch_size = args.batch_num
        self.n_epoch = args.n_epoch
        self.input_path = args.input_path
        self.label_path = args.label_path
        self.l_r = args.l_r
        self.model = NetForBertgrid(768, self.class_num)

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.l_r)
        elif args.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.l_r, momentum=0.9, weight_decay=0.0001)

        if lossname == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss()

    
    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins *60))
        return elapsed_mins, elapsed_secs


    def list_files(self, in_path):
        img_files = []
        for (dirpath, dirnames, filenames) in os.walk(in_path):
            for file in filenames:
                filename, ext = os.path.splitext(file)
                ext = str.lower(ext)
                if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                    img_files.append(file)
                    # img_files.append(os.path.join(dirpath, file))
        img_files.sort()
        return img_files


    def load_data(self, img_names, label_names, isImg):
        assert len(img_names) == len(label_names)

        for idx in range(len(img_names)):
            if isImg:
                img = np.array(cv2.imread(img_names[idx]))
                label = np.array(cv2.imread(label_names[idx]))
            else:
                img = np.load(img_names[idx])
                label = np.load(label_names[idx])
            
            if idx == 0:
                inputs = np.expand_dims(img, axis=0)
                labels = np.expand_dims(label, axis=0)
            else:
                img = np.expand_dims(img, axis=0)
                label = np.expand_dims(label, axis=0)

                inputs = np.concatenate((inputs, img))
                labels = np.concatenate((labels, label))

        return inputs, labels

    
    def calcul_accuracy(self, pred, ans):
        pred = torch.argmax(pred, dim=1)
        answer = answer.squeeze(1)

        correct = (pred == answer).float()
        a, b, c = answer.size()
        acc = correct.sum() / (b*c)

        flatten_ans = answer.flatten().cpu().numpy()
        flatten_pred = pred.flatten().cpu().numpy()
        
        f1_micro = f1_score(flatten_ans, flatten_pred, average='micro')
        f1_macro = f1_score(flatten_ans, flatten_pred, average='macro')
        
        return acc, f1_micro, f1_macro


    def train_epoch(self, input_lists):
        self.model.train()

        epoch_loss = 0

        for batch in range(int(len(input_lists)/self.batch_size)):
            self.optimizer.zero_grad()
            input_list, label_list = [], []

            for num in range(self.batch_size):
                idx = batch*self.batch_size + num
                filename = input_lists[idx]

                input_list.append(self.input_path + filename)
                label_list.append(self.label_path + filename)
            
            train_input, train_label = self.load_data(input_list, label_list, True)

            input_tensor = torch.tensor(train_input).cuda()
            input_tensor = input_tensor.permute(0, 3, 1, 2)
            label_tensor = torch.tensor(train_label).cuda()
            label_tensor = label_tensor.permute(0, 3, 1, 2)

            output = self.model(input_tensor)
            loss = self.criterion(output, label_tensor.squeeze(1))

            loss.backward()
            self.optimizer.step()

            acc = self.calcul_accuracy(output, label_tensor)

            epoch_loss += loss.item()
            epoch_acc += acc[0].item()
            epoch_mic += acc[1]
            epoch_mac += acc[2]

        return epoch_loss, epoch_acc, epoch_mic, epoch_mac

    
    def val_epoch(self, input_lists):
        self.model.eval()

        epoch_loss = 0

        for batch in range(int(len(input_lists)/self.batch_size)):
            self.optimizer.zero_grad()
            input_list, label_list = [], []

            for num in range(self.batch_size):
                idx = batch*self.batch_size + num
                filename = input_lists[idx]

                input_list.append(self.input_path + filename)
                label_list.append(self.label_path + filename)
            
            train_input, train_label = self.load_data(input_list, label_list, True)

            input_tensor = torch.tensor(train_input).cuda()
            input_tensor = input_tensor.permute(0, 3, 1, 2)
            label_tensor = torch.tensor(train_label).cuda()
            label_tensor = label_tensor.permute(0, 3, 1, 2)

            output = self.model(input_tensor)
            loss = self.criterion(output, label_tensor.squeeze(1))

            loss.backward()
            self.optimizer.step()

            acc = self.calcul_accuracy(output, label_tensor)

            epoch_loss += loss.item()
            epoch_acc += acc[0].item()
            epoch_mic += acc[1]
            epoch_mac += acc[2]

        return epoch_loss, epoch_acc, epoch_mic, epoch_mac


    def train(self):
        input_lists = self.list_files(self.input_path)

        for epoch in range(self.n_epoch):
            np.random.shuffle(input_lists)

            start_time = time.time()
            train_loss, train_acc, train_mic, train_mac = self.train_epoch(input_lists)
            end_time = time.time()

            mins, secs = self.epoch_time(start_time, end_time)
            print()
            print('--------------------------------------------------------------')
            print('Epoch: %02d | Epoch Time: %dm %ds' % (epoch+1, mins, secs))
            print('\tTrain Loss: %.3f | Train Acc: %.2f%% | mic: %.2f%% | mac: %.2f%%' % (train_loss, train_acc*100, train_mic*100, train_mac*100))


    def test(self):
        return


def main():
    return 0

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='')
    parser.add_argument('--label_path', type=str, default='')

    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_num', type=int, default=5)
    parser.add_argument('--class_num', type=int, default=31)
    parser.add_argument('--l_r', type=float, default=10^-4)

    parser.add_argument('--optimizer', type=str, default='Adam')


    args = parser.parse_args()
    main()