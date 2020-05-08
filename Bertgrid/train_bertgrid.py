import torch
import torch.nn as nn
import torch.optim as optim
import glob
import cv2
import numpy as np
import os
import time

from BERTgridNet import NetForBertgrid


class TrainOrTest():
    def __init__(self, class_num, batch_size, n_epoch, input_path, label_path, l_r, optimizer, lossname):
        self.class_num = class_num
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.input_path = input_path
        self.label_path = label_path
        self.l_r = l_r
        self.model = NetForBertgrid(768, self.class_num)

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.l_r)
        elif optimizer == 'SGD':
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
        return


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
            label_tensor = torch.tensor(train_label).cuda()
            input_tensor = input_tensor.permute(0, 3, 1, 2)
            label_tensor = label_tensor.permute(0, 3, 1, 2)

            output = self.model(input_tensor)
            loss = self.criterion(output, label_tensor.squeeze(1))

            epoch_loss += loss.item()

        return epoch_loss


    def train(self):
        input_lists = self.list_files(self.input_path)

        for epoch in range(self.n_epoch):
            np.random.shuffle(input_lists)

            start_time = time.time()
            self.train_epoch(input_lists)
            end_time = time.time()

            mins, secs = self.epoch_time(start_time, end_time)
            print('Epoch: %02d | Epoch Time: %dm %ds' % (epoch+1, mins, secs))

        return


    def test(self):
        return


def main():
    return 0

if __name__=='__main__':
    main()