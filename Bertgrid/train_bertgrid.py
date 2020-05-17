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


class trainNetwork():
    def __init__(self):
        self.class_num = args.class_num
        self.batch_size = args.batch_num
        self.n_epochs = args.n_epochs

        self.train_input_path = os.path.join(args.train_path, 'CORD', 'train', 'BERTgrid')
        self.train_label_path = os.path.join(args.train_path, 'CORD', 'train', 'LABELgrid')
        self.val_input_path = os.path.join(args.train_path, 'CORD', 'dev', 'BERTgrid')
        self.val_label_path = os.path.join(args.train_path, 'CORD', 'dev', 'LABELgrid')
        self.save_path = os.path.join(args.train_path, 'CORD', 'params')
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        self.l_r = args.l_r
        self.model = NetForBertgrid(768, self.class_num)
        self.model.cuda()

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.l_r)
        elif args.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.l_r, momentum=0.9, weight_decay=0.0001)

        if args.lossname == 'CrossEntropy':
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
                if ext == '.npz':
                    img_files.append(file)
                    # img_files.append(os.path.join(dirpath, file))
        img_files.sort()
        return img_files


    def load_data(self, img_names, label_names):
        assert len(img_names) == len(label_names)

        for idx in range(len(img_names)):
            img = np.load(img_names[idx])
            label = np.load(label_names[idx])
            img = img['arr_0']
            label = label['arr_0']
            
            if idx == 0:
                inputs = np.expand_dims(img, axis=0)
                labels = np.expand_dims(label, axis=0)

                _, inputs_h, inputs_w, _ = inputs.shape

                if inputs_h % 8 != 0:
                    remain = 8 - inputs_h % 8
                    inputs = np.pad(inputs, ((0, 0), (0, remain), (0, 0), (0, 0)), 'constant')
                    labels = np.pad(labels, ((0, 0), (0, remain), (0, 0), (0, 0)), 'constant')
                    
                if inputs_w % 8 != 0:
                    remain = 8 - inputs_w % 8
                    inputs = np.pad(inputs, ((0, 0), (0, 0), (0, remain), (0, 0)), 'constant')
                    labels = np.pad(labels, ((0, 0), (0, 0), (0, remain), (0, 0)), 'constant')

            else:
                img = np.expand_dims(img, axis=0)
                label = np.expand_dims(label, axis=0)

                _, img_h, img_w, _ = img.shape
                _, inputs_h, inputs_w, _ = inputs.shape

                if inputs_h % 8 != 0:
                    remain = 8 - inputs_h % 8
                    inputs = np.pad(inputs, ((0, 0), (0, remain), (0, 0), (0, 0)), 'constant') 
                    labels = np.pad(labels, ((0, 0), (0, remain), (0, 0), (0, 0)), 'constant')
                    inputs_h += remain
                    
                if inputs_w % 8 != 0:
                    remain = 8 - inputs_w % 8
                    inputs = np.pad(inputs, ((0, 0), (0, 0), (0, remain), (0, 0)), 'constant')
                    labels = np.pad(labels, ((0, 0), (0, 0), (0, remain), (0, 0)), 'constant')
                    inputs_w += remain

                if img_h > inputs_h:
                    inputs = np.pad(inputs, ((0, 0), (0, img_h - inputs_h), (0, 0), (0, 0)), 'constant')
                    labels = np.pad(labels, ((0, 0), (0, img_h - inputs_h), (0, 0), (0, 0)), 'constant')
                elif inputs_h > img_h:
                    img = np.pad(img, ((0, 0), (0, inputs_h - img_h), (0, 0), (0, 0)), 'constant')
                    label = np.pad(label, ((0, 0), (0, inputs_h - img_h), (0, 0), (0, 0)), 'constant')

                if img_w > inputs_w:
                    inputs = np.pad(inputs, ((0, 0), (0, 0), (0, img_w - inputs_w), (0, 0)), 'constant')
                    labels = np.pad(labels, ((0, 0), (0, 0), (0, img_w - inputs_w), (0, 0)), 'constant')
                elif inputs_w > img_w:
                    img = np.pad(img, ((0, 0), (0, 0), (0, inputs_w - img_w), (0, 0)), 'constant')
                    label = np.pad(label, ((0, 0), (0, 0), (0, inputs_w - img_w), (0, 0)), 'constant')

                inputs = np.concatenate((inputs, img))
                labels = np.concatenate((labels, label))

        return inputs, labels

    
    def calcul_accuracy(self, pred, ans):
        pred = torch.argmax(pred, dim=1)
        answer = ans.squeeze(1)

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

        epoch_loss, epoch_acc, epoch_mic, epoch_mac = 0, 0, 0, 0

        for batch in range(int(len(input_lists)/self.batch_size)):
            self.optimizer.zero_grad()
            input_list, label_list = [], []

            for num in range(self.batch_size):
                idx = batch*self.batch_size + num
                filename = input_lists[idx]

                input_list.append(self.train_input_path + '/'+filename)
                label_list.append(self.train_label_path + '/'+filename)
            
            train_input, train_label = self.load_data(input_list, label_list)

            input_tensor = torch.tensor(train_input, dtype=torch.float).cuda()
            input_tensor = input_tensor.permute(0, 3, 1, 2)
            label_tensor = torch.tensor(train_label, dtype=torch.float).cuda()
            label_tensor = label_tensor.permute(0, 3, 1, 2)

            output = self.model(input_tensor)
            loss = self.criterion(output, label_tensor.squeeze(1).long())

            loss.backward()
            self.optimizer.step()

            acc = self.calcul_accuracy(output, label_tensor.long())

            epoch_loss += loss.item()
            epoch_acc += acc[0].item()
            epoch_mic += acc[1]
            epoch_mac += acc[2]

        return epoch_loss / len(input_lists), epoch_acc / len(input_lists), epoch_mic / len(input_lists), epoch_mac / len(input_lists)

    
    def val_epoch(self, input_lists):
        self.model.eval()

        epoch_loss, epoch_acc, epoch_mic, epoch_mac = 0, 0, 0, 0

        for batch in range(int(len(input_lists)/self.batch_size)):
            self.optimizer.zero_grad()
            input_list, label_list = [], []

            for num in range(self.batch_size):
                idx = batch*self.batch_size + num
                filename = input_lists[idx]

                input_list.append(self.val_input_path + '/'+filename)
                label_list.append(self.val_label_path + '/'+filename)
            
            train_input, train_label = self.load_data(input_list, label_list)

            input_tensor = torch.tensor(train_input, dtype=torch.float).cuda()
            input_tensor = input_tensor.permute(0, 3, 1, 2)
            label_tensor = torch.tensor(train_label, dtype=torch.float).cuda()
            label_tensor = label_tensor.permute(0, 3, 1, 2)

            output = self.model(input_tensor)
            loss = self.criterion(output, label_tensor.squeeze(1).long())

            acc = self.calcul_accuracy(output, label_tensor.long())

            epoch_loss += loss.item()
            epoch_acc += acc[0].item()
            epoch_mic += acc[1]
            epoch_mac += acc[2]

        return epoch_loss / len(input_lists), epoch_acc / len(input_lists), epoch_mic / len(input_lists), epoch_mac / len(input_lists)


    def train(self):
        input_lists = self.list_files(self.train_input_path)
        val_input_lists = self.list_files(self.val_input_path)
        past_acc, past_mic, past_mac = 0, 0, 0

        for epoch in range(self.n_epochs):
            np.random.shuffle(input_lists)

            start_time = time.time()
            train_loss, train_acc, train_mic, train_mac = self.train_epoch(input_lists)
            val_loss, val_acc, val_mic, val_mac = self.val_epoch(val_input_lists)
            end_time = time.time()

            if past_acc < val_acc or past_mic < val_mic or past_mac < val_mac:
                torch.save(self.model.state_dict(), self.save_path + '/' + str(epoch+100) + '.pt')
                past_acc, past_mic, past_mac = val_acc, val_mic, val_mac

            mins, secs = self.epoch_time(start_time, end_time)
            print()
            print('--------------------------------------------------------------')
            print('Epoch: %02d | Epoch Time: %dm %ds' % (epoch+1, mins, secs))
            print('\tTrain Loss: %.3f | Train Acc: %.2f%% | mic: %.2f%% | mac: %.2f%%' % (train_loss, train_acc*100, train_mic*100, train_mac*100))
            print('\tVal Loss: %.3f | Val Acc: %.2f%% | mic: %.2f%% | mac: %.2f%%' % (val_loss, val_acc*100, val_mic*100, val_mac*100))



def main():
    trainNetwork().train()

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, default='/home/ny/pytorch_codes/DocumentIntelligence/dataset')

    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--class_num', type=int, default=43)
    parser.add_argument('--l_r', type=float, default=10**(-4))

    parser.add_argument('--lossname', type=str, default='CrossEntropy')
    parser.add_argument('--optimizer', type=str, default='Adam')


    args = parser.parse_args()
    main()