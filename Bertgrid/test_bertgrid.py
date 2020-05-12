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


class testNetwork():
    def __init__(self, lossname):
        self.class_num = args.class_num
        self.batch_size = args.batch_num

        self.test_input_path = os.path.join(args.test_path, 'CORD', 'test', 'BERTgrid')
        self.test_label_path = os.path.join(args.test_path, 'CORD', 'test', 'LABELgrid')

        self.model = NetForBertgrid(768, self.class_num)
        if lossname == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss()


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
        answer = ans.squeeze(1)

        correct = (pred == answer).float()
        a, b, c = answer.size()
        acc = correct.sum() / (b*c)

        flatten_ans = answer.flatten().cpu().numpy()
        flatten_pred = pred.flatten().cpu().numpy()
        
        f1_micro = f1_score(flatten_ans, flatten_pred, average='micro')
        f1_macro = f1_score(flatten_ans, flatten_pred, average='macro')
        
        return acc, f1_micro, f1_macro


    def val_epoch(self, input_lists):
        self.model.eval()

        epoch_loss = 0

        for batch in range(int(len(input_lists)/self.batch_size)):
            input_list, label_list = [], []

            for num in range(self.batch_size):
                idx = batch*self.batch_size + num
                filename = input_lists[idx]

                input_list.append(self.test_input_path + filename)
                label_list.append(self.test_label_path + filename)
            
            train_input, train_label = self.load_data(input_list, label_list, True)

            input_tensor = torch.tensor(train_input).cuda()
            input_tensor = input_tensor.permute(0, 3, 1, 2)
            label_tensor = torch.tensor(train_label).cuda()
            label_tensor = label_tensor.permute(0, 3, 1, 2)

            output = self.model(input_tensor)
            loss = self.criterion(output, label_tensor.squeeze(1))

            loss.backward()

            acc = self.calcul_accuracy(output, label_tensor)

            epoch_loss += loss.item()
            epoch_acc += acc[0].item()
            epoch_mic += acc[1]
            epoch_mac += acc[2]

        return epoch_loss / len(input_list), epoch_acc / len(input_list), epoch_mic / len(input_list), epoch_mac / len(input_list)


    def testMany(self):
        test_lists = self.list_files(self.test_path)
        val_loss, val_acc, val_mic, val_mac = self.val_epoch(test_lists)

        print('\tVal Loss: %.3f | Val Acc: %.2f%% | mic: %.2f%% | mac: %.2f%%' % (val_loss, val_acc*100, val_mic*100, val_mac*100))


    def testOne(self, img_path):
        self.model.eval()

        input_tensor = torch.tensor(input).cuda()
        input_tensor = input_tensor.permute(0,3,1,2)

        output = self.model(input_tensor)
        output = torch.argmax(output, axis=1)
        output = output.cpu().numpy()

        return output


def main():
    return 0

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_path', type=str, default='/home/ny/pytorch_codes/DocumentIntelligence/dataset/')

    parser.add_argument('--batch_num', type=int, default=5)
    parser.add_argument('--class_num', type=int, default=31)


    args = parser.parse_args()
    main()
