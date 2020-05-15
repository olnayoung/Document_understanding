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
    def __init__(self):
        self.class_num = args.class_num
        self.batch_size = args.batch_num

        self.test_input_path = os.path.join(args.test_path, 'CORD', 'test', 'BERTgrid')
        self.test_label_path = os.path.join(args.test_path, 'CORD', 'test', 'LABELgrid')
        weight_path = os.path.join(args.test_path, 'CORD', 'params', args.weight)

        self.model = NetForBertgrid(768, self.class_num)
        self.model.cuda()
        self.model.load_state_dict(torch.load(weight_path))
        if args.lossname == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss()


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


    def val_epoch(self, input_lists):
        self.model.eval()

        epoch_loss, epoch_acc, epoch_mic, epoch_mac = 0, 0, 0, 0

        for batch in range(int(len(input_lists)/self.batch_size)):
            input_list, label_list = [], []

            for num in range(self.batch_size):
                idx = batch*self.batch_size + num
                filename = input_lists[idx]

                input_list.append(self.test_input_path + '/' + filename)
                label_list.append(self.test_label_path + '/' + filename)
            
            train_input, train_label = self.load_data(input_list, label_list)

            input_tensor = torch.tensor(train_input, dtype=torch.float).cuda()
            input_tensor = input_tensor.permute(0, 3, 1, 2)
            label_tensor = torch.tensor(train_label, dtype=torch.float).cuda()
            label_tensor = label_tensor.permute(0, 3, 1, 2)

            output = self.model(input_tensor)
            loss = self.criterion(output, label_tensor.squeeze(1).long())

            loss.backward()

            acc = self.calcul_accuracy(output, label_tensor)

            epoch_loss += loss.item()
            epoch_acc += acc[0].item()
            epoch_mic += acc[1]
            epoch_mac += acc[2]

        return epoch_loss / len(input_list), epoch_acc / len(input_list), epoch_mic / len(input_list), epoch_mac / len(input_list)


    def testMany(self):
        test_lists = self.list_files(self.test_input_path)
        val_loss, val_acc, val_mic, val_mac = self.val_epoch(test_lists)

        print('\tVal Loss: %.3f | Val Acc: %.2f%% | mic: %.2f%% | mac: %.2f%%' % (val_loss, val_acc*100, val_mic*100, val_mac*100))


    def interpolation(self, input):
        _, h, w = input.shape
        output = np.zeros((h*6, w*6))
        
        output[0::6, 0::6] = input
        output[1::6, 1::6] = input
        output[2::6, 2::6] = input
        output[3::6, 3::6] = input
        output[4::6, 4::6] = input
        output[5::6, 5::6] = input
        
        return output


    def label_from_output(self, big_output, bboxes):
        labels = []
        for n in range(len(bboxes)):
            x1, x2, x3, x4, y1, y2, y3, y4 = bboxes[n]
            labels.append(big_output[int(y1)+3][int(x1)+3])
            
        return labels


    def draw_pred_n_answer(self, img, pos, new_label):
        x1, x2, x3, x4, y1, y2, y3, y4 = pos
        x1, x2, x3, x4 = int(x1), int(x2), int(x3), int(x4)
        y1, y2, y3, y4 = int(y1), int(y2), int(y3), int(y4)
            
        if new_label == 0:
            new_color = (0, 0, 0)
        elif 1 <= new_label < 17:
            new_color = (0, 0, 255)
        elif 17 <= new_label < 23:
            new_color = (50, 100, 200)
        elif 23 <= new_label < 31:
            new_color = (150, 150, 0)
        elif 31 <= new_label < 35:
            new_color = (200, 100, 0)
        elif 35 <= new_label <= 43:
            new_color = (150, 0, 150)
        
        minx, maxx = min(x1, x2, x3, x4), max(x1, x2, x3, x4)
        miny, maxy = min(y1, y2, y3, y4), max(y1, y2, y3, y4)
        
        # y = ax + b
        a1 = (y2-y1)/(x2-x1)
        if x3 != x2:
            a2 = (y3-y2)/(x3-x2)
        else:
            a2 = 0
        a3 = (y4-y3)/(x4-x3)
        if x1 != x4:
            a4 = (y1-y4)/(x1-x4)
        else:
            a4 = 0
        b1, b2, b3, b4 = y1-a1*x1, y2-a2*x2, y3-a3*x3, y4-a4*x4
        
        alpha = 0.7
        
        up2 = 0 if a2 >= 0 else 1
        up4 = 0 if a4 >= 0 else 1
        
    #     print(a1, a2, a3, a4)
        
        for x in range(minx, maxx):
            for y in range(miny, maxy):
                if y < a1*x+b1:
                    continue
                        
                if up2 == 0:
                    if y < a2*x+b2:
                        continue
                else:
                    if y > a2*x+b2:
                        continue
                        
                if y > a3*x+b3:
                    continue
                        
                if up4 == 0:
                    if y > a4*x+b4:
                        continue
                else:
                    if y < a4*x+b4:
                        continue
                img[y, x, :] = [img[y, x, 0] * alpha + (1-alpha) * new_color[0], img[y, x, 1] * alpha + (1-alpha) * new_color[1], img[y, x, 2] * alpha + (1-alpha) * new_color[2]]
    


    def testOne(self, img_path, label_path):
        self.model.eval()

        input_list = [img_path]
        label_list = [label_path]
        input_data, label_data = self.load_data(input_list, label_list)

        input_tensor = torch.tensor(input_data, dtype=torch.float).cuda()
        input_tensor = input_tensor.permute(0,3,1,2)

        output = self.model(input_tensor)
        output = torch.argmax(output, axis=1)
        output = output.cpu().numpy()

        big_output = self.interpolation(output)

        return


def main():
    testNetwork().testOne('D:/data/CORD/test/BERTgrid/receipt_00000.png.npz', 'D:/data/CORD/test/LABELgrid/receipt_00000.png.npz')

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_path', type=str, default='D:\data')
    parser.add_argument('--weight', type=str, default='100.pt')

    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--class_num', type=int, default=43)
    parser.add_argument('--lossname', type=str, default='CrossEntropy')

    args = parser.parse_args()
    main()
