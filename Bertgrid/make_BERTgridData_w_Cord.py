import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import imageio
import numpy as np
from tqdm import tqdm
import argparse
from transformers import BertTokenizer, BertModel


def get_BERTvector(model, input_ids, device):
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    seq_hiddens, pooled = model(input_tensor)

    return seq_hiddens, pooled


def img2BERTgrid(img, input_ids, bboxes, model, device):
    resize = args.resize
    hei, wid, _ = img.shape
    BERTgrid = np.zeros([int(hei/resize), int(wid/resize), 768])
    seq_hiddens, _ = get_BERTvector(model, input_ids, device)

    for word, bbox in zip(seq_hiddens[0][1:-1], bboxes):
        word = word.cpu().detach().numpy()
        # word = word.detach().numpy()

        minX = int(min(bbox[0:4])/resize)
        maxX = int(max(bbox[0:4])/resize)
        minY = int(min(bbox[4:])/resize)
        maxY = int(max(bbox[4:])/resize)

        BERTgrid[minY:maxY+1, minX:maxX+1] = word

    return BERTgrid


def label2LABELgrid(img, category_ids, bboxes):
    resize = args.resize
    hei, wid, _ = img.shape
    LABELgrid = np.zeros([int(hei/resize), int(wid/resize), 1])

    for label, bbox in zip(category_ids, bboxes):
        minX = int(min(bbox[0:4]))
        maxX = int(max(bbox[0:4]))
        minY = int(min(bbox[4:]))
        maxY = int(max(bbox[4:]))

        LABELgrid[minY:maxY+1, minX:maxX+1] = label

    return LABELgrid


class CordDataset(Dataset):
    def __init__(self):
        split = args.split
        assert split in  ['train', 'dev', 'test']
        self.verbose = args.verbose
        self.data_path = os.path.join(args.data_root, 'CORD', split)
        if not os.path.isdir(self.data_path):
            assert 'Please Download data'

        self.CORD_CATEGORY_TO_ID = {'menu.nm':1, 'menu.num':2, 'menu.unitprice':3, 'menu.cnt':4, 'menu.discountprice':5, 'menu.price':6,
                                    'menu.itemsubtotal':7, 'menu.vatyn':8, 'menu.etc':9, 'menu.sub_nm':10, 'menu.sub_num':11, 'menu.sub_unitprice':12,
                                    'menu.sub_cnt':13, 'menu.sub_discountprice':14, 'menu.sub_price':15, 'menu.sub_etc':16, 'void_menu.nm':17,
                                    'void_menu.num':18, 'void_menu.unitprice':19, 'void_menu.cnt':20, 'void_menu.price':21, 'void_menu.etc':22,
                                    'sub_total.subtotal_price':23, 'sub_total.discount_price':24, 'sub_total.subtotal_count':25,
                                    'sub_total.service_price':26, 'sub_total.othersvc_price':27, 'sub_total.tax_price':28, 'sub_total.tax_and_service':29,
                                    'sub_total.etc':30, 'void_total.subtotal_price':31, 'void_total.tax_price':32, 'void_total.total_price':33,
                                    'void_total.etc':34, 'total.total_price':35, 'total.total_etc':36, 'total.cashprice':37, 'total.changeprice':38,
                                    'total.creditcardprice':39, 'total.emoneyprice':40, 'total.menutype_cnt':41, 'total.menuqty_cnt':42}
        self.CORD_ID_TO_CATEGORY = {v: k for k, v in self.CORD_CATEGORY_TO_ID.items()}
        self.img, self.raw_annotation = self._load_data()
        self.annotation, self.label = self._parse_annotations()
        self._post_data_process()

    

    def _load_data(self):
        image_root = os.path.join(self.data_path, 'image')
        annotation_root = os.path.join(self.data_path, 'json')
        self.img_list = os.listdir(image_root)
        self.img_list.sort()
        annotation_list = os.listdir(annotation_root)
        annotation_list.sort()

        if self.verbose:
            self.img_list = tqdm(self.img_list)
            annotation_list = tqdm(annotation_list)
        
        images = [np.array(imageio.imread(os.path.join(image_root, path))) for path in self.img_list]
        annotations = [json.load(open(os.path.join(annotation_root, path))) for path in annotation_list]

        return images, annotations


    def getFilelist(self):
        return self.img_list


    def _parse_annotations(self):
        '''
        bbox annotation: x1, x2, x3, x4, y1, y2, y3, y4
        '''
        annotations = []
        labels = []

        for ann in self.raw_annotation:
            item_annotations = []
            item_labels = []
            items = ann['valid_line']

            for item in items:
                category = item['category']
                group_id = item['group_id']

                for word in item['words']:
                    is_key = word['is_key']
                    quad = word['quad']
                    text = word['text']
                    item_annotations.append(((quad['x1'], quad['x2'], quad['x3'], quad['x4'], quad['y1'], quad['y2'], quad['y3'], quad['y4']), text))

                    try:
                        item_labels.append((self.CORD_CATEGORY_TO_ID[category], is_key, group_id))
                    except Exception as e:
                        print('{} is not in self.CORD_CATEGORY_TO_ID')
                        raise e

            assert len(item_annotations) == len(item_labels)
            annotations.append(item_annotations)
            labels.append(item_labels)

        return annotations, labels


    def _post_data_process(self):
        self.bboxes = []
        self.input_ids = []
        self.category_ids = []

        for ann in self.annotation:
            bboxes, input_ids = zip(*ann)
            self.bboxes.append(np.array(bboxes))
            self.input_ids.append(np.array(input_ids))
        
        for label in self.label:
            category_ids, _, _ = zip(*label)
            self.category_ids.append(category_ids)


    def __len__(self):
        return len(self.img)

    
    def __getitem__(self, idx):
        img = self.img[idx]
        ann = self.annotation[idx]
        label = self.label[idx]
        bboxes = self.bboxes[idx]
        category_ids = self.category_ids[idx]
        input_ids = self.input_ids[idx]

        return img, input_ids, bboxes, category_ids



class CordBERTGridDataset(CordDataset):
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.return_id = args.return_id
        super(CordBERTGridDataset, self).__init__()


    def _parse_annotations(self):
        '''
        x1 ------------ x2
         |              |
        x4 ------------ x4
        '''
        annotations = []
        labels = []

        for ann in self.raw_annotation:
            item_annotations = []
            item_labels = []
            items = ann['valid_line']

            for item in items:
                category = item['category']
                group_id = item['group_id']

                for word in item['words']:
                    is_key = word['is_key']
                    quad = word['quad']
                    text = word['text']

                    tokens = self.tokenizer.tokenize(text)
                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                    try:
                        up_x_interval = (quad['x2'] - quad['x1']) / len(text)
                        down_x_interval = (quad['x3'] - quad['x4']) / len(text)
                        up_y_interval = (quad['y2'] - quad['y1']) / len(text)
                        down_y_interval = (quad['y3'] - quad['y4']) / len(text)
                    except Exception as e:
                        continue

                    x1_start = quad['x1']
                    y1_start = quad['y1']
                    x4_start = quad['x4']
                    y4_start = quad['y4']

                    for token, input_id in zip(tokens, input_ids):
                        loc = (x1_start, x1_start + up_x_interval, x4_start + down_x_interval, x4_start,
                               y1_start, y1_start + up_y_interval, y4_start + down_y_interval, y4_start)
                        
                        if self.return_id:
                            item_annotations.append((loc, input_id))
                        else:
                            item_annotations.append((loc, token))
                        
                        x1_start += up_x_interval
                        y1_start += up_y_interval
                        x4_start += down_x_interval
                        y4_start += down_y_interval

                        try:
                            item_labels.append((self.CORD_CATEGORY_TO_ID[category], is_key, group_id))
                        except Exception as e:
                            print('{} is not in self.CORD_CATEGORY_TO_ID')
                            raise e

            assert len(item_annotations) == len(item_labels)
            annotations.append(item_annotations)
            labels.append(item_labels)

        return annotations, labels

    
    def _post_data_process(self):
        self.bboxes = []
        self.input_ids = []
        self.category_ids = []

        for ann in self.annotation:
            bboxes, input_ids = zip(*ann)
            self.bboxes.append(np.array(bboxes))

            if self.return_id:
                self.input_ids.append(np.array((self.tokenizer.cls_token_id, *input_ids, self.tokenizer.sep_token_id)))
            else:
                self.input_ids.append(np.array((self.tokenizer.cls_token, *input_ids, self.tokenizer.sep_token)))

        for label in self.label:
            category_ids, _, _ = zip(*label)
            self.category_ids.append(np.array(category_ids))

    
    def __getitem__(self, idx):
        img = self.img[idx]
        ann = self.annotation[idx]
        label = self.label[idx]
        bboxes = self.bboxes[idx]
        category_ids = self.category_ids[idx]
        input_ids = self.input_ids[idx]

        BERTgrid = img2BERTgrid(img, input_ids, bboxes, self.model, self.device)
        LABELgrid = label2LABELgrid(img, category_ids, bboxes)

        return BERTgrid, LABELgrid


def saveData(datas, filelist):
    data_root = args.data_root
    split = args.split

    path_data = os.path.join(data_root, 'CORD', split, 'BERTgrid')
    path_label = os.path.join(data_root, 'CORD', split, 'LABELgrid')

    if not os.path.isdir(path_data):
        os.mkdir(path_data)
    if not os.path.isdir(path_label):
        os.mkdir(path_label)

    idx = 0
    for filename in zip(filelist):
        np.savez(os.path.join(path_data+'/'+filename[0]), datas[idx][0])
        np.savez(os.path.join(path_label+'/'+filename[0]), datas[idx][1])
        idx += 1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.tokenizer == 'Bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
        model = model.to(device)

    bert_grid_dataset = CordBERTGridDataset(tokenizer, model, device)
    filelist = bert_grid_dataset.getFilelist()
    saveData(bert_grid_dataset, filelist)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='D:\data')
    parser.add_argument('--split', type=str, default='train', help='train, dev, test')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--resize', type=int, default=6, help='image will be resized into 1/resize')

    parser.add_argument('--tokenizer', type=str, default='Bert')
    parser.add_argument('--return_id', type=bool, default=True, help='True: return id / False: return token')

    args = parser.parse_args()
    main()