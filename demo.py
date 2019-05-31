import torch
from PIL import Image
from detect_models.transforms import build_transforms
import numpy as np
from detect_models.models import init_model
from detect_models.utils.misc import process_output, mkdirs
from detect_models.utils.misc import find_bottom, find_long_edges, split_edge_seqence, \
    norm2, vector_cos, vector_sin, cos
import pickle
from functools import partial
import cv2
import queue
import glob
from tqdm import tqdm
import os
from scipy import io

from args import argument_parser
from recognition_models.utils import CTCLabelConverter, AttnLabelConverter, TransformerLabelConverter
from recognition_models.dataset import RawDataset, AlignCollate
from recognition_models.model import Model
import torchvision.transforms as T

args = argument_parser().parse_args()


def load_model(model, model_path):
    checkpoint = torch.load(model_path, pickle_module=pickle)
    pretrain_dict = checkpoint['state_dict']
    model.load_state_dict(pretrain_dict)
    print("Loaded pretrained weights from '{}'".format(model_path))
    return model


def init_detect_model(save_path):
    model = init_model(name='se_resnext101_32x4d')
    load_model(model, save_path)
    model = model.to('cuda')
    model.eval()
    transform = build_transforms(
        maxHeight=768, maxWidth=768, is_train=False, dynamic=False)
    return model, transform


def extract_detect_contour(model, im, transform):
    # im = torch.Tensor(im)
    image = im.copy()
    img, _ = transform(im, None)
    im = img.transpose(2, 0, 1)
    im = torch.Tensor(im).to('cuda').unsqueeze(0)
    # while True:
    model.eval()
    with torch.no_grad():
        output = model(im)
    contours = process_output(img, output[0].to('cpu').numpy(),
                              image, threshold=0.4, min_area=10)
    return contours


def get_transform(img, cnt, resizeH=32, resizeW=100):
    x, y, w, h = cv2.boundingRect(cnt)
    tmp_img = np.zeros(img.shape).astype(np.uint8)
    # print(img.shape)
    cv2.drawContours(tmp_img, [cnt], -1, (1, 1, 1), cv2.FILLED)
    img *= tmp_img
    src_pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
    dest_pts = np.array(
        [[0, 0], [resizeW, 0], [resizeW, resizeH], [0, resizeH]])
    h, mask = cv2.findHomography(src_pts, dest_pts, cv2.RANSAC)
    imgReg = cv2.warpPerspective(img, h, (resizeW, resizeH))
    return imgReg


def extract_text_bbox(raw_img, contours):
    list_text_img = []
    for cnt in contours:
        bbox_img = get_transform(raw_img.copy(), cnt)
        list_text_img.append(bbox_img)
    return list_text_img


detect_model, detect_transform = init_detect_model(args.detection_checkpoint)
im = cv2.imread(args.input_img)
raw_image = im.copy()
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
contours = extract_detect_contour(detect_model, im, detect_transform)

list_text_img = extract_text_bbox(raw_image, contours)

cv2.drawContours(raw_image,contours,-1,(0,255,0),3)
cv2.imwrite('result/detection_result.jpg',raw_image)




def init_recognition_model(save_path, opt):
    converter = TransformerLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    model = Model(opt)
    checkpoint = torch.load(save_path)
    if type(checkpoint) == dict:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    del checkpoint
    torch.cuda.empty_cache()
    model = model.to('cuda')
    model.eval()
    return model, converter


recognition_model, converter = init_recognition_model(args.recognition_checkpoint, args)
toTensor = T.ToTensor()

for i, text_img in enumerate(list_text_img):
    img = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
    img = toTensor(img).cuda().unsqueeze(0)
    img.sub_(0.5).div_(0.5)
    length_for_pred = torch.cuda.IntTensor(
        [args.batch_max_length] * 1)
    text_for_pred = torch.cuda.LongTensor(
        1, args.batch_max_length + 1).fill_(0)
    with torch.no_grad():
        preds = recognition_model(img, text_for_pred, is_train=False)
        print(preds.size())
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
        pred = preds_str[0]
        pred = pred[:pred.find('</s>')]
        print(pred)
    text_img = cv2.resize(text_img,(200,64))
    tmp_img = np.zeros((96,200,3),np.uint8)
    tmp_img.fill(255)
    tmp_img[:64,:200] = text_img
    raw_img = tmp_img
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (5, 90)
    fontScale = 1  
    fontColor = (0, 0, 255)
    lineType = 2
    cv2.putText(raw_img, pred,(5,90),font,fontScale,(0, 255, 0),lineType)
    cv2.imshow('1',raw_img)
    cv2.waitKey(0)
    cv2.imwrite('result/{}.jpg'.format(i+1),raw_img)

cv2.destroyAllWindows()