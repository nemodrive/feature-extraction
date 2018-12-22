import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

import xception, nasnet, nasnet_mobile, mobilenet_v2, inception_v3, resnet

import sys
#sys.path.append('PyTorchYOLOv3')
#import PyTorchYOLOv3.models as yolomodels

sys.path.append('RetinaNet')
import RetinaNet.retinanet as retinamodels
import RetinaNet.resnet_features as resnet_features

import cv2

import time

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def resize_image(img, width):
    wpercent = (width / float(img.shape[1]))
    hsize = int((float(img.shape[0]) * float(wpercent)))
    img = cv2.resize(img, (width, hsize))
    return img

def get_features(path):

    #model = nasnet.nasnetalarge(num_classes=1000, pretrained='imagenet')
    model = xception.xception(num_classes=1000, pretrained='imagenet')

    #state_dict = torch.load('data/mobilenet_v2.pth.tar')
    #model.load_state_dict(state_dict)

    #model = resnet.resnet152(pretrained=True)

    #model = yolomodels.Darknet('darknet_config_weights/darknet53.cfg')
    #model.load_weights('darknet_config_weights/darknet53.weights')

    #resnet50 = resnet_features.resnet50_features(pretrained=True)
    #model = retinamodels.FeaturePyramid(resnet50)

    model.cuda()
    model.eval()

    # number of parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    # read video data
    cap = cv2.VideoCapture(path)
    success = True
    average_fps = 0
    avg_mem_alloc = 0
    avg_mem_cached = 0
    count = 0
    start_time = time.time()
    while success:
        success, image = cap.read()
        if success:
            #start_time = time.time()
            count += 1
            image = image.astype(np.float32)
            #image = cv2.resize(image, (224, 125))
            image = cv2.resize(image, (320, 180))
            #image = cv2.resize(image, (640, 360))
            #print(image.shape)
            copy = np.copy(image)

            # reshape image for torchx`
            image = image.reshape((1, 3, image.shape[0], image.shape[1]))

            # show image

            #cv2.imshow('frame', copy / 255.0)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break


            #extract features
            t_image = Variable(torch.from_numpy(image)).cuda()
            features = model(t_image)

            # compute fps
            #fps = 1.0 / (time.time() - start_time)
            avg_mem_alloc += torch.cuda.memory_allocated()
            avg_mem_cached += torch.cuda.memory_cached()
            #print("FPS: ", fps)
            #average_fps += fps
            #print([i.shape for i in features])
            #print(features.shape)
            torch.save(features, 'features/f{}'.format(count))

    average_fps /= count
    avg_mem_alloc /= count
    avg_mem_cached /= count
    #print(params)
    #print(count)
    #print("{0:.2f}".format(average_fps))
    print(count)
    print("{0:.2f}".format(count / (time.time() - start_time)))
    print("{0:.2f}".format(avg_mem_alloc))
    print("{0:.2f}".format(avg_mem_cached))
    print("{0:.2f}".format(torch.cuda.max_memory_allocated()))
    print("{0:.2f}".format(torch.cuda.max_memory_cached()))

def main():
    get_features('data/test.mp4')

if __name__ == "__main__":
    main()
