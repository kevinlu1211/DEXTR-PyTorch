import os
import pdb
import torch
from collections import OrderedDict
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

from torch.nn.functional import upsample
from torch.autograd import Variable

import networks.deeplab_resnet as resnet
from mypath import Path
from dataloaders import helpers as helpers

def to_gray_scale(images, min_val=0, max_val=255):
    images_min = np.min(images)
    images_max = np.max(images)
    a = (max_val - min_val)/(images_max - images_min)
    b = max_val - a * images_max
    gs_masks = (a * np.array(images) + b)
    if max_val > 255:
        return gs_masks.astype(np.uint16)
    else:
        return gs_masks.astype(np.uint8)

modelName = 'dextr_pascal-sbd'
pad = 50
thres = 0.8
gpu_id = 0
default_n_points = 6
use_cuda = torch.cuda.is_available()

#  Create the network and load the weights
net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
print("Initializing weights from: {}".format(
    os.path.join(Path.models_dir(), modelName + '.pth')))
state_dict_checkpoint = torch.load(os.path.join(Path.models_dir(), modelName + '.pth'),
                                   map_location=lambda storage, loc: storage)
# Remove the prefix .module from the model when it is trained using DataParallel
if 'module.' in list(state_dict_checkpoint.keys())[0]:
    new_state_dict = OrderedDict()
    for k, v in state_dict_checkpoint.items():
        name = k[7:]  # remove `module.` from multi-gpu training
        new_state_dict[name] = v
else:
    new_state_dict = state_dict_checkpoint
net.load_state_dict(new_state_dict)
net.eval()
if use_cuda and gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()

#  Read image and click the points
image = np.array(Image.open(
    'RAS0019.GenevaCohort.V2_P00001177_M00008117_C0008733.ISQ_ROI_S1_8bit.tif'))
if len(image.shape) == 2:
    image = np.stack([image] * 3, -1)

plt.ion()
plt.axis('off')
plt.imshow(image)
plt.title(f'Click the {default_n_points} extreme points of the objects')
results = []
class_idx = 1
all_objects_segmentated = False
first_time = True

while not all_objects_segmentated:
    if first_time is True:
        extreme_points_ori = np.array(plt.ginput(
            default_n_points, timeout=0)).astype(np.int)
    else:
        new_point = np.array(plt.ginput(
            n_extra_points, timeout=0)).astype(np.int)
        extreme_points_ori = np.concatenate([extreme_points_ori, new_point])

    #  Crop image to the bounding box from the extreme points and resize
    bbox = helpers.get_bbox(
        image, points=extreme_points_ori, pad=pad, zero_pad=True)
    crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
    resize_image = helpers.fixed_resize(
        crop_image, (512, 512)).astype(np.float32)

    #  Generate extreme point heat map normalized to image values
    extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [pad,
                                                                                                                  pad]
    extreme_points = (512 * extreme_points *
                      [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
    extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
    extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

    #  Concatenate inputs and convert to tensor
    input_dextr = np.concatenate(
        (resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
    input_dextr = torch.from_numpy(
        input_dextr.transpose((2, 0, 1))[np.newaxis, ...])

    # Run a forward pass
    inputs = Variable(input_dextr, volatile=True)
    if use_cuda and gpu_id >= 0:
        inputs = inputs.cuda()

    outputs = net.forward(inputs)
    outputs = upsample(outputs, size=(512, 512), mode='bilinear')
    if use_cuda and gpu_id >= 0:
        outputs = outputs.cpu()

    pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0)) 
    pred = 1 / (1 + np.exp(-pred))
    pred = np.squeeze(pred)
    result = helpers.crop2fullmask(
        pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=pad) > thres

    # Plot the results
    plt.imshow(helpers.overlay_masks(image / 255, results + [result]))
    plt.plot(extreme_points_ori[:, 0], extreme_points_ori[:, 1], 'gx')

    while True:
        is_satisfied = input(
            "Are you satisifed? (y if satisfied, if n, provide the number of extra points you need)\n")
        if is_satisfied == "y":
            is_satisfied = True
            break
        else:
            try:
                n_extra_points = int(is_satisfied)
            except ValueError:
                continue
            is_satisfied = False
            break

    if is_satisfied:
        results.append(result.astype(np.uint8) * class_idx)
        while True:
            all_objects_segmentated = input(
                "Are all the objects segmentated? (y/n)\n")
            if all_objects_segmentated == "y":
                all_objects_segmentated = True
                break
            elif all_objects_segmentated == "n":
                all_objects_segmentated = False
                break
            else:
                continue
    else:
        first_time = False
        continue

    if all_objects_segmentated:
        break
    else:
        first_time = True
        class_idx += 1
        continue

h, w = image.shape[:2]
output = np.zeros((h, w))
for result in results:
    output += result
output = to_gray_scale(output)
cv2.imwrite("./output.png", output)
