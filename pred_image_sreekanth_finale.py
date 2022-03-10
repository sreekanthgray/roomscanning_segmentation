# Some basic setup:
# Setup detectron2 logger
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import pandas as pd
from statistics import mode
from PIL import Image
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
# from DefPredictor import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import Boxes
# from detectron2.utils.visualizer import Visualizer
from visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference_single_image

from matplotlib.image import imread
from collections import Counter
import scipy.misc
from PIL import Image  
from glob import glob


# Color Recognition
def recognize_color(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
            chex = csv.loc[i, "hex"]
    return cname, chex


def cropper(org_image_path, mask_array, out_file_name):
  img = imread(org_image_path)
  print(img.shape)
  print(mask_array.shape)
  mask_array = np.expand_dims(mask_array, axis=0)
  print(mask_array.shape)
  mask_array = np.moveaxis(mask_array, 0, -1)
  # torch.moveaxis(mask_array, 0, -1)
  mask_array = np.repeat(mask_array, 3, axis=2)

  #pixel where mask_array is false will be 255 (white),  else will show image
  output = np.where(mask_array==False, 255, img)
  im = Image.fromarray(output)
  print(f'im => {im}')
  #convert image from rgb to rgba
  im_rgba = im.convert("RGBA")
  im_data = im_rgba.getdata()

  newData = []
  for item in im_data:
    if item[0] == 255  and item[1] == 255 and item[2] == 255:
      newData.append((255,255,255,0))
    else:
      newData.append(item)
  
  im_rgba.putdata(newData)
  im_rgba.save(out_file_name, 'PNG')


def get_pixelvalues(image):
  res_col = []
  res_hex = []
  im = Image.fromarray(image)
  im_data = im.getdata()
  
  colors = Counter(im_data) ##
  asc = colors.most_common(4) ##
  print(asc) ##

  for a in asc:
    if a[0] == (255, 255, 255):
      # if a[0] == (85, 35, 28):
      asc.remove(a)

  asc = asc[:3]
  
  for (r, g, b), cnt in asc:
    temp, temp_hex = recognize_color(r, g, b)
    res_col.append(temp)
    res_hex.append(temp_hex)
  
  # width, height = im.size
  # r, g, b = im_data[int(width/2)*int(height/2)]
  # res_col = recognize_color(r,g,b)

  return res_col[0], res_hex[0]


def cropper_masks(org_image_path, mask_array):
  img = imread(org_image_path)
  print(img.shape)
  print(mask_array.shape)
  mask_array = np.expand_dims(mask_array, axis=0)
  print(mask_array.shape)
  mask_array = np.moveaxis(mask_array, 0, -1)
  # torch.moveaxis(mask_array, 0, -1)
  mask_array = np.repeat(mask_array, 3, axis=2)
  output = np.where(mask_array==False, 255, img)
  # im = Image.fromarray(output)
  # im.save(out_file_name)
  return output


def new_method(outputs, image_path):

  need = [28, 56, 57, 59, 60, 61, 62, 63, 68, 69, 71, 72, 75]
  idxofClass = [i for i, x in enumerate(list(outputs['instances'].pred_classes)) if x in need]

  o = outputs["instances"]

  #classes : a vector of N labels in range [0, num_categories)
  classes = o.pred_classes[idxofClass]

  #scores : a vector of N confidence scores
  scores = o.scores[idxofClass]

  #boxes : object storing N boxes, one for each detected instance
  boxes = o.pred_boxes[idxofClass]

  #masks : a shape (N,H,W) masks for each detected instance
  masks = o.pred_masks[idxofClass]

  # print(f'classes => {classes} scores => {scores} boxes => {boxes} masks => {masks}')

  mat_classes = []
  mat_colors = []
  mat_colorhexs = []
  # print(scores)
  # print(boxes[0])
  # print(boxes[0][0])
  # print(len(masks[0]))
  # print(len(masks[0][0]))

  for cat in range(len(boxes)):
    # print(boxes[cat])
    # print(boxes[cat][0])

    box = boxes[cat].tensor.cpu().numpy() #get bounding box for detected object
    # print(box)

    # #save the cropped image according to the mask with background being white
    # cropper(image_path, masks[cat].cpu(), f'mat_data/MaskedImages/glass/{os.path.basename(image_path)}') 

    full_masked = cropper_masks(image_path, masks[cat].cpu())

    # full_masked = cropper_masks_trans(image_path, masks[cat].cpu())

    #crop the whole image according to the bounding box
    full_masked = full_masked[int(box[0][1]):int(box[0][3]),int(box[0][0]):int(box[0][2])]

    #use the crop image for prediction
    ans = predict_image(to_pil(full_masked))
    color_img, color_hex = get_pixelvalues(full_masked)
    print("Color is: ", color_img)
    print("Color Hex is: ", color_hex)
    mat_colors.append(color_img)
    mat_colorhexs.append(color_hex)

    print("Prediction is: ", ans)
    ans_cls = mat_names[ans]
    print("Prediction Class is: ", ans_cls)
    crop_masked = Image.fromarray(full_masked)
    crop_masked.save(f'masked_{cat}.jpg')
    mat_classes.append(ans_cls)

    basename = os.path.basename(image_path)
    basename_png = basename.rstrip('jpg') + 'png'
    # print("basename_png =>" + basename_png)

    #save cropped image
    # cropper(image_path, masks[cat].cpu(), f'mat_data/MaskedImages/{(ans_cls).lower()}/{cat}_{basename_png}') 
    # cropper(image_path, masks[cat].cpu(), f'cropped_sample_images/{cat}_{basename_png}')
    cropper(image_path, masks[cat].cpu(), f'{cat}_{basename_png}')
    # cropper(image_path, masks[cat].cpu(), f'./isaac_out/test_cropped.png')

  obj = detectron2.structures.Instances(image_size=(im.shape[0], im.shape[1]))

  obj.set('pred_classes', classes)
  obj.set('scores', scores)
  obj.set('pred_boxes', boxes)
  obj.set('pred_masks', masks)
  obj.set('mat_class', mat_classes)
  obj.set('mat_color', mat_colors)
  obj.set('mat_colorhex', mat_colorhexs)

  return obj


def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index


test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])

to_pil = transforms.ToPILImage()

mat_names = ["Cloth", "Glass", "Leather","Others","Plastic","Porcelain","Steel","Wood"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model=torch.load('isaac_material_intermodel_best_test_acc_003.pth')
model=torch.load('mat_data/isaac_material_model/isaac_material_intermodel_best_test_acc_003.pth')
model.eval()

# Color Recognition Setup

index=["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)


# Segmentation Setup

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

COCO_CLASSES = {0: u'__background__',
 1: u'person',
 2: u'bicycle',
 3: u'car',
 4: u'motorcycle',
 5: u'airplane',
 6: u'bus',
 7: u'train',
 8: u'truck',
 9: u'boat',
 10: u'traffic light',
 11: u'fire hydrant',
 12: u'stop sign',
 13: u'parking meter',
 14: u'bench',
 15: u'bird',
 16: u'cat',
 17: u'dog',
 18: u'horse',
 19: u'sheep',
 20: u'cow',
 21: u'elephant',
 22: u'bear',
 23: u'zebra',
 24: u'giraffe',
 25: u'backpack',
 26: u'umbrella',
 27: u'handbag',
 28: u'tie',
 29: u'suitcase',
 30: u'frisbee',
 31: u'skis',
 32: u'snowboard',
 33: u'sports ball',
 34: u'kite',
 35: u'baseball bat',
 36: u'baseball glove',
 37: u'skateboard',
 38: u'surfboard',
 39: u'tennis racket',
 40: u'bottle',
 41: u'wine glass',
 42: u'cup',
 43: u'fork',
 44: u'knife',
 45: u'spoon',
 46: u'bowl',
 47: u'banana',
 48: u'apple',
 49: u'sandwich',
 50: u'orange',
 51: u'broccoli',
 52: u'carrot',
 53: u'hot dog',
 54: u'pizza',
 55: u'donut',
 56: u'cake',
 57: u'chair',
 58: u'couch',
 59: u'potted plant',
 60: u'bed',
 61: u'dining table',
 62: u'toilet',
 63: u'tv',
 64: u'laptop',
 65: u'mouse',
 66: u'remote',
 67: u'keyboard',
 68: u'cell phone',
 69: u'microwave',
 70: u'oven',
 71: u'toaster',
 72: u'sink',
 73: u'refrigerator',
 74: u'book',
 75: u'clock',
 76: u'vase',
 77: u'scissors',
 78: u'teddy bear',
 79: u'hair drier',
 80: u'toothbrush'}


# IMGS = glob('mat_data/Images/wood/*.jpg')
# IMGS = ['mat_data/Images/glass/*.jpg','mat_data/Images/leather/*.jpg','mat_data/Images/plastic/*.jpg','mat_data/Images/wood/*.jpg']
# IMGS = ["sample_images/348823.jpg"]
# IMGS = ['mat_data/Images/glass/glass_000008.jpg']

# IMGS = glob('sample_images/*.jpg')
# IMGS = ['sample_images/424192.jpg']
IMGS = ['tests_pointcloud/004.jpg']
# IMGS = ['sample_images/676205.jpg']
# SAV_DIR = 'isaac_best'
# SAV_DIR = 'mat_data/MaskedImages/glass'
SAV_DIR = '.'

for image_path in IMGS:
  print(image_path)
  image_name = image_path.split(os.sep)[-1]
  im_base = image_name.split('.')[0]
  im = cv2.imread(image_path)
  outputs = predictor(im)
  print("Outputs: ", outputs)
  # print(outputs["instances"].pred_classes)
  # print(outputs["instances"].pred_boxes)

  o = outputs['instances']
  ## newly added part for finding top 3 classes and their confidence

  boxes = o.pred_boxes.tensor
  scores = o.scores
  image_shape = im.shape[:2]
  score_thresh = 0.5
  nms_thresh = 0.5
  topk_per_image = 3

  ## main function to use is this fast_rcnn_inference_single
  pred_instances, kept_indices = fast_rcnn_inference_single_image(boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image)
  # boxes = pred_instances.pred_boxes.tensor
  scores = pred_instances.scores
  scores_numpy = scores.cpu().numpy()
  
  print("Scores are: ", scores)
  print(scores_numpy)

  print(scores[kept_indices])
  
  print("Pred Instances: ")
  print(pred_instances)

  for score in scores_numpy:
    ''' filter out the top 3 highest score in numpy'''
    temp = np.argpartition(-score, 3)
    result_args = temp[:3]

    temp = np.partition(-score,3)
    result = -temp[:3]
    ''' print result '''
    print(f'top 3 classes:\n1.{COCO_CLASSES[result_args[0] + 1]} - {result[0]}\n2.{COCO_CLASSES[result_args[1] + 1]} - {result[1]}\n3.{COCO_CLASSES[result_args[2] + 1]} - {result[2]}')

  # obj = onlykeep_person_class(outputs)
  # obj = onlykeepselectclass(outputs)
  obj = new_method(outputs, image_path)

  v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  # # out = v.draw_instance_predictions(obj["instances"].to("cpu"))
  out = v.draw_instance_predictions(obj.to("cpu"))

  cv2.imwrite(os.path.join(SAV_DIR, f'{im_base}_out.jpg'),out.get_image()[:, :, ::-1])
  # cv2.imshow("img",out.get_image()[:, :, ::-1])


