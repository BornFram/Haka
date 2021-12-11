import cv2
import matplotlib.pyplot as plt
from utils import *
from darknet import Darknet

nms_threshold = 0.6

iou_threshold = 0.4
cfg_file = "cfg/yolov3.cfg"
weight_file = "weights/yolov3.weights"
namesfile = "data/coco1.names"
m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)
original_image = cv2.imread("images/men.jpg")
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
img = cv2.resize(original_image, (m.width, m.height))
boxes = detect_objects(m, img, iou_threshold, nms_threshold)
plot_boxes(original_image, boxes, class_names, plot_labels=True)