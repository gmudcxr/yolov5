import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

import cv2
import numpy as np
import webbrowser
from matplotlib import pyplot as plt

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

pwd = os.path.dirname(os.path.abspath(__file__))


class ObjectDetection(object):

    def __init__(self, model_path, standalone=False):
        self.weights_path = os.path.join(model_path, model_path)

        self.labels = []
        self.net = None

        self.standalone = standalone

    def get_path_basename(self, filename):
        path = os.path.dirname(filename)
        base_name = os.path.basename(filename)
        name, _ = os.path.splitext(base_name)
        return path, name

    def seg_result_filename(self, filename):
        path, name = self.get_path_basename(filename)
        name = f'{name}-seg.png'
        return os.path.join(path, name)

    def od_result_filename(self, filename):
        path, name = self.get_path_basename(filename)
        name = f'{name}-od.png'
        return os.path.join(path, name)

    def init_net(self):
        # self.net = cv2.dnn.readNetFromONNX(self.weights_path)
        pass

    def draw_image(self, image, boxes, ids, labels, confidences, offset_x=0, offset_y=0):
        image = np.ascontiguousarray(image, dtype=np.uint8)
        for idx, i in enumerate(ids):
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            x += offset_x
            y += offset_y

            # draw a bounding box rectangle and label on the image
            color = (0, 0, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}".format(labels[idx], confidences[idx])
            cv2.putText(image, text, (x + 15, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2)

        return image

    def display_img(self, image, filename=None):

        # change order
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.standalone is None:
            return

        if self.standalone:
            plt.close('all')
            fig = plt.figure(figsize=(12, 12))
            plt.axis(False)
            ax = fig.add_subplot(111)
            ax.imshow(image)
            plt.show(block=False)
        else:
            # save to file
            if filename is None:
                filename = 'image_for_show.png'
            cv2.imwrite(filename, image)
            webbrowser.open(filename)

    def display_seg_img(self, filename):
        if os.path.exists(filename):
            webbrowser.open(filename)

    def __call__(self, image, *args, draw_image=False, offset_x=0, offset_y=0, image_for_draw=None, **kwargs):
        # Load model
        device = select_device('0')
        model = DetectMultiBackend(self.weights_path, device=device)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

        # Half
        half = False
        half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()

        imgsz = (640, 640)

        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Padded resize
        img = letterbox(image, imgsz, stride=stride, auto=pt)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        augment = False
        pred = model(im, augment=augment, visualize=False)

        # NMS
        conf_thres = 0.25
        iou_thres = 0.45
        classes = None
        agnostic_nms = False
        max_det = 1000
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        im0 = image.copy()

        save_txt = True
        hide_conf = False
        hide_labels = False

        # Process predictions
        for i, det in enumerate(pred):  # per image
            line_thickness = 3
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

        cv2.imshow('result', im0)
        cv2.waitKey(0)

        return {}, None

        # # image show be in rgb order, if use cv2.imread,
        # # should run cv2.cvtColor(img, cv2.COLOR_BGR2RGB) first
        #
        # # initialize a list of colors to represent each possible class label
        # np.random.seed(42)
        # COLORS = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")
        # (H, W) = image.shape[:2]
        #
        # # determine only the "ouput" layers name which we need from YOLO
        # ln = self.net.getLayerNames()
        # ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        #
        # # construct a blob from the input image and then perform a forward pass of the YOLO object detector,
        # # giving us our bounding boxes and associated probabilities
        # blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        # self.net.setInput(blob)
        # layerOutputs = self.net.forward(ln)
        #
        # boxes = []
        # confidences = []
        # classIDs = []
        # threshold = 0.2
        # # possible classes with confidence
        # poss_dict = {}  # class: confidence
        #
        # conf_thres = 0.25  # confidence threshold
        # iou_thres = 0.45  # NMS IOU threshold
        # classes = None
        # agnostic_nms = False
        # max_det = 1000
        # # loop over each of the layer outputs
        # for output in layerOutputs:
        #     # loop over each of the detections
        #     nc = output.shape[2] - 5  # number of classes
        #     xc = output[..., 4] > conf_thres  # candidates
        #     for detection in output:
        #         # extract the class ID and confidence (i.e., probability) of
        #         # the current object detection
        #         scores = detection[5:]
        #         classID = np.argmax(scores)
        #         confidence = scores[classID]
        #
        #         if confidence > 0.0:
        #             poss_dict[self.labels[classID]] = confidence
        #
        #         # filter out weak predictions by ensuring the detected
        #         # probability is greater than the minimum probability
        #         # confidence type=float, default=0.5
        #         if confidence > threshold:
        #             # scale the bounding box coordinates back relative to the
        #             # size of the image, keeping in mind that YOLO actually
        #             # returns the center (x, y)-coordinates of the bounding
        #             # box followed by the boxes' width and height
        #             box = detection[0:4] * np.array([W, H, W, H])
        #             (centerX, centerY, width, height) = box.astype("int")
        #
        #             # use the center (x, y)-coordinates to derive the top and
        #             # and left corner of the bounding box
        #             x = int(centerX - (width / 2))
        #             y = int(centerY - (height / 2))
        #
        #             # update our list of bounding box coordinates, confidences,
        #             # and class IDs
        #             boxes.append([x, y, int(width), int(height)])
        #             confidences.append(float(confidence))
        #             classIDs.append(classID)
        #
        # # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # ids = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)
        #
        # # classes with confidence
        # print('Class with confidence in confidence desc order:')
        # print(sorted(poss_dict.items(), key=lambda x: x[1], reverse=True))
        #
        # # failed to detect objects
        # result = {}
        # result_image = None
        # if len(ids):
        #     ids = ids.flatten()
        #     # # filter ids
        #     # boxes = [boxes[i] for i in ids]
        #     # classIDs = [classIDs[i] for i in ids]
        #     confidences = [confidences[i] for i in ids]
        #     labels = [self.labels[classIDs[i]] for i in ids]
        #
        #     result = dict(zip(labels, confidences))
        #     if draw_image:
        #         image = image_for_draw if image_for_draw is not None else image
        #         result_image = self.draw_image(image, boxes, ids, labels, confidences, offset_x=offset_x,
        #                                        offset_y=offset_y)
        # return result, result_image

    @staticmethod
    def run(image, draw_image=True, standalone=False, model_path='runs/train/exp3/weights/best.pt'):
        model_path = os.path.join(pwd, model_path)
        od = ObjectDetection(model_path, standalone=standalone)
        od.init_net()
        result, drawn_image = od(image, draw_image=draw_image)
        if drawn_image is not None:
            od.display_img(drawn_image)
        return result, drawn_image

    @staticmethod
    def run_with_seg(filename, draw_image=True, standalone=False,
                     show_seg_result=True, show_od_result=True,
                     model_path='runs/train/exp3/weights/best.pt'):
        pass
        # # for draw image
        # image_for_draw = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        #
        # model_path = os.path.join(pwd, model_path)
        # od = ObjectDetection(model_path, labels_name, weights_name, config_name, standalone=standalone)
        # od.init_labels()
        # od.init_net()
        #
        # # get seg result
        # result = {}
        # drawn_image = None
        # od_image_filename = od.od_result_filename(filename)
        # seg_image_filename = od.seg_result_filename(filename)
        # for image, x, y, w, h in holo_process.clip(filename, write_seg_result_to=seg_image_filename):
        #     temp, drawn_image = od(image, draw_image=draw_image, offset_x=x, offset_y=y, image_for_draw=image_for_draw)
        #
        #     result.update(temp)
        #
        # if show_seg_result:
        #     od.display_seg_img(seg_image_filename)
        #
        # if show_od_result:
        #     od.display_img(image_for_draw, filename=od_image_filename)
        #
        # return result, drawn_image


if __name__ == '__main__':
    image_path = 'D:/IMG_20211221_203903.jpg'

    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (640, 640))
    ObjectDetection.run(image)
