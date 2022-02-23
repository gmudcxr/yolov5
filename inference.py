import os
import webbrowser

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

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

        if image_for_draw is None:
            im0 = image.copy()
        else:
            im0 = image_for_draw

        hide_conf = False
        hide_labels = False

        confidences = []
        labels = []

        line_thickness = 1
        # Process predictions
        for i, det in enumerate(pred):  # per image
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # add offset
                    xyxy[0] += offset_x
                    xyxy[1] += offset_y
                    xyxy[2] += offset_x
                    xyxy[3] += offset_y
                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    confidences.append(f'{conf:.2f}')
                    labels.append(names[c])

        result = dict(zip(labels, confidences))
        return result, im0

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
        # for draw image
        image_for_draw = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

        model_path = os.path.join(pwd, model_path)
        od = ObjectDetection(model_path, standalone=standalone)
        od.init_net()

        # get seg result
        result = {}
        drawn_image = None
        od_image_filename = od.od_result_filename(filename)
        seg_image_filename = od.seg_result_filename(filename)
        for image, x, y, w, h in holo_process.clip(filename, write_seg_result_to=seg_image_filename):
            temp, drawn_image = od(image, draw_image=draw_image, offset_x=x, offset_y=y, image_for_draw=image_for_draw)

            result.update(temp)

        if show_seg_result:
            od.display_seg_img(seg_image_filename)

        if show_od_result:
            od.display_img(image_for_draw, filename=od_image_filename)

        return result, drawn_image


if __name__ == '__main__':
    image_path = 'D:/IMG_20211221_203903.jpg'

    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (640, 640))
    ObjectDetection.run(image)
