import cv2
import time
import torch
import numpy as np
import utils.datasets as datasets

from mss import mss
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device
from models.experimental import attempt_load



def detect_custom():
    """ Variables. """
    conf = .25
    source = 'data/test'
    weights = 'weights/best_final.pt'
    classes_path = 'data/classes.txt'
    monitor = {"top": 40, "left": 0, "width": 1024, "height": 640}

    """ Create label mapping. """
    labels = {}
    with open(classes_path) as f:
        for i, line in enumerate(f):
            labels[i] = line.strip()

    """ Initialization. """
    sct = mss()
    device = select_device('')  # need to change the argument if using gpu device.
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(640, s=stride)

    # dataset = LoadImages(source, img_size=imgsz, stride=stride)

    while True:
        last_time = time.time()

        """ Grab Image. """
        img0 = sct.grab(monitor)
        img0 = cv2.cvtColor(np.array(img0), cv2.COLOR_BGR2RGB)

        """ Read Image. """
        # img0 = cv2.imread(f'{source}/sample.png')

        """ Padded resize. """
        img = datasets.letterbox(img0, 640, stride=stride)[0]

        """ Convert. """
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()  
        img /= 255.0 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        """ Prediction. """
        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.25, 0.45)
        pred_numpy = pred[0].numpy()

        if len(pred_numpy):
            for pred in pred_numpy:
                c1, c2, label = tuple(pred[:2]), tuple(pred[2:4]), int(pred[-1])
                print(labels[label], c1, c2)

        """ Check FPS. """
        print("fps: {}\n".format(1 / (time.time() - last_time)))

        """ Window Viewer. """
        cv2.imshow("OpenCV", img0)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    detect_custom()