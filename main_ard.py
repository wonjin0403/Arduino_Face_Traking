from __future__ import print_function
import os
import math
import torch
import numpy as np
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import hydra
from omegaconf import DictConfig, OmegaConf
import time
from utils.utils import check_keys, remove_prefix, load_model, get_screen, get_model, prior_box

# Update the port to match your specific port
ARDUINO_PORT = '/dev/cu.usbserial-14110'

# Open serial connection to Arduino
arduino = serial.Serial(ARDUINO_PORT, 115200, timeout=1)

def read_camera_frame():
    # Send a command to the Arduino to capture a frame
    arduino.write(b'C\n')

    # Read the image size (assuming a fixed size for simplicity)
    image_size = int(arduino.readline().decode().strip())

    # Read the image data
    image_data = arduino.read(image_size)

    # Convert the image data to a NumPy array
    image_array = np.frombuffer(image_data, dtype=np.uint8)

    # Decode the image array
    frame = cv2.imdecode(image_array, 1)

    return frame

def visual_bbox(cfg: DictConfig, frame: np.array, dets: np.array) -> tuple[np.array, int, int, int]:
    b_height = 0
    end_x, end_y, end_z = frame.shape[1]//2, frame.shape[0]//2, 0
    for b in dets:
        if b[4] < cfg["vis_thres"]:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        if b_height < (b[2] - b[0]):
            b_height = b[2] - b[0]
            end_z = int(cfg["resolution_wide"] / b_height)
            end_x = int(b[0]+b[2])//2
            end_y = int(b[1]+b[3])//2
    return frame, end_x, end_y, end_z


def visual_arrow(cfg: DictConfig, frame: np.array, end_x: int, end_y: int, end_z: int) -> np.array:
    principal_point = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))
    if cfg["draw_distance"]:
        cv2.arrowedLine(frame, (int(frame.shape[1] / 2), int(frame.shape[0])), (end_x, end_z*50), (0, 255, 0), 2)
    else:
        cv2.arrowedLine(frame, principal_point, (end_x, end_y), (0, 255, 0), 2)
        cv2.arrowedLine(frame, (0, int(frame.shape[0] / 2)), (end_x, end_y), (255, 0, 0), 2)
        cv2.arrowedLine(frame, (int(frame.shape[1]), int(frame.shape[0] / 2)), (end_x, end_y), (0, 0, 255), 2)
    return frame


def show_screen(cfg: DictConfig, capture: cv2.VideoCapture, net: torch.nn.Module, device: torch.device, resize: int = 1) -> None:
    _t = {'forward_pass': Timer(), 'misc': Timer()}
    current_posX, current_posY = 0, 0


        # Display the frame
        print(frame.shape)

    while cv2.waitKey(33) < 0:
        
        # Read a frame from the Arduino
        frame = read_camera_frame()

        #ret, frame = capture.read()

        img = np.float32(frame)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        current_posX = im_height/2
        current_posY = im_width/2
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf, _ = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()

        prior_data = prior_box(cfg, (im_height, im_width), device)

        boxes = decode(loc.data.squeeze(0), prior_data, list(cfg["model"]['variance']))
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        
        # ignore low scores
        inds = np.where(scores > cfg["confidence_threshold"])[0]
        boxes = boxes[inds]
        scores = scores[inds]
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, cfg["nms_threshold"])
        dets = dets[keep, :]
        _t['misc'].toc()

        frame, end_x, end_y, end_z = visual_bbox(cfg, frame, dets)
        # face_h : b_height = 25cm : resolution_hight
        #1cm z axis: 25cm y axis = z : face_h
        frame = visual_arrow(cfg, frame, end_x, end_y, end_z)

        X_angle = math.degrees(math.atan(end_x-current_posX))
        Y_angle = math.degrees(math.atan(end_y-current_posY))
        current_posX = end_x
        current_posY = end_y
        print(X_angle, Y_angle)
        cv2.imshow("VideoFrame", np.flip(frame, axis=1))

        arduino.write(b'C\n')
        ser.write(struct.pack(X_angle, Y_angle))
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the serial connection and close the window
    arduino.close()
    cv2.destroyAllWindows()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)

    capture = get_screen(cfg)
    device = torch.device("cpu" if cfg["cpu"] else "cuda")
    net = get_model(cfg, device)
    show_screen(cfg, capture, net, device)

if __name__ == '__main__':
    main()