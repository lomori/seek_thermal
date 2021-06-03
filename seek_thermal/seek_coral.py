#!/usr/bin/env python
"""
ON THE RASPI:

ros2 run opencv object_detector

   0------------------> x (cols) Image Frame
   |
   |        c    Camera frame
   |         o---> x
   |         |
   |         V y
   |
   V y (rows)


SUBSCRIBES TO:
    None

PUBLISHES TO:
    /composite_detector/thermal: visible light and thermal images with object detection combined.
"""

import sys
import time
import os
import threading
import subprocess

import numpy as np
import cv2

from cv_bridge import CvBridge

import rclpy
import rclpy.node
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.msg import ParameterDescriptor

from sensor_msgs.msg import Image

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

try:
  from camera import Camera
except:
  from seek_thermal.camera import Camera

# V4L2 doesn't have a concept of video vs stills modes,
# but the Pi does. Anything H264 or MJPEG is always counted
# as video. JPEG is always counted as stills.
# For raw pixel formats there is a switch at (by default)
# 1280x720 between the two.
# To change (Raspian):
# Create a file /etc/modprobe.d/bcm2835-v4l2.conf containing the text
# options bcm2835-v4l2 max_video_width=1920 max_video_height=1088
CAMERA_RESOLUTION_X = 640
CAMERA_RESOLUTION_Y = 640
CAMERA_FPS = 1

COLOR_TEXT = (0, 0, 255)

# Obtained using CV2 homography and manually set matching points
HOMOGRAPHY = np.array([
    [ 4.92827876e-01, -2.64288519e-02, -4.47105882e+00],
    [-1.31069908e-02,  5.52725560e-01, -7.36707524e+01],
    [-3.37703516e-04, -7.97689643e-05,  1.00000000e+00]
])

class CompositeDetector(rclpy.node.Node):

    #=======================================================
    # Constructor
    #=======================================================
    def __init__(self):
        super().__init__('composite_detector')
        self.logger = self.get_logger()

        default_model_dir = '/home/pi/models'
        default_model = os.path.join(default_model_dir,
                                     'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
        default_labels = os.path.join(default_model_dir,'coco_labels.txt')
        default_font = os.path.join(default_model_dir,'Ubuntu-R.ttf')
        self.declare_parameter("root_pub", "/composite_detector",
                               ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                               description='Root node for publishers'))
        self.declare_parameter("model", default_model,
                               ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                               description='Edge TPU model'))
        self.declare_parameter("labels", default_labels,
                               ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                               description='Labels for Edge TPU model'))
        self.declare_parameter("font", default_font,
                               ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                               description='Font for image annotations'))
        self.declare_parameter("top_k", 3,
                               ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER,
                               description='Maximum number of objected detected'))
        self.declare_parameter("threshold", 0.5,
                               ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                               description='Classifier score threshold'))
        self.declare_parameter("rate", 6,
                               ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER,
                               description='Update rate (Hz)'))

        font = self.get_parameter("font").value
        self.ft = cv2.freetype.createFreeType2()
        self.ft.loadFontData(fontFileName=font, id=0)

        model = self.get_parameter("model").value
        labels = self.get_parameter("labels").value
        self.logger.info(f'Loading {model} with {labels} labels.')

        self.interpreter = make_interpreter(model)
        self.interpreter.allocate_tensors()
        self.labels = read_label_file(labels)
        self.inference_size = input_size(self.interpreter)

        self.logger.info(f'Inference size: {self.inference_size}')
        self.logger.info("Initialization completed.")

    #=======================================================
    # Nice display of targets found
    #=======================================================
    def append_objs_to_img(self, cv2_image, objs, labels):
        height, width, _ = cv2_image.shape
        scale_x, scale_y = width / self.inference_size[0], height / self.inference_size[1]
        boxes = []
        titles = []
        for obj in objs:
            bbox = obj.bbox.scale(scale_x, scale_y)
            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)
            start = (x0, y0)
            end = (x1, y1)
            boxes.append([start, end])
            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
            titles.append(label)
            cv2_image = cv2.rectangle(cv2_image, start, end, (0, 255, 0), 2)
            cv2_image = self.ft.putText(img=cv2_image, text=label, org=(x0, y0+30),
                fontHeight=16, color=COLOR_TEXT, thickness=-1, line_type=cv2.LINE_AA,
                bottomLeftOrigin=False)
            #cv2_image = cv2.putText(cv2_image, label, (x0, y0+30),
            #                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        center_x = round(width/2)
        center_y = round(height/2)
        cv2_image = cv2.line(cv2_image, (center_x, center_y),
            (center_x+20, center_y), (255, 0, 0), 2)
        cv2_image = cv2.line(cv2_image, (center_x, center_y),
            (center_x, center_y+20), (0, 255, 0), 2)
        return cv2_image, boxes, titles

    #=======================================================
    # Nice display of targets found
    #=======================================================
    def append_boxes_to_img(self, cv2_image, boxes, labels,
        temperatures):
        if boxes is None:
            return cv2_image
        height, width, _= cv2_image.shape
        mask_max = np.zeros((height, width), 'uint8')
        x0, y0, x1, y1 = 0, 0, 0, 0
        for i in range(len(boxes)):
            box = boxes [i]
            start = (box[0].astype(int))
            end = (box[1].astype(int))
            x0, y0 = start
            x1, y1 = end
            x0 = min(width-1,  max(0, x0))
            y0 = min(height-1, max(0, y0))
            x1 = min(width-1,  max(0, x1))
            y1 = min(height-1, max(0, y1))

            if (x1-x0) < 5 or (y1-y0) < 5:
                continue

            tb = temperatures[y0:y1, x0:x1]
            min_temp = np.min(tb)
            max_temp = np.max(tb)

            cv2_image = cv2.rectangle(cv2_image, (x0, y0), (x1, y1), (0, 255, 0), 1)
            cv2_image = self.ft.putText(img=cv2_image, text=labels[i], org=(x0+2, y0+5),
                fontHeight=10, color=COLOR_TEXT, thickness=-1, line_type=cv2.LINE_AA,
                bottomLeftOrigin=False)
            cv2_image = self.ft.putText(img=cv2_image, text=f'Min: {min_temp:4.1f}', org=(x0+2, y0+15),
                fontHeight=10, color=COLOR_TEXT, thickness=-1, line_type=cv2.LINE_AA,
                bottomLeftOrigin=False)
            cv2_image = self.ft.putText(img=cv2_image, text=f'Max: {max_temp:4.1f}', org=(x0+2, y0+25),
                fontHeight=10, color=COLOR_TEXT, thickness=-1, line_type=cv2.LINE_AA,
                bottomLeftOrigin=False)
        mask3 = cv2.cvtColor(mask_max, cv2.COLOR_GRAY2BGR) 
        cv2_image = cv2.bitwise_or(cv2_image, mask3)
        return cv2_image

    #=======================================================
    # Make an image square without scaling it
    #=======================================================
    def make_square(self, img):
        #--- make image square to avoid different x, y scaling
        #Getting the bigger side of the image
        s = max(img.shape[0:2])
        #Creating a dark square with NUMPY  
        f = np.zeros((s,s,3),np.uint8)
        #Getting the centering position
        ax,ay = (s - img.shape[1])//2,(s - img.shape[0])//2
        #Pasting the 'image' in a centering position
        f[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img
        return f

    #=======================================================
    # Pre-process acquired image
    #=======================================================
    def pre_process_image(self, img):
        #--- our camera is mounted upsidedown
        img = cv2.flip(img, 0)
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return self.make_square(img)

    #=======================================================
    # Thread for processing targets
    #=======================================================
    def run(self):
        self.logger.info("Starting to process images...")

        root = self.get_parameter("root_pub").value
        thermal_pub = self.create_publisher(Image, root+"/thermal", 2)

        sk = Camera()
        sk.open()
        temperatures, image, filtered = sk.read_images()
        frame_rows, frame_cols = temperatures.shape
        self.logger.info(f'Thermal image resolution: {frame_cols}, {frame_rows}')

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.logger.error('Video capture could not be started')
            return

        command = "v4l2-ctl --set-ctrl=scene_mode=11"
        _ = subprocess.call(command, shell=True)

        #--- native low resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION_X)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION_Y)
        #--- speed up capture
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        frame_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.logger.info(f'Visible image resolution: {frame_width}, {frame_height}')

        #--- move this down to make it dynamically reconfigurable
        top_k = self.get_parameter("top_k").value
        threshold = self.get_parameter("threshold").value

        bridge = CvBridge()
        rate = self.create_rate(self.get_parameter("rate").value)
        while rclpy.ok():
            try:
                ret, cv2_image = cap.read()
                if not ret:
                    self.logger.warning(f'Failed to acquire image({ret})')
                    continue
                    #break
                temperatures, thermal_rgb, thermal_gray = sk.read_images()
                thermal_rgb = cv2.cvtColor(thermal_rgb, cv2.COLOR_BGRA2RGB)

                cv2_image_rgb = self.pre_process_image(cv2_image)
                cv2_image_inference = cv2.resize(cv2_image_rgb, self.inference_size)
                run_inference(self.interpreter, cv2_image_inference.tobytes())
                objs = get_objects(self.interpreter, threshold)[:top_k]
                cv2_visible_composite, boxes, titles = self.append_objs_to_img(cv2_image_rgb,
                    objs, self.labels)

                length = len(boxes)
                if length > 0:
                    pts = np.array(boxes)
                    pts = np.float32(pts).reshape(-1,1,2)
                    boxes = cv2.perspectiveTransform(pts, HOMOGRAPHY)
                    boxes = boxes.reshape(length, 2, 2)
                    thermal_composite = self.append_boxes_to_img(thermal_rgb,
                        boxes, titles, temperatures)
                else:
                    thermal_composite = thermal_rgb
                thermal_composite = self.make_square(thermal_composite)
                dim = (cv2_visible_composite.shape[1], cv2_visible_composite.shape[0])
                thermal_composite = cv2.resize(thermal_composite, dim)
                composed = np.concatenate((cv2_visible_composite, thermal_composite), axis=1)
                thermal_pub.publish(bridge.cv2_to_imgmsg(composed, "rgb8"))
            except KeyboardInterrupt:
                self.logger.info('Keyboard interrupt detected')
                break
            except Exception as e:
                self.logger.error(f'Exception processing image: {e}')
            rate.sleep()


#=======================================================
# Main entry point
#=======================================================
def main(args=None):
    rclpy.init(args=args)

    cd = CompositeDetector()

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(cd, ), daemon=True)
    thread.start()

    # Start control loop
    cd.run()

    cd.destroy_node()
    print('Finished CompositeDetector')


if __name__ == '__main__':
    main(sys.argv)
