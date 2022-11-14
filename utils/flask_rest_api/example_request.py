# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Perform test request
"""

import pprint

import requests
import sys
DETECTION_URL = "http://localhost:8081/analytic"
IMAGE = "zidane.jpg"

import cv2
import numpy as np
import io

def cv2ToBytes(image: np.ndarray, format: str="jpg"):
    success, im_buffer = cv2.imencode("."+format,image)
    if not success:
        raise Exception("failed to convert numpy array to image")
    return im_buffer.tobytes()


cap = cv2.VideoCapture('/home/pst/Documents/6229_TimeLapse_2021-11-18_15-59-53.mp4')




while True:
    resp, img = cap.read()
    print (resp)
    if img is None or resp == False:
        break
    frame_data = cv2ToBytes(img)
    #io_buf = io.BytesIO(frame_data)
    #print (io_buf)
    response = requests.post(DETECTION_URL, files= {"image":frame_data})
    pprint.pprint(response)
