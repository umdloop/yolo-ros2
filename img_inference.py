import cv2
import time
import threading
from ultralytics import YOLO
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

model = YOLO("best.pt")

rclpy.init()

class YOLONode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.subscription = self.create_subscription(
            RosImage,
            'cam_in',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        self.latest_results = None
        self.frame_lock = threading.Lock()

    def image_callback(self, msg):
        
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self.frame_lock:
            frame_to_process = frame.copy()
            results = model(frame_to_process)
            self.latest_results = results
            self.output_bounding_boxes(frame)
            cv2.imshow("YOLO Output", frame)
            cv2.waitKey(1)

    def output_bounding_boxes(self, frame):
        if self.latest_results:
            for result in self.latest_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{model.names[cls]}: {conf:.2f}"
                    print(label)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

yolo_node = YOLONode()
rclpy.spin(yolo_node)

cv2.destroyAllWindows()