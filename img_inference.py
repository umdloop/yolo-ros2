import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import cv2
import threading
from ultralytics import YOLO
from custom_msgs.msg import BB 

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
        self.model = YOLO("best.pt")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self.frame_lock:
            results = self.model(frame)
            self.latest_results = results
            self.output_bounding_boxes(frame, msg.width, msg.height)
            cv2.imshow("YOLO Output", frame)
            cv2.waitKey(1)

    def output_bounding_boxes(self, frame, img_width, img_height):
        if self.latest_results:
            for result in self.latest_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{self.model.names[cls]}: {conf:.2f}"
                    print(f"Detected: {label} | Bounding Box: ({x1}, {y1}), ({x2}, {y2}) | Image Size: ({img_width}, {img_height})")
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    bb_msg = BB()
                    bb_msg.img_width = img_width
                    bb_msg.img_height = img_height
                    bb_msg.bb_top_left_x = x1
                    bb_msg.bb_top_left_y = y1
                    bb_msg.bb_bottom_right_x = x2
                    bb_msg.bb_bottom_right_y = y2
                    print(bb_msg)

def main():
    rclpy.init()
    yolo_node = YOLONode()
    rclpy.spin(yolo_node)
    
    yolo_node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()