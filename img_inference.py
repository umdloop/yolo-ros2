import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from custom_msgs.msg import BB  # Custom Bounding Box message

class InferenceNode(Node):
    def __init__(self):
        super().__init__('yolo_inference_node')

        # ROS 2 Subscribers
        self.image_sub = self.create_subscription(RosImage, 'cam_in', self.image_callback, 10)
        self.depth_sub = self.create_subscription(RosImage, 'depth_in', self.depth_callback, 10)

        # YOLO Model
        self.model = YOLO("best.pt")

        # Storage for latest images
        self.latest_image = None
        self.latest_depth = None
        self.bridge = CvBridge()

    def depth_callback(self, msg):
        """Store latest depth frame."""
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def image_callback(self, msg):
        """Run YOLO and get 3D positions."""
        if self.latest_depth is None:
            return  # Skip if no depth received yet

        # Convert ROS Image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLO Inference
        results = self.model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = f"{self.model.names[class_id]}: {confidence:.2f}"

                # Get center pixel for depth estimation
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                # Retrieve depth value at bounding box center
                if 0 <= center_x < self.latest_depth.shape[1] and 0 <= center_y < self.latest_depth.shape[0]:
                    depth_value = self.latest_depth[center_y, center_x]
                    if np.isnan(depth_value) or depth_value == 0:
                        continue  # Skip invalid depth

                    # Draw 3D position
                    cv2.putText(frame, f"Depth: {depth_value:.2f}m", (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Publish bounding box message (Optional)
                bb_msg = BB()
                bb_msg.img_width = msg.width
                bb_msg.img_height = msg.height
                bb_msg.bb_top_left_x = x1
                bb_msg.bb_top_left_y = y1
                bb_msg.bb_bottom_right_x = x2
                bb_msg.bb_bottom_right_y = y2
                print(bb_msg)  # You can publish this if needed

        # Display results
        cv2.imshow("YOLO + 3D Detection", frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    inference_node = InferenceNode()
    rclpy.spin(inference_node)
    inference_node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
