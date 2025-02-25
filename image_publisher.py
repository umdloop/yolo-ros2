import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import pyzed.sl as sl
import numpy as np

class ZEDPublisher(Node):
    def __init__(self):
        super().__init__('zed_publisher')

        # ROS 2 Publishers
        self.image_pub = self.create_publisher(RosImage, 'cam_in', 10)
        self.depth_pub = self.create_publisher(RosImage, 'depth_in', 10)

        # Initialize ZED Camera
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init_params.coordinate_units = sl.UNIT.METER  # Depth in meters

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error("Failed to open ZED camera")
            return

        self.bridge = CvBridge()
        self.timer = self.create_timer(1 / 30.0, self.publish_images)  # 30 FPS

    def publish_images(self):
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Capture RGB image
            image = sl.Mat()
            self.zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()

            # Convert to ROS Image message
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.image_pub.publish(img_msg)

            # Capture depth map
            depth = sl.Mat()
            self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            depth_array = depth.get_data().astype(np.float32)  # Convert depth to NumPy

            # Convert depth to ROS message
            depth_msg = self.bridge.cv2_to_imgmsg(depth_array, encoding='32FC1')
            self.depth_pub.publish(depth_msg)

def main():
    rclpy.init()
    zed_publisher = ZEDPublisher()
    rclpy.spin(zed_publisher)
    zed_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
