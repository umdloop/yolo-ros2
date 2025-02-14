import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import cv2

class ImagePublisher(Node):
    def __init__(self, video_path):
        super().__init__('image_publisher')
        self.publisher = self.create_publisher(RosImage, 'cam_in', 10)
        self.bridge = CvBridge()
        self.video_capture = cv2.VideoCapture(video_path)

        if not self.video_capture.isOpened():
            self.get_logger().error(f"Failed to open video file: {video_path}")
            return

        self.timer = self.create_timer(1 / 30.0, self.publish_frame)

    def publish_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher.publish(ros_image)
        else:
            self.get_logger().info("End of video reached. Stopping publisher.")
            self.destroy_node()

def main():
    rclpy.init()
    video_path = "IMG_0565.MOV"
    image_publisher = ImagePublisher(video_path)
    
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(image_publisher)
    executor.spin()
    
    image_publisher.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()