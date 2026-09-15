import math

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class SimplePoseSender(Node):
    def __init__(self, x, y):
        super().__init__("simple_pose_sender")

        self.target_x = x  # meters
        self.target_y = y  # meters
        self.target_yaw = 0.0  # radians

        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        self.current_pose = None
        self.goal_sent = False

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info(f"Initialized. Target: X={self.target_x}, Y={self.target_y}")

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def control_loop(self):
        if self.current_pose is None:
            self.get_logger().info("Waiting for Odom...", throttle_duration_sec=2)
            return

        dx = self.target_x - self.current_pose.position.x
        dy = self.target_y - self.current_pose.position.y
        distance = math.hypot(dx, dy)

        self.get_logger().info(f"Distance to goal: {distance:.2f}m", throttle_duration_sec=1)

        if distance < 0.65:
            self.get_logger().info("Target reached Stopping.")
            self.stop_robot()
            raise SystemExit

        # Resend periodically so the planner always has the current goal.
        self.send_goal()

    def send_goal(self):
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()

        goal.pose.position.x = float(self.target_x)
        goal.pose.position.y = float(self.target_y)
        goal.pose.position.z = 0.0

        self.goal_pub.publish(goal)

    def stop_robot(self):
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)


def main():
    rclpy.init()
    node = SimplePoseSender(2, 2)

    try:
        rclpy.spin(node)
    except SystemExit:
        rclpy.logging.get_logger("main").info("Goal reached, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
