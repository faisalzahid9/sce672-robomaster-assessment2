import math
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Range
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from .control import MAX_TURN_RPS, desired_command
from .fsm import State, choose_state
from .trajectory import make_trajectory
from .vision import detect_target
from .telemetry import VoltageLogger


def display(value):
    return "-" if value is None else f"{value:.3f}"


class PassivePlanner(Node):
    def __init__(self, target_id):
        super().__init__("rm_flatness_planner")
        if target_id not in (1, 2, 3):
            raise ValueError("Choose tag 1, 2, or 3")
        self.target_id = target_id
        self.state = State.SEARCHING
        self.camera_matrix = None
        self.distortion = None
        self.bearing = None
        self.last_bearing = None
        self.camera_forward = None
        self.last_seen = None
        self.tof = None
        self.tof_time = None

        self.plan = None
        self.plan_time = 0.0
        self.last_tick = time.monotonic()
        self.last_log = 0.0
        self.swept = 0.0
        self.close_mode = False
        self.close_mode_entered_at = None
        self.best_close_distance = None
        self.sweep_limit = float(
            self.declare_parameter("sweep_limit_rad", 2.0 * math.pi).value
        )
        self.tof_stop_m = float(
            self.declare_parameter("tof_stop_m", 0.20).value
        )
        self.max_search_backups = int(
            self.declare_parameter("max_search_backups", 2).value
        )
        self.search_backup_duration_s = float(
            self.declare_parameter("search_backup_duration_s", 2.0).value
        )
        self.return_home_yaw_gain = float(
            self.declare_parameter("return_home_yaw_gain", 1.0).value
        )
        self.return_home_start_err = float(
            self.declare_parameter("return_home_start_err", 0.40).value
        )
        self.return_home_stop_err = float(
            self.declare_parameter("return_home_stop_err", 0.70).value
        )
        self.return_home_timeout_s = float(
            self.declare_parameter("return_home_timeout_s", 60.0).value
        )
        self.backup_count = 0
        self.backup_started_at = None
        self.home_pose = None
        self.current_pose = None
        self.return_home_driving = False
        self.return_home_started_at = None
        self.armed = False
        self.motion_enabled = bool(
            self.declare_parameter("enable_motion", False).value
        )
        self.camera_stop_enabled = bool(
            self.declare_parameter("enable_camera_stop", False).value
        )
        self.camera_offset_m = float(
            self.declare_parameter("camera_tof_offset_m", 0.0).value
        )
        self.camera_scale = float(
            self.declare_parameter("camera_tof_scale", 1.0).value
        )
        if self.motion_enabled and not self.camera_stop_enabled:
            raise RuntimeError("Camera stop must be enabled before motion")

        self.cmd_pub = (
            self.create_publisher(Twist, "/cmd_vel", 10)
            if self.motion_enabled else None
        )
        self.voltage_log = None
        self.started_at = time.monotonic()
        self.next_log_at = 1.0
        if self.motion_enabled:
            csv_path = Path.home() / "robomaster_problem3/motor_voltage_log.csv"
            self.voltage_log = VoltageLogger(csv_path)
            self.voltage_log.record(0.0, 0.0, 0.0, self.state)
        self.get_logger().info(f"Motion enabled: {self.motion_enabled}")

        self.create_subscription(
            CameraInfo, "/camera/camera_info",
            self.on_camera_info, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, "/camera/image_color",
            self.on_image, qos_profile_sensor_data
        )
        self.create_subscription(
            Range, "/range_0",
            self.on_range, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, "/odom",
            self.on_odom, 10
        )
        self.create_timer(0.05, self.update_state)

    def set_target(self, target_id):
        if target_id not in (1, 2, 3):
            raise ValueError("Choose tag 1, 2, or 3")
        self.target_id = target_id
        self.swept = 0.0
        self.backup_count = 0
        self.backup_started_at = None
        self.plan = None
        self.plan_time = 0.0
        self.close_mode = False
        self.close_mode_entered_at = None
        self.best_close_distance = None
        self.bearing = None
        self.last_bearing = None
        self.last_seen = None
        self.camera_forward = None
        self.last_tick = time.monotonic()
        self.armed = True

        need_home = False
        if self.home_pose is not None and self.current_pose is not None:
            hx, hy, _ = self.home_pose
            cx, cy, _ = self.current_pose
            d_home = math.hypot(hx - cx, hy - cy)
            if d_home >= 0.05:
                need_home = True

        if need_home:
            self.state = State.RETURNING_HOME
            self.return_home_driving = False
            self.return_home_started_at = time.monotonic()
            self.get_logger().info(
                f"New target: T{target_id}; returning home first "
                f"(gain={self.return_home_yaw_gain:+.1f})"
            )
        else:
            self.state = State.SEARCHING
            self.get_logger().info(
                f"New target: T{target_id}; at home, starting search"
            )

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.current_pose = (float(p.x), float(p.y), float(yaw))
        if self.home_pose is None:
            self.home_pose = self.current_pose
            self.get_logger().info(
                f"Home pose: x={p.x:.3f} y={p.y:.3f} yaw={yaw:.3f}"
            )

    def on_camera_info(self, msg):
        self.camera_matrix = np.asarray(msg.k).reshape(3, 3)
        self.distortion = np.asarray(msg.d)

    def on_image(self, msg):
        if self.camera_matrix is None or msg.encoding != "bgr8":
            return

        pixels = np.frombuffer(msg.data, np.uint8).reshape(
            msg.height, msg.step
        )
        image = pixels[:, :msg.width * 3].reshape(
            msg.height, msg.width, 3
        )
        measurement = detect_target(
            image, self.camera_matrix, self.distortion, self.target_id
        )

        if measurement is None:
            self.bearing = None
            self.camera_forward = None
        else:
            self.bearing, self.camera_forward = measurement
            self.last_bearing = self.bearing
            self.last_seen = time.monotonic()
            if (self.camera_stop_enabled and
                    self.camera_scale * self.camera_forward + self.camera_offset_m <= 0.20 + 1e-6):
                self.state = State.ARRIVED
                self.publish_command(0.0, 0.0)

    def on_range(self, msg):
        reading = float(msg.range)
        if math.isfinite(reading) and msg.min_range <= reading <= msg.max_range:
            self.tof = reading
        else:
            self.tof = None
        self.tof_time = time.monotonic()
        # ToF is a safety veto only (see control.py). Do not set
        # ARRIVED from ToF: it measures the nearest obstacle, not
        # the chosen tag.

    def publish_command(self, v, yaw):
        if self.cmd_pub is None:
            return
        command = Twist()
        command.linear.x = float(v)
        command.angular.z = float(yaw)
        self.cmd_pub.publish(command)

    def update_state(self):
        now = time.monotonic()
        dt = max(0.0, now - self.last_tick)
        self.last_tick = now

        if not self.armed:
            self.publish_command(0.0, 0.0)
            if now - self.last_log >= 2.0:
                self.get_logger().info("Waiting for target selection...")
                self.last_log = now
            return

        # Odom-based return-to-home: turn to face home, then
        # drive forward with speed proportional to distance.
        # Floor 0.05 m/s keeps the chassis above its dead-band.
        if self.state == State.RETURNING_HOME:
            if self.home_pose is None or self.current_pose is None:
                self.publish_command(0.0, 0.0)
                if now - self.last_log >= 1.0:
                    self.get_logger().info(
                        "state=RETURNING_HOME waiting for odometry"
                    )
                    self.last_log = now
                return
            hx, hy, _ = self.home_pose
            cx, cy, cyaw = self.current_pose
            dx = hx - cx
            dy = hy - cy
            d_home = math.hypot(dx, dy)
            if d_home < 0.05:
                self.publish_command(0.0, 0.0)
                self.state = State.SEARCHING
                self.swept = 0.0
                self.backup_count = 0
                self.backup_started_at = None
                self.plan = None
                self.plan_time = 0.0
                self.close_mode = False
                self.close_mode_entered_at = None
                self.best_close_distance = None
                self.last_bearing = None
                self.last_seen = None
                self.get_logger().info(
                    f"Returned home (d={d_home:.3f} m); searching for T{self.target_id}"
                )
                return
            if (self.return_home_started_at is not None
                    and now - self.return_home_started_at
                        > self.return_home_timeout_s):
                self.get_logger().warning(
                    f"Return-home timeout ({self.return_home_timeout_s:.0f} s): "
                    f"d={d_home:.3f} m; aborting return, searching anyway"
                )
                self.publish_command(0.0, 0.0)
                self.state = State.SEARCHING
                self.swept = 0.0
                self.backup_count = 0
                self.backup_started_at = None
                self.return_home_driving = False
                self.return_home_started_at = None
                return
            desired_yaw = math.atan2(dy, dx)
            yaw_err = desired_yaw - cyaw
            while yaw_err > math.pi:
                yaw_err -= 2.0 * math.pi
            while yaw_err < -math.pi:
                yaw_err += 2.0 * math.pi
            abs_err = abs(yaw_err)
            # Hysteresis: start driving when well aligned, stop
            # driving only when significantly misaligned. This
            # prevents the stop-start wobble from toggling
            # around a single threshold.
            if self.return_home_driving:
                if abs_err > self.return_home_stop_err:
                    self.return_home_driving = False
            else:
                if abs_err < self.return_home_start_err:
                    self.return_home_driving = True
            yaw_cmd = self.return_home_yaw_gain * yaw_err
            yaw_cmd = max(-MAX_TURN_RPS, min(MAX_TURN_RPS, yaw_cmd))
            if self.return_home_driving:
                v_cmd = min(0.10, max(0.06, 0.6 * d_home))
            else:
                v_cmd = 0.0
            self.publish_command(v_cmd, yaw_cmd)
            if now - self.last_log >= 1.0:
                self.get_logger().info(
                    f"state=RETURNING_HOME d={d_home:.3f} m "
                    f"yaw_err={yaw_err:.3f} rad "
                    f"driving={self.return_home_driving} "
                    f"v={v_cmd:.3f} m/s yaw={yaw_cmd:.3f} rad/s"
                )
                self.last_log = now
            return

        unseen_for = (
            float("inf") if self.last_seen is None
            else now - self.last_seen
        )
        current_tof = (
            self.tof
            if self.tof_time is not None and now - self.tof_time < 0.5
            else None
        )

        previous = self.state
        camera_distance = (
            self.camera_scale * self.camera_forward + self.camera_offset_m
            if self.camera_stop_enabled and self.bearing is not None
            else None
        )
        self.state = choose_state(
            previous, unseen_for, self.bearing,
            camera_distance, current_tof
        )

        # Close mode: once the camera sees the tag at < 0.35 m,
        # commit to it. Do not fall back to searching on brief
        # detection loss (swiveling at close range only makes
        # things worse), and stop after 5 s if not yet ARRIVED.
        if camera_distance is not None and camera_distance < 0.35:
            if not self.close_mode:
                self.close_mode = True
                self.close_mode_entered_at = now
                self.best_close_distance = camera_distance
                self.get_logger().info(
                    f"Entering close mode at {camera_distance:.3f} m"
                )
            elif camera_distance < self.best_close_distance:
                self.best_close_distance = camera_distance

        if self.close_mode and self.state == State.SEARCHING:
            self.state = State.CHASING

        if (self.close_mode
                and self.state != State.ARRIVED
                and self.close_mode_entered_at is not None
                and now - self.close_mode_entered_at > 5.0):
            self.get_logger().warning(
                f"Close mode timeout: stopping at best "
                f"{display(self.best_close_distance)} m"
            )
            self.state = State.ARRIVED

        if (self.close_mode
                and self.state != State.ARRIVED
                and current_tof is not None
                and current_tof <= self.tof_stop_m + 1e-6):
            self.get_logger().warning(
                f"ToF stop at {current_tof:.3f} m "
                f"(tof_stop_m={self.tof_stop_m:.3f})"
            )
            self.state = State.ARRIVED

        if self.state != State.CHASING:
            self.plan = None
            self.plan_time = 0.0
        elif self.plan is None:
            plan_distance = camera_distance
            if plan_distance is None:
                plan_distance = current_tof
            if plan_distance is not None and plan_distance > 0.20 + 1e-6:
                self.plan = make_trajectory(plan_distance)
                self.plan_time = 0.0

        requested_v, requested_yaw = desired_command(
            self.state, self.bearing, self.last_bearing,
            unseen_for, current_tof, self.plan, self.plan_time,
            camera_distance=camera_distance
        )
        if self.state == State.CHASING and requested_v > 0.0:
            self.plan_time += dt

        # Track how far we've swept while searching; give up after a
        # full rotation without finding the chosen tag.
        if self.state == State.SEARCHING:
            self.swept += abs(requested_yaw) * dt
            if self.swept > self.sweep_limit:
                if self.backup_count < self.max_search_backups:
                    self.backup_count += 1
                    self.get_logger().warning(
                        f"Swept {self.swept:.2f} rad without finding "
                        f"T{self.target_id}; backing up "
                        f"({self.backup_count}/{self.max_search_backups})"
                    )
                    self.state = State.BACKING_UP
                    self.backup_started_at = now
                    self.swept = 0.0
                else:
                    self.get_logger().warning(
                        f"Swept {self.swept:.2f} rad without finding "
                        f"T{self.target_id} after "
                        f"{self.backup_count} backups; giving up"
                    )
                    self.state = State.SEARCH_FAILED
        else:
            self.swept = 0.0

        if self.state == State.BACKING_UP:
            if (self.backup_started_at is not None
                    and now - self.backup_started_at
                        >= self.search_backup_duration_s):
                self.get_logger().info("Backup complete, resuming search")
                self.state = State.SEARCHING
                self.backup_started_at = None
                self.swept = 0.0

        self.publish_command(requested_v, requested_yaw)

        if self.voltage_log is not None:
            elapsed = now - self.started_at
            if elapsed >= self.next_log_at:
                self.voltage_log.record(
                    elapsed, requested_v, requested_yaw,
                    self.state, current_tof
                )
                self.next_log_at += 1.0

        if self.state != previous or now - self.last_log >= 1.0:
            self.get_logger().info(
                f"state={self.state.value} "
                f"T{self.target_id}={'YES' if self.bearing is not None else 'no'} "
                f"bearing={display(self.bearing)} rad "
                f"camera_forward={display(self.camera_forward)} m "
                f"camera_equiv={display(camera_distance)} m "
                f"ToF={display(current_tof)} m "
                f"swept={self.swept:.2f} rad "
                f"request_v={requested_v:.3f} m/s "
                f"request_yaw={requested_yaw:.3f} rad/s"
            )
            self.last_log = now


def ask_for_target():
    while True:
        try:
            answer = input(
                "Which AprilTag should I visit? Enter 1, 2, or 3 (or q to quit): "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if answer in ("1", "2", "3"):
            return int(answer)
        if answer in ("q", "quit", "exit"):
            return None
        print("Please enter 1, 2, or 3, or q to quit.")


def main():
    initial_target = ask_for_target()
    if initial_target is None:
        print("No target selected; exiting.")
        return

    rclpy.init()
    node = PassivePlanner(target_id=initial_target)
    node.set_target(initial_target)

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    spin_thread.start()

    try:
        while rclpy.ok():
            while (rclpy.ok()
                    and node.state not in (State.ARRIVED, State.SEARCH_FAILED)):
                time.sleep(0.1)
            if not rclpy.ok():
                break
            if node.state == State.ARRIVED:
                node.get_logger().info(
                    f"ARRIVED at T{node.target_id}; asking for next target"
                )
            else:
                node.get_logger().warning(
                    f"SEARCH_FAILED for T{node.target_id}; asking for next target"
                )
            target_id = ask_for_target()
            if target_id is None:
                break
            node.set_target(target_id)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_command(0.0, 0.0)
        except Exception:
            pass
        try:
            spin_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            rclpy.try_shutdown()
        except BaseException:
            pass
        try:
            if node.voltage_log is not None:
                node.voltage_log.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
