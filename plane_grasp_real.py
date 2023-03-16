import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from UR_Robot import UR_Robot
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps
from utils.visualisation.plot import plot_grasp
import cv2
from PIL import Image
import torchvision.transforms as transforms


class PlaneGraspClass:
    def __init__(self, saved_model_path=None,use_cuda=True,visualize=False,include_rgb=True,include_depth=True,output_size=300):
        if saved_model_path==None:
            saved_model_path = 'trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93'
        self.model = torch.load(saved_model_path)
        self.device = "cuda:0" if use_cuda else "cpu"
        self.visualize = visualize

        self.cam_data = CameraData(include_rgb=include_rgb,include_depth=include_depth,output_size=output_size)

        # Load camera pose and depth scale (from running calibration)
        self.ur_robot = UR_Robot(tcp_host_ip="192.168.50.100", tcp_port=30003, workspace_limits=None, is_use_robotiq85=False,
                            is_use_camera=False)
        self.cam_pose = self.ur_robot.cam_pose
        self.intrinsic = self.ur_robot.cam_intrinsics
        self.cam_depth_scale = self.ur_robot.cam_depth_scale

        if self.visualize:
            self.fig = plt.figure(figsize=(6, 6))
        else:
            self.fig = None

    def generate(self):
        ## if you want to use RGBD from camera,use me
        # Get RGB-D image from camera
        # rgb, depth= self.ur_robot.get_camera_data() # meter
        # depth = depth *self.cam_depth_scale
        # depth[depth >1.2]=0 # distance > 1.2m ,remove it

        ## if you don't have realsense camera, use me
        num =2 # change me num=[1:6],and you can see the result in '/result' file
        rgb_path = f"./cmp{num}.png"
        depth_path = f"./hmp{num}.png"
        rgb = np.array(Image.open(rgb_path))
        depth = np.array(Image.open(depth_path)).astype(np.float32)
        depth = depth * self.cam_depth_scale
        depth[depth > 1.2] = 0  # distance > 1.2m ,remove it
        depth= np.expand_dims(depth, axis=2)
        x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)
        rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.model.predict(xc)
        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

        grasps = detect_grasps(q_img, ang_img, width_img)
        if len(grasps) ==0:
            print("Detect 0 grasp pose!")
            if self.visualize:
                plot_grasp(fig=self.fig, rgb_img=self.cam_data.get_rgb(rgb, False), grasps=grasps, save=True)
            return False
        ## For real UR robot
        # Get grasp position from model output
        # pos_z = depth[grasps[0].center[0] + self.cam_data.top_left[0], grasps[0].center[1] + self.cam_data.top_left[1]]
        # pos_x = np.multiply(grasps[0].center[1] + self.cam_data.top_left[1] - self.intrinsic[0][2],
        #                     pos_z / self.intrinsic[0][0])
        # pos_y = np.multiply(grasps[0].center[0] + self.cam_data.top_left[0] - self.intrinsic[1][2],
        #                     pos_z / self.intrinsic[1][1])
        # if pos_z == 0:
        #     return False
        #
        # target = np.asarray([pos_x, pos_y, pos_z])
        # target.shape = (3, 1)
        #
        # # Convert camera to robot coordinates
        # camera2robot = self.cam_pose
        # target_position = np.dot(camera2robot[0:3, 0:3], target) + camera2robot[0:3, 3:]
        # target_position = target_position[0:3, 0]
        #
        # # Convert camera to robot angle
        # angle = np.asarray([0, 0, grasps[0].angle])
        # angle.shape = (3, 1)
        # target_angle = np.dot(camera2robot[0:3, 0:3], angle)
        #
        # # compute gripper width
        # width = grasps[0].length # mm
        # if width < 25:    # detect error
        #     width = 70
        # elif width <40:
        #     width =45
        # elif width > 85:
        #     width = 85
        #
        # # Concatenate grasp pose with grasp angle
        # grasp_pose = np.append(target_position, target_angle[2])
        # print('grasp_pose: ', grasp_pose)
        # print('grasp_width: ',grasps[0].length)
        #
        # # np.save(self.grasp_pose, grasp_pose)
        # if self.visualize:
        #     plot_grasp(fig=self.fig, rgb_img=self.cam_data.get_rgb(rgb, False), grasps=grasps, save=True)
        #
        # success = self.ur_robot.plane_grasp([grasp_pose[0],grasp_pose[1],grasp_pose[2]-0.005], yaw=grasp_pose[3], open_size=width/100)
        # return success

        ## For having not real robot
        if self.visualize:
            plot_grasp(fig=self.fig, rgb_img=self.cam_data.get_rgb(rgb, False), grasps=grasps, save=True)
        return True


if __name__ == '__main__':
    g = PlaneGraspClass(
        saved_model_path='trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93',
        # saved_model_path='trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_42_iou_0.93',
        # saved_model_path='trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_35_iou_0.92',
        visualize=True,
        include_rgb=True
    )
    g.generate()
