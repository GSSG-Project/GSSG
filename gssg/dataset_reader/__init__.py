#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from gssg.dataset_reader.sources import build_source
from gssg.utils.arguments import DatasetParams


class MockSceneInfo:
    def __init__(self):
        self.nerf_normalization = {"radius": 5.0}
        self.mesh_path = ""
        self.train_cameras = []


class Dataset:
    def __init__(
        self,
        args: DatasetParams,
    ):
        self.train_cameras = {}
        self.test_cameras = {}
        self.dataset_type = args.dataset_type
        self.source = build_source(args)

        if self.source.is_streaming:
            print(f"Initializing live stream mode (dataset_type={args.dataset_type})...")
            self.scene_info = MockSceneInfo()
            self.cameras_extent = self.scene_info.nerf_normalization["radius"]
            self.mesh_path = ""
            return

        scene_info = self.source.read_scene_info()
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.mesh_path = scene_info.mesh_path
        self.scene_info = scene_info

    def getTrainCameras(self, scale=1.0):
        if self.dataset_type in ("ros", "ros2"):
            return []
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
