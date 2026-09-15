# Force headless matplotlib backend before anything imports pyplot — avoids tkinter
# "Tcl_AsyncDelete" segfaults when room-segmentation figures get GC'd from non-main threads.
import matplotlib

matplotlib.use("Agg")

import os
from argparse import ArgumentParser

import numpy as np

from gssg.utils.config_utils import _QUALITY_FILES, read_config

parser = ArgumentParser(description="Training script parameters")
parser.add_argument("--config", type=str, default="configs/datasets/replica.yaml")
parser.add_argument(
    "--quality",
    type=str,
    default=None,
    choices=list(_QUALITY_FILES),
    help="Quality preset; overrides the dataset default (base<-quality<-dataset).",
)
args = parser.parse_args()
args = read_config(args.config, quality=args.quality)
device_list = args.device_list
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in device_list)

import torch
import torch.multiprocessing as mp

from gssg.dataset_reader import Dataset
from gssg.map.system import SLAM
from gssg.utils.arguments import DatasetParams, MapParams, OptimizationParams
from gssg.utils.general_utils import safe_state
from gssg.utils.utils import save_resolved_config

torch.set_printoptions(4, sci_mode=False)
np.set_printoptions(4)
mp.set_sharing_strategy("file_system")


def main():
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser, sentinel=True)
    map_params = MapParams(parser)

    safe_state(args.quiet)
    optimization_params = optimization_params.extract(args)
    dataset_params = dataset_params.extract(args)
    map_params = map_params.extract(args)

    dataset = Dataset(dataset_params)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    slam = SLAM(map_params, optimization_params, dataset, args)
    # SLAM.__init__ rmtree's save_path (create_workspace), so persist the resolved
    # config after that and before the run.
    save_resolved_config(args, args.save_path)
    slam.run()


if __name__ == "__main__":
    main()
