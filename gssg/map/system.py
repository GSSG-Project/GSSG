import os
import signal

import torch
import torch.multiprocessing as mp

from gssg.map.mapper import MappingProcess
from gssg.map.perception import PerceptionProcess
from gssg.scene_graph import SceneGraph
from gssg.utils.rerun_utils import RECORDING_ID, rerun_init
from gssg.utils.rerun_utils import set_enabled as rerun_set_enabled
from gssg.utils.utils import merge_ply


class SLAM:
    """Coordinates the perception and mapping subprocesses for multi-process SLAM.

    Perception reads and preprocesses frames and streams them to Mapping over a queue;
    Mapping optimizes the gaussian map and streams updates back. Owns the shared IPC
    primitives and the save path.
    """

    def __init__(self, map_params, optimization_params, dataset, args) -> None:
        self.args = args
        if args.visualize:
            print("Visualizing init")
            rerun_init(args.save_path)

        # Per-process done flags (index 0 = perception, 1 = mapper), shared across processes.
        self._end = torch.zeros(2).int().share_memory_()
        # Graceful-stop flag, set by the parent's SIGINT handler; the perception loop polls
        # it and ends scanning cleanly so the normal sentinel -> save path runs.
        self._stop = torch.zeros(1).int().share_memory_()
        self.dataset = dataset
        self.scene_graph = SceneGraph(args)

        self.sync_tracker2mapper_method = (
            map_params.sync_tracker2mapper_method
        )  # strict / loose / free
        self.sync_tracker2mapper_frames = map_params.sync_tracker2mapper_frames

        # perception -> mapper. Bounded so a slow mapper can't grow RAM: perception drops
        # the newest frame when full rather than stalling intake.
        self._mapper_queue_maxsize = int(getattr(args, "mapper_queue_maxsize", 3))
        self._tracker2mapper_call = mp.Condition()
        self._tracker2mapper_frame_queue = mp.Queue(maxsize=self._mapper_queue_maxsize)

        # mapper -> perception
        self._mapper2tracker_call = mp.Condition()
        self._mapper2tracker_map_queue = mp.Queue()

        # mapper -> system (final-model handoff for saving)
        self._mapper2system_call = mp.Condition()
        self._mapper2system_requires = [False, False]  # [reserved, save_model]
        self._mapper2system_map_queue = mp.Queue()

        self.map_process = MappingProcess(
            args, optimization_params, self, scene_graph=self.scene_graph
        )
        self.perception_process = PerceptionProcess(self, args)
        self.save_path = self.map_process.save_path

    def run(self):
        # Children must NOT handle SIGINT: Ctrl+C is delivered to the whole process group,
        # so a child would otherwise die mid-frame (KeyboardInterrupt slips past their
        # `except Exception`) and never run the final save. Each child installs SIG_IGN at
        # the top of its target (spawn resets signal handlers in the fresh interpreter).
        # The parent installs its own graceful handler below.
        processes = []
        prev_sigint = signal.getsignal(signal.SIGINT)

        def _sigint_handler(signum, frame):
            if self._stop[0] == 1:
                print(
                    "\n[STOP] FORCED EXIT on second Ctrl+C — scene may NOT be saved.",
                    flush=True,
                )
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                os._exit(130)
            print(
                "\n[STOP] Graceful stop requested. Perception will finish the current "
                "frame, then the mapper runs global fusion + final optimization + save "
                "(PLY, scene_graph, OpenLex3D). This can take 10-30 s. Press Ctrl+C "
                "AGAIN to abandon without saving.",
                flush=True,
            )
            self._stop[0] = 1

        # Install the parent handler before spawning so a Ctrl+C during startup is still
        # caught gracefully. Children override to SIG_IGN.
        signal.signal(signal.SIGINT, _sigint_handler)

        for rank in range(2):
            target = self._run_mapper if rank == 0 else self._run_perception
            print(f"Start {'mapping' if rank == 0 else 'perception'} process")
            p = mp.Process(target=target, args=(rank,))
            p.start()
            processes.append(p)

        while self._end.sum() < 2:
            with self._mapper2system_call:
                # Timeout is critical: a missed notify would otherwise hang forever.
                if self._mapper2system_requires.count(True) == 0:
                    self._mapper2system_call.wait(timeout=0.1)

                if self._mapper2system_requires[1]:
                    while not self._mapper2system_map_queue.empty():
                        print("System: receiving map for saving...")
                        map_output = self._mapper2system_map_queue.get()
                        self.save_model(map_output)
                        del map_output
                        break
                    self._mapper2system_requires[1] = False

        signal.signal(signal.SIGINT, prev_sigint)
        print("System: both processes finished.")

        # The final-model handoff usually arrives here rather than in the loop above.
        while not self._mapper2system_map_queue.empty():
            print("System: saving final residual model...")
            map_output = self._mapper2system_map_queue.get()
            self.save_model(map_output)
            del map_output

        print("System: stopping processes...")
        self.perception_process.stop()
        self.map_process.stop()
        self.release()

        print("System: joining processes...")
        for p in processes:
            p.join(timeout=2.0)
            if p.is_alive():
                print(f"System: process {p.pid} did not exit gracefully, terminating...")
                p.terminate()
                p.join()

        print("System: done.")

    def _run_perception(self, rank):
        # Ignore Ctrl+C in the child; the parent handles it and ends the loop via the
        # shared _stop flag. spawn resets signal handlers, so this must be set in the child.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if self.args.visualize:
            import rerun as rr

            print("Visualizing init perception")
            rr.init(self.args.save_path or "rerun_example", recording_id=RECORDING_ID, spawn=True)
            rerun_set_enabled(True)
        torch.manual_seed(rank)
        print("start perception")
        try:
            self.perception_process.run()
        except Exception as e:
            print(f"Perception exception: {e}")
            import traceback

            traceback.print_exc()

    def _run_mapper(self, rank):
        # Ignore Ctrl+C in the child (see _run_perception). The mapper exits via the
        # time==-1 sentinel sent by perception after the loop ends.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if self.args.visualize:
            import rerun as rr

            print("Visualizing init mapper")
            rr.init(self.args.save_path or "rerun_example", recording_id=RECORDING_ID, spawn=True)
            rerun_set_enabled(True)
        torch.manual_seed(rank)
        print("start mapping")
        try:
            self.map_process.run()
        except Exception as e:
            print(f"Mapping exception: {e}")
            import traceback

            traceback.print_exc()

    def release_mp_queue(self, mp_queue):
        """Drain and close a multiprocessing queue."""
        try:
            while not mp_queue.empty():
                try:
                    _ = mp_queue.get_nowait()
                except Exception:
                    break
            mp_queue.cancel_join_thread()
            mp_queue.close()
            mp_queue.join_thread()
        except Exception as e:
            print(f"Error closing queue: {e}")

    def release(self):
        print("System: Releasing queues...")
        self.release_mp_queue(self._tracker2mapper_frame_queue)
        self.release_mp_queue(self._mapper2system_map_queue)
        self.release_mp_queue(self._mapper2tracker_map_queue)

    def save_model(self, map_output, save_data=True, save_sibr=True, save_merge=False):
        print("System: Save Model Start")
        try:
            self.active_gaussians = map_output["active_gaussians"]
            self.stable_gaussians = map_output["stable_gaussians"]
            self.map_time = map_output["time"]
            self.map_iter = map_output["iter"]

            # Final save uses time == -1.
            if self.map_time == -1:
                frame_name = "final_result"
            else:
                frame_name = f"frame_{self.map_time:04d}"

            print(f"Saving model: {frame_name}")
            frame_save_path = os.path.join(self.save_path, "save_model", frame_name)
            os.makedirs(frame_save_path, exist_ok=True)

            path = os.path.join(frame_save_path, f"iter_{self.map_iter:04d}")

            if save_data:
                self.active_gaussians.save_model_ply(path + ".ply", include_confidence=True)
                if self.stable_gaussians.get_points_num > 0:
                    self.stable_gaussians.save_model_ply(
                        path + "_stable.ply", include_confidence=True, include_anchor=True
                    )

            if save_sibr:
                self.active_gaussians.save_model_ply(path + "_sibr.ply", include_confidence=False)
                if self.stable_gaussians.get_points_num > 0:
                    self.stable_gaussians.save_model_ply(
                        path + "_stable_sibr.ply", include_confidence=False
                    )

            has_stable = self.stable_gaussians.get_points_num > 0
            if save_data and save_merge and has_stable:
                merge_ply(
                    path + ".ply",
                    path + "_stable.ply",
                    path + "_merge.ply",
                    include_confidence=True,
                )
            if save_sibr and save_merge and has_stable:
                merge_ply(
                    path + "_sibr.ply",
                    path + "_stable_sibr.ply",
                    path + "_merge_sibr.ply",
                    include_confidence=False,
                )

            print("System: Save finish")

        except Exception as e:
            print(f"Error in save_model: {e}")
            import traceback

            traceback.print_exc()
