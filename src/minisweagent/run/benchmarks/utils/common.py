"""Progress reporting for the controller benchmark."""

from minisweagent.agents.kernel_controller import KernelControllerAgent
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager


class ProgressControllerAgent(KernelControllerAgent):
    """Wrapper around kernel_controller.KernelLoopCudaOpencodeAgent (controller-managed envelope,
    cpu_gpu) with progress updates — experiment route for the decision-envelope loop."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, instance_id=instance_id, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        return super().step()
