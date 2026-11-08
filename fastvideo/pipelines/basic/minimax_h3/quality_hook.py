# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced quality measurement: H3Q_DUMP=dir records every step's inputs and velocities; H3Q_REPLAY=dir feeds
a candidate the recorded inputs and logs the relative L2 of its velocities against the recording."""
from __future__ import annotations

import os
from pathlib import Path

import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)


class H3QualityHook:

    def __init__(self) -> None:
        self.dump = os.environ.get("H3Q_DUMP")
        self.replay = os.environ.get("H3Q_REPLAY")
        self.request = -1

    @property
    def active(self) -> bool:
        return bool(self.dump or self.replay)

    def begin_request(self) -> None:
        self.request += 1

    def _path(self, root: str, step: int) -> Path:
        path = Path(root) / f"r{self.request}" / f"s{step}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def before(self, step: int, latents: torch.Tensor, audio: torch.Tensor) -> None:
        if self.replay:
            ref = torch.load(self._path(self.replay, step), map_location=latents.device)
            latents.copy_(ref["latents"])
            audio.copy_(ref["audio"])
        else:
            self._inputs = (latents.detach().cpu().clone(), audio.detach().cpu().clone())

    def after(self, step: int, video: torch.Tensor, audio: torch.Tensor) -> None:
        if self.replay:
            ref = torch.load(self._path(self.replay, step), map_location=video.device)
            rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()  # noqa: E731
            logger.info("H3Q r=%d s=%d rel_v=%.5f rel_a=%.5f", self.request, step, rel(video, ref["video"]),
                        rel(audio, ref["audio_v"]))
        else:
            torch.save({"latents": self._inputs[0], "audio": self._inputs[1], "video": video.detach().cpu(),
                        "audio_v": audio.detach().cpu()}, self._path(self.dump, step))


HOOK = H3QualityHook()
