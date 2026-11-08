# SPDX-License-Identifier: Apache-2.0
"""Encode decoded MiniMax-H3 frames while the VAE is still decoding (FASTVIDEO_H3_YUV_STREAM=1).

The VAE hands over each finalized temporal chunk's yuv420p frames (already copying into a pinned host buffer) with a
CUDA event; a writer thread waits for the copy and feeds the frames to an ffmpeg process that encodes a video-only
MP4. Saving then only muxes the audio next to it (``finish`` + ``VideoGenerator._mux_streamed_video``), so the H.264
encode overlaps the decode instead of following it. Same encoder and settings as the single-pass save.
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import threading
from typing import Any

import torch

import fastvideo.envs as envs
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class YuvStreamEncoder:
    """A video-only H.264 encode fed chunk by chunk from a pinned [frames, H * W * 3 / 2] yuv420p buffer."""

    def __init__(self, yuv: torch.Tensor, width: int, height: int, fps: int) -> None:
        self.yuv = yuv
        self.fps = fps
        self.frames = 0
        self.done = False
        self.error: BaseException | None = None
        self.dir = tempfile.mkdtemp(prefix="h3-yuv-")
        self.path = os.path.join(self.dir, "video.mp4")
        cmd = [
            shutil.which(envs.FASTVIDEO_FFMPEG_BIN.get()) or "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f",
            "rawvideo", "-pix_fmt", "yuv420p", "-s:v", f"{width}x{height}", "-r",
            str(fps), "-i", "pipe:0", "-c:v",
            envs.FASTVIDEO_VIDEO_CODEC.get(), "-preset",
            envs.FASTVIDEO_X264_PRESET.get(), "-pix_fmt",
            envs.FASTVIDEO_OUTPUT_PIX_FMT.get(), self.path
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.chunks: queue.Queue[Any] = queue.Queue()
        self.thread = threading.Thread(target=self._write, name="h3-yuv-stream", daemon=True)
        self.thread.start()

    @staticmethod
    def supported() -> bool:
        codec = envs.FASTVIDEO_VIDEO_CODEC.get()
        return (envs.FASTVIDEO_H3_YUV_STREAM.get() and not codec.endswith("_nvenc")
                and shutil.which(envs.FASTVIDEO_FFMPEG_BIN.get()) is not None)

    def chunk_ready(self, start: int, count: int, event: torch.cuda.Event) -> None:
        """Called by the VAE after it queued the copy of frames [start, start + count) and recorded ``event``."""
        self.chunks.put((start, count, event))

    def _write(self) -> None:
        try:
            while (item := self.chunks.get()) is not None:
                start, count, event = item
                event.synchronize()
                self.proc.stdin.write(memoryview(self.yuv[start:start + count].numpy()))
                self.frames = start + count
        except BaseException as error:  # reported by finish()
            self.error = error

    def finish(self) -> str | None:
        """Close the input and wait for the encode; the video-only MP4's path, or None if it failed."""
        self.done = True
        self.chunks.put(None)
        self.thread.join()
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        rc = self.proc.wait()
        if self.error is not None or rc != 0:
            logger.warning("Streamed H.264 encode failed (rc %s, %s)", rc, self.error)
            return None
        return self.path

    def abort(self) -> None:
        self.done = True
        self.chunks.put(None)
        self.proc.kill()
        self.proc.wait()
        self.thread.join(timeout=5)
        shutil.rmtree(self.dir, ignore_errors=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)
