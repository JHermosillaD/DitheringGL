import subprocess
import datetime
import time
import numpy as np
from pathlib import Path


class Recorder:
    """
    Drop-in FFmpeg frame recorder for any ModernGL project.
    Usage:
        from recorder import Recorder
        self.recorder = Recorder(width, height, fps=30, output_dir="recordings")

        # in on_render:
        self.recorder.write_frame(self.ctx)

        # in on_key_event:
        if action == keys.ACTION_PRESS and key == keys.R:
            self.recorder.toggle()

        # in on_close:
        self.recorder.stop()
    """

    def __init__(self, width: int, height: int, fps: int = 30, output_dir: str = "recordings", crf: int = 18):
        self.width      = width
        self.height     = height
        self.fps        = fps
        self.crf        = crf
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._proc          = None
        self.recording      = False
        self.filepath       = None
        self._frame_interval = 1.0 / fps
        self._next_frame_t   = 0.0
        self._last_frame     = None

    def toggle(self):
        if self.recording:
            self.stop()
        else:
            self.start()

    def start(self):
        if self.recording:
            return

        timestamp     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = self.output_dir / f"rec_{timestamp}.mp4"

        cmd = [
            "ffmpeg",
            "-y",
            "-f",        "rawvideo",
            "-pix_fmt",  "rgb24",
            "-s",        f"{self.width}x{self.height}",
            "-r",        str(self.fps),
            "-i",        "pipe:0",
            "-vcodec",   "libx264",
            "-crf",      str(self.crf),
            "-pix_fmt",  "yuv420p",
            "-movflags", "+faststart",
            str(self.filepath),
        ]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        self.recording     = True
        self._next_frame_t = time.perf_counter()
        self._last_frame   = None
        print(f"REC started -> {self.filepath}")

    def stop(self):
        if not self.recording or self._proc is None:
            return

        self._proc.stdin.close()
        self._proc.wait()
        self._proc     = None
        self.recording = False
        print(f"REC saved -> {self.filepath}")

    def write_frame(self, ctx):

        if not self.recording or self._proc is None:
            return

        # Capture current framebuffer
        data  = ctx.screen.read(components=3)
        frame = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        frame = np.ascontiguousarray(frame[::-1])
        raw   = frame.tobytes()
        self._last_frame = raw
        now = time.perf_counter()

        wrote = 0
        while now >= self._next_frame_t:
            try:
                self._proc.stdin.write(raw)
            except BrokenPipeError:
                self.recording = False
                print("FFmpeg pipe closed unexpectedly.")
                return
            self._next_frame_t += self._frame_interval
            wrote += 1

            # Safety cap
            if wrote >= 4:
                self._next_frame_t = now + self._frame_interval
                break