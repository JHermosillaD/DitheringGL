import cv2
import numpy as np
import moderngl
import moderngl_window as mglw
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from recorder import Recorder

CAM_DEVICE = "/dev/video2"
WIN_W, WIN_H = 540, 960

class WebcamApp(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "PixelPointCloudGL with MediaPipe"
    window_size = (WIN_W, WIN_H)
    aspect_ratio = 9 / 16  
    resource_dir = Path(__file__).parent / "shaders"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize MediaPipe
        self.base_options = python.BaseOptions(model_asset_path='models/selfie_segmenter_landscape.tflite')
        self.options = vision.ImageSegmenterOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(self.options)

        # Initialize Webcam
        self.cap = cv2.VideoCapture(CAM_DEVICE, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Initialize ModernGL Geometry
        quad = np.array([
            -1.0, -1.0,  0.0,  1.0,
             1.0, -1.0,  1.0,  1.0,
             1.0,  1.0,  1.0,  0.0,
            -1.0, -1.0,  0.0,  1.0,
             1.0,  1.0,  1.0,  0.0,
            -1.0,  1.0,  0.0,  0.0,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(quad.tobytes())
        self.prog = self.load_program(
            vertex_shader='passthrough.vert',
            fragment_shader='passthrough.frag',
        )
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, '2f 2f', 'in_vert', 'in_uv')]
        )

        self.texture = self.ctx.texture((608, 1080), 3)
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Histogram texture
        self.hist_texture = self.ctx.texture((256, 1), 1, dtype='f4')
        self.hist_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hist_texture.repeat_x = False
        self.hist_smooth = np.zeros(256, dtype=np.float32)

        # Initialize Recorder
        self.recorder = Recorder(WIN_W, WIN_H, fps=30, output_dir="recordings")
        print("Press R to start/stop recording.")

    def _compute_histogram(self, frame, alpha):
        luma = (0.299 * frame[:,:,0] +
                0.587 * frame[:,:,1] +
                0.114 * frame[:,:,2]).astype(np.float32)

        mask = alpha[:,:,0]
        weights = mask.flatten()
        pixels  = luma.flatten()

        hist, _ = np.histogram(pixels, bins=256, range=(0, 255), weights=weights)
        hist = hist.astype(np.float32)
        peak = hist.max()

        if peak > 0:
            hist /= peak

        return hist

    def on_render(self, time, frame_time):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.ascontiguousarray(frame)
        h, w = frame.shape[:2]

        # Run MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(time * 1000)
        segmentation_result = self.segmenter.segment_for_video(mp_image, timestamp_ms)

        confidence = segmentation_result.confidence_masks[0].numpy_view()
        confidence = np.where(confidence > 0.9, confidence, 0.0)

        conf_uint8 = (confidence * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        conf_closed = cv2.morphologyEx(conf_uint8, cv2.MORPH_CLOSE, kernel)
        conf_blurred = cv2.GaussianBlur(conf_closed, (31, 31), 0)
        alpha = (conf_blurred / 255.0)[..., np.newaxis]

        # Pre-processing
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l_channel, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        lab = cv2.merge((cl, a_ch, b_ch))
        enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        small_frame = cv2.resize(enhanced_frame, (0, 0), fx=0.5, fy=0.5)
        smoothed_small = cv2.bilateralFilter(small_frame, d=7, sigmaColor=100, sigmaSpace=100)
        smoothed_frame = cv2.resize(smoothed_small, (w, h), interpolation=cv2.INTER_NEAREST)

        # Apply mask
        bg_image = np.zeros_like(frame, dtype=np.uint8)
        output_image = (smoothed_frame * alpha + bg_image * (1.0 - alpha)).astype(np.uint8)
        output_image = np.ascontiguousarray(output_image)

        # Compute histogram
        hist = self._compute_histogram(smoothed_frame, alpha)
        ema  = 0.15
        self.hist_smooth = ema * hist + (1.0 - ema) * self.hist_smooth
        self.hist_texture.write(self.hist_smooth.tobytes())

        # Resize main texture if needed
        if self.texture.size != (w, h):
            self.texture.release()
            self.texture = self.ctx.texture((w, h), 3)
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self.texture.write(output_image.tobytes())
        self.texture.use(location=0)
        self.hist_texture.use(location=1)

        self.prog['webcam']     = 0
        self.prog['histogram']  = 1
        self.prog['time']       = time
        self.prog['resolution'] = (WIN_W, WIN_H)

        self.ctx.clear(0.0, 0.0, 0.0)
        self.vao.render()

        self.recorder.write_frame(self.ctx)

    def on_key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if action != keys.ACTION_PRESS:
            return
        if key == keys.R:
            self.recorder.toggle()

    def on_close(self):
        self.recorder.stop()
        self.cap.release()
        self.segmenter.close()
        self.hist_texture.release()

if __name__ == '__main__':
    mglw.run_window_config(WebcamApp)