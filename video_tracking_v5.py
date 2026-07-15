# =============================================================================
# FINAL UNIFIED VIDEO TRACKING WITH:
# HybridFuzzyNet (checkpoint-safe) + YOLO-12 + ByteTrack
# Threaded I/O + Side-by-Side Visualization + FPS Controls
# =============================================================================

import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from collections import deque
from ultralytics import YOLO
from kornia.color import rgb_to_hsv, hsv_to_rgb
from threading import Thread
from queue import Queue
import warnings
import os

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# ========================= USER CONTROLS =========================
DEMO_MODE = False             # True = real-time demo (faster), False = full accuracy
FPS_MODE  = "instant"               # "instant" | "avg"

ENHANCE_SIZE_FULL = 512
ENHANCE_SIZE_DEMO = 384         # Lower res for faster inference

YOLO_SKIP_FULL = 1              # YOLO every frame
YOLO_SKIP_DEMO = 3              # YOLO every N frames (tracking fills gaps)

FPS_SMOOTH_WINDOW = 30          # Window for avg FPS smoothing
USE_TORCH_COMPILE = False        # PyTorch >= 2.0 only
# ================================================================


# =============================================================================
# 1. THREADED I/O CLASSES (The Speed Boost)
# =============================================================================

class ThreadedVideoGet:
    """
    Dedicated thread for grabbing video frames.
    Prevents the main loop from waiting for I/O.
    """
    def __init__(self, src, queue_size=128):
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        self.Q = Queue(maxsize=queue_size)
        self.w = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        
    def start(self):
        t = Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            if not self.Q.full():
                ret, frame = self.stream.read()
                if not ret:
                    self.stop()
                    return
                self.Q.put(frame)
            else:
                time.sleep(0.005)

    def read(self):
        return self.Q.get()

    def running(self):
        return not (self.stopped and self.Q.empty())

    def stop(self):
        self.stopped = True
        self.stream.release()

class ThreadedVideoShow:
    def __init__(self, output_path, fps, resolution, queue_size=128):
        # CHANGE CODEC TO avc1 (H.264) for Windows compatibility
        self.writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, resolution
        )
        self.Q = Queue(maxsize=queue_size)
        self.stopped = False

    def start(self):
        t = Thread(target=self.write_loop, args=(), daemon=True)
        t.start()
        return self

    def write_loop(self):
        while True:
            if self.stopped and self.Q.empty():
                self.writer.release()
                return
            
            if not self.Q.empty():
                frame = self.Q.get()
                if frame is not None:
                    self.writer.write(frame)
            else:
                time.sleep(0.005)

    def write(self, frame):
        if not self.stopped:
            self.Q.put(frame)

    def stop(self):
        self.stopped = True
        # Block main thread until writer finishes flushing queue
        while not self.Q.empty():
            time.sleep(0.1)
        self.writer.release()


# =============================================================================
# 2. MODEL COMPONENTS
# =============================================================================

class IlluminationTransportLayer(nn.Module):
    def __init__(self, channels, K=3):
        super().__init__()
        self.K = K
        self.kappa_predictor = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1, 1, 0), nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1, 1, 0), nn.Sigmoid()
        )
        self.source_predictor = nn.Sequential(
            nn.Conv2d(channels, 1, 1, 1, 0), nn.Tanh()
        )
        self.diffusion_kernel = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        with torch.no_grad():
            self.diffusion_kernel.weight.fill_(1.0 / 9.0)
        for p in self.diffusion_kernel.parameters():
            p.requires_grad = False
        self._tau = nn.Parameter(torch.tensor(0.1))

    def forward(self, L_bottle, skip_feat):
        L = torch.sigmoid(L_bottle[:, :1, :, :])
        context = torch.cat([L_bottle, skip_feat], dim=1)
        tau = F.softplus(self._tau)
        for _ in range(self.K):
            kappa = self.kappa_predictor(context)
            source = self.source_predictor(context)
            diffusion = kappa * self.diffusion_kernel(L) - L 
            L = L + tau * (diffusion + source)
            L = torch.clamp(L, 0.0, 1.0)
        return L

class FuzzyAttentionLayer(nn.Module):
    def __init__(self, channels):
        super(FuzzyAttentionLayer, self).__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        self.alpha_predictor = nn.Conv2d(channels // 4, channels, kernel_size=1)
        self.beta_predictor = nn.Conv2d(channels // 4, channels, kernel_size=1)
        self.gamma_predictor = nn.Conv2d(channels // 4, channels, kernel_size=1)
        
    def forward(self, x):
        context = F.adaptive_avg_pool2d(x, (1, 1))
        context = self.squeeze(context)
        alpha = F.sigmoid(self.alpha_predictor(context)) * 1.0 + 0.5 
        beta = F.sigmoid(self.beta_predictor(context)) * 0.5 + 0.5 
        gamma = F.sigmoid(self.gamma_predictor(context)) * 0.2 
        x_norm = F.sigmoid(x) 
        mu = torch.pow(x_norm + 1e-6, alpha)
        nu = torch.pow(1.0 - x_norm + 1e-6, beta)
        pi = torch.clamp(1.0 - mu - nu, 0.0, 1.0)
        fuzzy_boost = mu + gamma * pi * mu
        out = x * fuzzy_boost 
        return out

class HybridFuzzyNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=48):
        super(HybridFuzzyNet, self).__init__()
        self.base_channels = base_channels
        self.conv_in = nn.Sequential(nn.Conv2d(1, base_channels, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        
        self.L_enc1 = self._make_downsample_block(base_channels, base_channels * 2) 
        self.L_enc2 = self._make_downsample_block(base_channels * 2, base_channels * 4) 
        self.L_bottleneck = self._make_upsample_conv_block(base_channels * 4, base_channels * 4)
        
        self.L_ITL = IlluminationTransportLayer(channels=base_channels * 6, K=3)
        self.L_refine = nn.Conv2d(1, 1, 3, 1, 1)
        
        self.R_enc1 = self._make_upsample_conv_block(base_channels * 3, base_channels * 2) 
        self.fuzzy_att1 = FuzzyAttentionLayer(base_channels * 2) 
        self.fuzzy_att2 = FuzzyAttentionLayer(base_channels * 2) 
        
        self.R_dec1 = self._make_upsample_conv_block(base_channels * 3, base_channels * 2) 
        self.R_dec2 = self._make_upsample_conv_block(base_channels * 2, base_channels)
        self.R_out = nn.Conv2d(base_channels, 1, 3, 1, 1) 

    def _make_downsample_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _make_upsample_conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def _downsample_feature(self, x):
        return F.avg_pool2d(x, kernel_size=2, stride=2)
        
    def forward(self, x):
        hsv_tensor = rgb_to_hsv(x)
        H, S, V = hsv_tensor[:, 0:1, :, :], hsv_tensor[:, 1:2, :, :], hsv_tensor[:, 2:3, :, :]
        feat = self.conv_in(V) 
        
        l1 = self.L_enc1(feat)
        l2 = self.L_enc2(l1)
        l_bottle = self.L_bottleneck(l2)
        l_bottle_fused = l_bottle + l2 
        l_bottle_up = F.interpolate(l_bottle_fused, scale_factor=2, mode='bilinear', align_corners=False) 
        L_half = self.L_ITL(l_bottle_up, l1)
        L_hat = torch.sigmoid(self.L_refine(
            F.interpolate(L_half, scale_factor=2, mode='bilinear', align_corners=False)
        ))

        R_feat = feat / (L_hat.expand_as(feat) + 1e-6) 
        r_feat_down = self._downsample_feature(R_feat) 
        r1 = self.R_enc1(torch.cat([r_feat_down, l1], dim=1)) 
        
        r_fuzzy = self.fuzzy_att1(r1)
        
        r_dec1_up = F.interpolate(r_fuzzy, scale_factor=2, mode='nearest')
        r_dec1_cat = torch.cat([r_dec1_up, R_feat], dim=1) 
        r_dec1 = self.R_dec1(r_dec1_cat) 
        
        r_dec1 = self.fuzzy_att2(r_dec1) 
        r_dec2 = self.R_dec2(r_dec1)
        V_enhanced = F.sigmoid(self.R_out(r_dec2))
        
        new_hsv = torch.cat([H, S, V_enhanced], dim=1)
        I_enhanced = hsv_to_rgb(new_hsv)
        
        return I_enhanced, L_hat


# =============================================================================
# 3. VIDEO PIPELINE (OPTIMIZED)
# =============================================================================

def process_video(INPUT, OUTPUT, MODEL):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- CONFIGURATION ---")
    print(f"Device      : {device}")
    print(f"Mode        : {'DEMO (Fast)' if DEMO_MODE else 'FULL (Accurate)'}")
    print(f"Torch       : {torch.__version__}")
    
    # 1. Load Enhancement Model
    print("Loading HybridFuzzyNet...")
    enhancer = HybridFuzzyNet().to(device)
    ckpt = torch.load(MODEL, map_location=device)
    enhancer.load_state_dict(ckpt["model_state_dict"], strict=False)
    enhancer.eval()

    if USE_TORCH_COMPILE and torch.__version__ >= "2.0":
        print("Compiling model (this takes ~30-60s on first run)...")
        enhancer = torch.compile(enhancer)

    # 2. Load YOLO (Try TensorRT first, then PT)
    yolo_model_path = "yolo12s.pt"
    if os.path.exists("yolo12s.engine"):
        yolo_model_path = "yolo12s.engine"
        print(f"Loading TensorRT Engine: {yolo_model_path}")
    else:
        print(f"Loading Standard YOLO: {yolo_model_path}")
    
    detector = YOLO(yolo_model_path)
    TARGET_CLASSES = [0, 39, 63]  # person, bottle, laptop

    # 3. Start Threaded Streams
    print(f"Starting Video Stream: {INPUT}")
    video_getter = ThreadedVideoGet(INPUT).start()
    
    # Wait for the first frame to get dimensions
    while video_getter.Q.empty():
        time.sleep(0.1)
        
    print(f"Video Info: {video_getter.w}x{video_getter.h} @ {video_getter.fps} FPS")
    
    # Ensure dimensions are Integers for the writer
    out_width = int(video_getter.w * 2)
    out_height = int(video_getter.h)

    video_writer = ThreadedVideoShow(
        OUTPUT, 
        video_getter.fps, 
        (out_width, out_height)
    ).start()

    # Pre-allocate transform
    enh_size = ENHANCE_SIZE_DEMO if DEMO_MODE else ENHANCE_SIZE_FULL
    yolo_skip = YOLO_SKIP_DEMO if DEMO_MODE else YOLO_SKIP_FULL
    
    transform = transforms.Compose([
        transforms.Resize((enh_size, enh_size)),
        transforms.ToTensor()
    ])

    times = deque(maxlen=FPS_SMOOTH_WINDOW)
    frame_id = 0
    last_results = None
    
    print("\nProcessing started... Press Ctrl+C to stop manually.")

    try:
        while video_getter.running():
            if not video_getter.Q.empty():
                t0 = time.time()
                frame = video_getter.read()
                
                if frame is None:
                    break

                orig = frame.copy()

                # --- A. ENHANCEMENT ---
                # Convert BGR -> RGB -> Tensor
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)

                # Inference
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                    enhanced, _ = enhancer(tensor)

                # Post-process: Tensor -> Numpy -> Resize to original
                enh = (enhanced[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                enh = cv2.resize(enh, (video_getter.w, video_getter.h))
                # Note: 'enh' is RGB from tensor, usually OpenCV expects BGR for display/write
                # But since we use RGB for YOLO/PIL, let's convert back to BGR for saving
                enh = cv2.cvtColor(enh, cv2.COLOR_RGB2BGR)

                # --- B. DETECTION ---
                if frame_id % yolo_skip == 0:
                    last_results = detector.track(
                        enh, 
                        persist=True, 
                        conf=0.05, 
                        classes=TARGET_CLASSES,
                        tracker="bytetrack.yaml", 
                        verbose=False
                    )

                # --- C. DRAWING ---
                if last_results and last_results[0].boxes.id is not None:
                    boxes = last_results[0].boxes.xyxy.cpu().numpy()
                    ids = last_results[0].boxes.id.cpu().numpy()
                    clss = last_results[0].boxes.cls.cpu().numpy()
                    
                    for box, tid, cls in zip(boxes, ids, clss):
                        x1, y1, x2, y2 = box.astype(int)
                        label = detector.names[int(cls)]
                        color = (0, 255, 0)
                        
                        cv2.rectangle(enh, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(enh, f"{label} {int(tid)}", 
                                   (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, color, 2)

                # --- D. METRICS & WRITE ---
                latency = (time.time() - t0) * 1000
                times.append(latency)
                
                if FPS_MODE == "instant":
                    fps_disp = 1000.0 / latency if latency > 0 else 0
                else:
                    fps_disp = 1000.0 / np.mean(times) if len(times) > 0 else 0

                # UI Text
                cv2.putText(enh, f"FPS: {fps_disp:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(enh, f"Lat: {latency:.1f}ms", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                # Combine Side-by-Side
                final_frame = np.hstack([orig, enh])
                
                # Push to Write Queue
                video_writer.write(final_frame)
                frame_id += 1
                
                if frame_id % 30 == 0:
                    print(f"Processed Frame {frame_id} | FPS: {fps_disp:.2f}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
        
    finally:
        print("Stopping threads and saving video...")
        video_getter.stop()
        video_writer.stop()
        print(f"Done. Output saved to: {OUTPUT}")


if __name__ == "__main__":
    process_video(
        INPUT  = "test.mp4",
        OUTPUT = "output_test.mp4",
        MODEL  = "iccv_fuzzy_net_epoch_120.pth"
    )