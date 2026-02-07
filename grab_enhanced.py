"""
Enhanced Smart Screengrabber Pro
Features: Batch processing, multi-threading, smart pre-filtering, pause/resume,
duplicate detection, scene detection, comprehensive logging, and more.
"""

import os
import sys
import cv2
import json
import pickle
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import queue
import time

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image
import numpy as np

# Deferred imports
torch = None
open_clip = None
mp = None
imagehash = None

# ===== CONFIGURATION =====
@dataclass
class Config:
    """Application configuration with validation"""
    video_folder: str = ""
    output_folder: str = ""
    fps_rate: float = 2.0  # Sample 2 frames per second (40 frames from a 20s video)
    top_k: int = 5  # Get top 5 frames instead of 2
    max_frames: int = 100
    batch_size: int = 8
    num_workers: int = 2
    model_name: str = "ViT-B-32"
    prompts: List[str] = None
    enable_face_detection: bool = True
    enable_scene_detection: bool = True
    enable_duplicate_filter: bool = True
    duplicate_threshold: int = 5
    selection_mode: str = "overall"  # or "per_minute"
    blur_threshold: float = 45.0  # Lower threshold - accept more frames, filter only very blurry ones
    min_score: float = 0.0
    jpeg_quality: int = 95
    use_lightweight_mode: bool = False
    # Performance optimization for 4K/8K video
    max_processing_width: int = 1920  # Downscale to 1080p for analysis, save original
    
    def __post_init__(self):
        if self.prompts is None:
            self.prompts = [
                "a person's face filling the frame with eyes wide open looking at camera",
                "professional close-up portrait with sharp focus on eyes",
                "person making direct eye contact with good lighting",
                "clear facial features centered in frame",
                "high quality headshot with person speaking to camera"
            ]
    
    def validate(self) -> Tuple[bool, str]:
        """Validate configuration"""
        if not self.video_folder or not os.path.isdir(self.video_folder):
            return False, "Invalid video folder"
        if not self.output_folder:
            return False, "Output folder not specified"
        if self.top_k < 1:
            return False, "Top-K must be at least 1"
        if self.max_frames < 1:
            return False, "Max frames must be at least 1"
        if self.batch_size < 1:
            return False, "Batch size must be at least 1"
        return True, ""
    
    def save(self, path: str):
        """Save config to JSON"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load config from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# ===== LOGGING SETUP =====
class GUILogHandler(logging.Handler):
    """Custom logging handler that writes to GUI"""
    def __init__(self, log_callback):
        super().__init__()
        self.log_callback = log_callback
    
    def emit(self, record):
        msg = self.format(record)
        if self.log_callback:
            self.log_callback(msg)


def setup_logging(output_folder: str, gui_callback=None) -> logging.Logger:
    """Setup file and GUI logging"""
    logger = logging.getLogger('screengrabber')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # File handler
    log_file = os.path.join(output_folder, f"screengrabber_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # GUI handler
    if gui_callback:
        gui_handler = GUILogHandler(gui_callback)
        gui_handler.setLevel(logging.INFO)
        gui_formatter = logging.Formatter('%(levelname)s: %(message)s')
        gui_handler.setFormatter(gui_formatter)
        logger.addHandler(gui_handler)
    
    return logger


# ===== UTILITIES =====
def compute_image_hash(image: Image.Image) -> str:
    """Compute perceptual hash for duplicate detection"""
    global imagehash
    if imagehash is None:
        import imagehash as ih
        imagehash = ih
    return str(imagehash.phash(image))


def is_duplicate(hash1: str, hash2: str, threshold: int = 5) -> bool:
    """Check if two hashes represent duplicate images"""
    global imagehash
    if imagehash is None:
        import imagehash as ih
        imagehash = ih
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return abs(h1 - h2) < threshold


def detect_scenes(video_path: str, threshold: float = 30.0) -> List[int]:
    """Detect scene changes in video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    scenes = [0]
    prev_frame = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            score = np.mean(diff)
            if score > threshold:
                scenes.append(frame_idx)
        
        prev_frame = gray
        frame_idx += 1
    
    cap.release()
    return scenes


def resize_for_processing(image: Image.Image, max_width: int = 1920) -> Tuple[Image.Image, float]:
    """
    Resize image for processing if it's too large (e.g., 4K/8K video).
    Returns resized image and scale factor.
    For 4K (3840x2160) -> 1080p (1920x1080) = 4x faster processing!
    """
    width, height = image.size
    
    if width <= max_width:
        # Already small enough, no resize needed
        return image, 1.0
    
    # Calculate new dimensions maintaining aspect ratio
    scale = max_width / width
    new_width = max_width
    new_height = int(height * scale)
    
    # Resize using high-quality Lanczos resampling
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized, scale


# ===== SCORING FUNCTIONS =====
class FrameScorer:
    """Handles all frame scoring operations"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.mp_face = None
        self.mp_obj = None
        self.model = None
        self.preprocess = None
        self.text_features = None
        self.device = None
        
    def init_mediapipe(self):
        """Initialize MediaPipe models"""
        if not self.config.enable_face_detection:
            return
        
        global mp
        if mp is None:
            import mediapipe as mp_module
            mp = mp_module
        
        self.logger.info("Initializing MediaPipe Face Mesh...")
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, 
            refine_landmarks=True, 
            max_num_faces=1
        )
        
        self.logger.info("Initializing MediaPipe Objectron...")
        self.mp_obj = mp.solutions.objectron.Objectron(
            static_image_mode=True, 
            max_num_objects=5, 
            model_name='Cup'
        )
    
    def init_clip_model(self, progress_callback=None):
        """Initialize CLIP model with progress tracking"""
        if self.config.use_lightweight_mode:
            self.logger.info("Using lightweight mode - skipping CLIP model")
            return
        
        global torch, open_clip
        
        # Load PyTorch
        if torch is None:
            self.logger.info("Loading PyTorch...")
            if progress_callback:
                progress_callback("Loading PyTorch...")
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            import torch as torch_module
            torch = torch_module
            self.logger.info("PyTorch loaded")
        
        # Load OpenCLIP
        if open_clip is None:
            self.logger.info("Loading OpenCLIP library...")
            if progress_callback:
                progress_callback("Loading OpenCLIP library...")
            import open_clip as open_clip_module
            open_clip = open_clip_module
            self.logger.info("OpenCLIP library loaded")
        
        # Check if model is cached
        cache_dir = os.path.expanduser("~/.cache/clip")
        model_name = self.config.model_name
        self.logger.info(f"Checking for cached model: {model_name}")
        
        if progress_callback:
            progress_callback(f"Loading {model_name} model (this may download ~350MB on first run)...")
        
        # Load model
        self.logger.info(f"Loading model: {model_name}")
        try:
            pretrained_tag = {
                "ViT-B-32": "openai",
                "ViT-B-32-quickgelu": "laion400m_e32",
                "ViT-L-14": "openai",
                "ViT-H-14": "laion2b_s32b_b79k"
            }.get(model_name, "openai")
            
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, 
                pretrained=pretrained_tag
            )
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load {model_name}: {e}")
            self.logger.info("Falling back to lightweight mode")
            self.config.use_lightweight_mode = True
            return
        
        # Prepare tokenizer and device
        if progress_callback:
            progress_callback("Preparing model for inference...")
        
        tokenizer = open_clip.get_tokenizer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()
        
        # Encode text prompts
        self.logger.info(f"Encoding {len(self.config.prompts)} text prompts...")
        text_tokens = tokenizer(self.config.prompts).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        
        self.logger.info(f"Model ready on device: {self.device}")
    
    def get_blur_score(self, image: Image.Image) -> float:
        """Calculate blur score (Laplacian variance) focused on center region"""
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Focus on center 60% of image (crop 20% from each edge)
        center_h_start = int(h * 0.2)
        center_h_end = int(h * 0.8)
        center_w_start = int(w * 0.2)
        center_w_end = int(w * 0.8)
        
        # Extract center region
        center_region = gray[center_h_start:center_h_end, center_w_start:center_w_end]
        
        # Calculate blur score on center region
        return float(cv2.Laplacian(center_region, cv2.CV_64F).var())
    
    def get_saliency_score(self, image: Image.Image) -> float:
        """Calculate saliency score"""
        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency.computeSaliency(np.array(image))
            return float(np.mean(saliency_map)) if success else 0.0
        except:
            return 0.0
    
    def get_composition_score(self, image: Image.Image) -> float:
        """Calculate composition score - rewards CENTERED subjects with good framing
        Works for any aspect ratio: landscape (16x9), portrait (9x16), or square"""
        w, h = image.size
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Detect edges and interest points
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply Gaussian blur to get smoother interest regions
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Calculate interest map (high gradient = interesting)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        interest_map = np.sqrt(grad_x**2 + grad_y**2)
        interest_map = (interest_map / interest_map.max() * 255).astype(np.uint8)
        
        # CENTER POINT - this is where the subject should be!
        center_x, center_y = w // 2, h // 2
        
        # Sample radius around center (larger area for center scoring)
        center_radius = min(w, h) // 8  # Larger radius for center
        
        def score_center_region(cx, cy, interest_map, edges, radius):
            """Score the center region - high interest and edges = well-framed subject"""
            x1, y1 = max(0, cx - radius), max(0, cy - radius)
            x2, y2 = min(w, cx + radius), min(h, cy + radius)
            
            # Interest score (high gradient/detail in center)
            roi_interest = interest_map[y1:y2, x1:x2]
            interest_score = np.mean(roi_interest) / 255.0
            
            # Edge score (strong edges in center = subject present)
            roi_edges = edges[y1:y2, x1:x2]
            edge_score = np.sum(roi_edges > 0) / (roi_edges.size + 1e-6)
            
            # Combined score (favor interest slightly more)
            return (interest_score * 0.7 + edge_score * 0.3)
        
        # Score the center region
        center_score = score_center_region(center_x, center_y, interest_map, edges, center_radius)
        
        # BONUS: Check if interest is concentrated in center vs edges
        # Good centered composition = high center interest, lower edge interest
        edge_margin = min(w, h) // 10
        edge_regions = [
            interest_map[0:edge_margin, :],  # Top
            interest_map[-edge_margin:, :],  # Bottom
            interest_map[:, 0:edge_margin],  # Left
            interest_map[:, -edge_margin:]   # Right
        ]
        edge_interest = np.mean([np.mean(region) / 255.0 for region in edge_regions])
        
        # Centering bonus: center interest > edge interest
        centering_ratio = center_score / (edge_interest + 0.1)  # Avoid division by zero
        centering_bonus = min(centering_ratio * 0.3, 0.5)  # Cap bonus at 0.5
        
        # Final composition score
        composition_score = center_score + centering_bonus
        
        # Normalize to 0-1 range
        return min(composition_score * 1.5, 1.0)  # Boost and cap at 1.0
    
    def get_face_score(self, image: Image.Image) -> Tuple[float, float]:
        """Calculate face detection score and eye openness
        Returns: (face_detected, eyes_open_score)
        """
        if not self.mp_face:
            return 0.0, 0.0
        
        try:
            rgb = np.array(image)
            results = self.mp_face.process(rgb)
            
            if not results.multi_face_landmarks:
                return 0.0, 0.0
            
            # Face detected
            landmarks = results.multi_face_landmarks[0]
            
            # Calculate eye openness using Eye Aspect Ratio (EAR)
            def eye_aspect_ratio(eye_points):
                """Calculate eye aspect ratio from landmarks"""
                # Vertical distance
                v1 = abs(eye_points[0].y - eye_points[1].y)
                # Horizontal distance
                h1 = abs(eye_points[2].x - eye_points[3].x)
                # EAR = vertical / horizontal
                return v1 / (h1 + 1e-6)
            
            # Left eye landmarks: 159 (top), 145 (bottom), 33 (left), 133 (right)
            left_eye = [
                landmarks.landmark[159],
                landmarks.landmark[145],
                landmarks.landmark[33],
                landmarks.landmark[133]
            ]
            
            # Right eye landmarks: 386 (top), 374 (bottom), 263 (left), 362 (right)
            right_eye = [
                landmarks.landmark[386],
                landmarks.landmark[374],
                landmarks.landmark[263],
                landmarks.landmark[362]
            ]
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Threshold: EAR > 0.15 typically means eyes are open
            # Scale to 0-1 range: 0.15-0.30 maps to 0-1
            eyes_open = min(max((avg_ear - 0.15) / 0.15, 0.0), 1.0)
            
            return 1.0, eyes_open
            
        except Exception as e:
            return 0.0, 0.0
    
    def get_prompt_score(self, image: Image.Image) -> float:
        """Calculate CLIP prompt similarity score"""
        if self.config.use_lightweight_mode or self.model is None:
            return 0.0
        
        try:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_feature = self.model.encode_image(image_input)
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_feature @ self.text_features.T).squeeze(0)
                return float(similarity.max().item())
        except:
            return 0.0
    
    def score_frame(self, image: Image.Image) -> Dict[str, float]:
        """Calculate comprehensive frame score - focused on CENTERED COMPOSITION, CENTER SHARPNESS, and OPEN EYES
        Weights dynamically adjust based on whether CLIP is enabled"""
        face_detected, eyes_open = self.get_face_score(image)
        
        scores = {
            'blur': self.get_blur_score(image),
            'saliency': self.get_saliency_score(image),
            'composition': self.get_composition_score(image),
            'face': face_detected,
            'eyes_open': eyes_open,
            'prompt': self.get_prompt_score(image)
        }
        
        # DYNAMIC WEIGHTS: Change based on whether CLIP is enabled
        if self.config.use_lightweight_mode:
            # LIGHTWEIGHT MODE: Pure quality-focused (no CLIP)
            weights = {
                'prompt': 0.0,  # No CLIP
                'face': 40.0,  # Face detected
                'eyes_open': 80.0,  # Open eyes
                'saliency': 50.0,  # Visual interest
                'blur': 2.0,  # Center sharpness
                'composition': 100.0  # Centered composition - PRIMARY
            }
        else:
            # CLIP ENABLED: Semantic matching becomes PRIMARY factor
            weights = {
                'prompt': 150.0,  # CLIP similarity - NOW DOMINANT! (was 50)
                'face': 60.0,  # Face detected (increased)
                'eyes_open': 100.0,  # Open eyes (increased)
                'saliency': 40.0,  # Visual interest (decreased)
                'blur': 2.0,  # Center sharpness (unchanged)
                'composition': 80.0  # Centered composition (decreased from 100)
            }
        
        scores['total'] = sum(scores[k] * weights.get(k, 1.0) for k in scores)
        
        # Bonuses adjust based on mode
        if self.config.use_lightweight_mode:
            # Quality-focused bonuses
            if scores['composition'] > 0.7:
                scores['total'] += 50.0  # Exceptional centered composition bonus!
            if scores['blur'] > 100.0:
                scores['total'] += 30.0  # Very sharp bonus!
            if scores['face'] > 0 and scores['eyes_open'] > 0.7:
                scores['total'] += 30.0  # Wide open eyes bonus!
        else:
            # CLIP-focused bonuses
            if scores['prompt'] > 30.0:
                scores['total'] += 60.0  # Strong CLIP match bonus!
            if scores['prompt'] > 25.0 and scores['eyes_open'] > 0.7:
                scores['total'] += 40.0  # CLIP + open eyes combo!
            if scores['composition'] > 0.6 and scores['prompt'] > 25.0:
                scores['total'] += 50.0  # Good composition + CLIP match!
        
        return scores


# ===== PROCESSING STATE =====
class ProcessingState:
    """Manages processing state for pause/resume/cancel"""
    def __init__(self):
        self.is_running = False
        self.is_paused = False
        self.should_cancel = False
        self.current_video = ""
        self.processed_videos = []
        self.lock = __import__('threading').Lock()
    
    def start(self):
        with self.lock:
            self.is_running = True
            self.should_cancel = False
    
    def pause(self):
        with self.lock:
            self.is_paused = True
    
    def resume(self):
        with self.lock:
            self.is_paused = False
    
    def cancel(self):
        with self.lock:
            self.should_cancel = True
    
    def is_cancelled(self) -> bool:
        with self.lock:
            return self.should_cancel
    
    def wait_if_paused(self):
        while True:
            with self.lock:
                if not self.is_paused or self.should_cancel:
                    break
            time.sleep(0.1)


# ===== VIDEO PROCESSOR =====
class VideoProcessor:
    """Main video processing engine"""
    
    def __init__(self, config: Config, logger: logging.Logger, state: ProcessingState):
        self.config = config
        self.logger = logger
        self.state = state
        self.scorer = FrameScorer(config, logger)
        self.seen_hashes = set()
    
    def initialize(self, progress_callback=None):
        """Initialize all components"""
        try:
            # Validate configuration
            valid, error = self.config.validate()
            if not valid:
                raise ValueError(error)
            
            # Create output directory
            os.makedirs(self.config.output_folder, exist_ok=True)
            
            # Initialize components
            self.scorer.init_mediapipe()
            self.scorer.init_clip_model(progress_callback)
            
            self.logger.info("Initialization complete")
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def process_video(self, video_path: str, progress_callback=None) -> List[Tuple[Image.Image, Dict, int]]:
        """Process a single video and return scored frames"""
        self.logger.info(f"Processing: {os.path.basename(video_path)}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Could not open video: {video_path}")
            return []
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = frame_count / fps if fps > 0 else 0
            
            # Get first frame to check orientation
            ret, first_frame = cap.read()
            if ret:
                h, w = first_frame.shape[:2]
                orientation = "VERTICAL (9x16)" if h > w else "HORIZONTAL (16x9)" if w > h else "SQUARE"
                self.logger.info(f"Video: {frame_count} frames, {fps:.2f} FPS, {duration_sec:.1f}s, {w}x{h} {orientation}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
            else:
                self.logger.info(f"Video: {frame_count} frames, {fps:.2f} FPS, {duration_sec:.1f}s")
            
            # Detect scenes if enabled
            scenes = []
            if self.config.enable_scene_detection:
                self.logger.info("Detecting scenes...")
                scenes = detect_scenes(video_path)
                self.logger.info(f"Found {len(scenes)} scene changes")
            
            # Extract and score frames
            frames_data = []
            frame_interval = max(1, int(fps / self.config.fps_rate))
            count = 0
            extracted = 0
            sampled = 0
            blur_rejected = 0
            dup_rejected = 0
            score_rejected = 0
            
            self.logger.info(f"Frame interval: {frame_interval} (sampling every {frame_interval} frames)")
            
            while count < frame_count and extracted < self.config.max_frames:
                # Check for pause/cancel
                self.state.wait_if_paused()
                if self.state.is_cancelled():
                    self.logger.info("Processing cancelled")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                if count % frame_interval == 0:
                    sampled += 1
                    
                    # Convert to PIL Image (original resolution for saving later)
                    image_full = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    # Resize for processing if 4K/8K (MUCH faster!)
                    image_processing, scale = resize_for_processing(image_full, self.config.max_processing_width)
                    if scale < 1.0:
                        # Only log resize on first occurrence
                        if sampled == 1:
                            orig_w, orig_h = image_full.size
                            proc_w, proc_h = image_processing.size
                            self.logger.info(f"🚀 4K OPTIMIZATION: Resizing {orig_w}x{orig_h} → {proc_w}x{proc_h} for analysis (~{1/scale**2:.1f}x faster!)")
                    
                    # Quick pre-filtering: check blur (on processing size for speed)
                    blur_score = self.scorer.get_blur_score(image_processing)
                    if blur_score < self.config.blur_threshold:
                        blur_rejected += 1
                        # Log first 10 blur rejections to see typical values
                        if blur_rejected <= 10:
                            self.logger.info(f"  Frame {count} BLUR REJECT: {blur_score:.2f} < threshold {self.config.blur_threshold}")
                        count += 1
                        continue
                    
                    # Check for duplicates (use processing size for speed)
                    if self.config.enable_duplicate_filter:
                        img_hash = compute_image_hash(image_processing)
                        is_dup = any(is_duplicate(img_hash, h, self.config.duplicate_threshold) 
                                   for h in self.seen_hashes)
                        if is_dup:
                            dup_rejected += 1
                            count += 1
                            continue
                        self.seen_hashes.add(img_hash)
                    
                    # Full scoring (use processing size for ALL analysis - MUCH faster on 4K!)
                    scores = self.scorer.score_frame(image_processing)
                    
                    # Log every sampled frame for first 10, then every 10th
                    if sampled <= 10 or sampled % 10 == 0:
                        face_str = "FACE" if scores['face'] > 0 else "no-face"
                        eyes_str = f"eyes={scores['eyes_open']:.2f}" if scores['face'] > 0 else ""
                        self.logger.info(f"  Frame {count}: {face_str} {eyes_str} blur={blur_score:.0f} comp={scores['composition']:.2f} sal={scores['saliency']:.4f} → total={scores['total']:.1f} (min={self.config.min_score})")
                    
                    if scores['total'] >= self.config.min_score:
                        # Store FULL resolution image for saving (not the downscaled one!)
                        frames_data.append((image_full, scores, count))
                        extracted += 1
                        
                        if progress_callback and extracted % 10 == 0:
                            progress_callback(f"Extracted {extracted} frames...")
                    else:
                        score_rejected += 1
                        # Log score rejections for first few
                        if score_rejected <= 3:
                            self.logger.info(f"    ↳ REJECTED (score {scores['total']:.1f} < min {self.config.min_score})")
                
                count += 1
            
            self.logger.info(f"━━━ Sampling stats ━━━")
            self.logger.info(f"  Total frames in video: {frame_count}")
            self.logger.info(f"  Sampled (every {frame_interval}): {sampled}")
            self.logger.info(f"  Rejected (blur < {self.config.blur_threshold}): {blur_rejected}")
            self.logger.info(f"  Rejected (duplicate): {dup_rejected}")
            self.logger.info(f"  Rejected (low score): {score_rejected}")
            self.logger.info(f"  ✓ ACCEPTED: {extracted}")
            self.logger.info(f"━━━━━━━━━━━━━━━━━━━━━")
            
            cap.release()
            self.logger.info(f"Extracted {len(frames_data)} candidate frames")
            
            return frames_data
            
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            cap.release()
            return []
    
    def select_best_frames(self, frames_data: List[Tuple[Image.Image, Dict, int]], 
                          video_duration: float) -> List[Tuple[Image.Image, Dict, int]]:
        """Select best frames based on selection mode"""
        if not frames_data:
            return []
        
        if self.config.selection_mode == "per_minute":
            # Select best K frames per minute
            minutes = int(video_duration / 60) + 1
            frames_per_minute = [[] for _ in range(minutes)]
            
            for img, scores, frame_idx in frames_data:
                minute = int((frame_idx / 30) / 60)  # Assuming 30fps
                if minute < minutes:
                    frames_per_minute[minute].append((img, scores, frame_idx))
            
            selected = []
            for minute_frames in frames_per_minute:
                if minute_frames:
                    sorted_frames = sorted(minute_frames, key=lambda x: x[1]['total'], reverse=True)
                    selected.extend(sorted_frames[:self.config.top_k])
            
            return selected
        else:
            # Select overall best K frames
            sorted_frames = sorted(frames_data, key=lambda x: x[1]['total'], reverse=True)
            return sorted_frames[:self.config.top_k]
    
    def save_frames(self, frames_data: List[Tuple[Image.Image, Dict, int]], video_name: str):
        """Save selected frames to disk"""
        save_dir = os.path.join(self.config.output_folder, "screengrabs")
        os.makedirs(save_dir, exist_ok=True)
        
        base_name = os.path.splitext(video_name)[0]
        
        for i, (img, scores, frame_idx) in enumerate(frames_data):
            filename = f"{base_name}_frame{frame_idx}_score{int(scores['total'])}_rank{i+1}.jpg"
            filepath = os.path.join(save_dir, filename)
            img.save(filepath, quality=self.config.jpeg_quality)
            
            # Log detailed scoring breakdown
            score_breakdown = f"comp:{scores['composition']:.2f} blur:{scores['blur']:.0f} eyes:{scores['eyes_open']:.2f} sal:{scores['saliency']:.2f} face:{scores['face']:.1f}"
            self.logger.info(f"Saved: {filename}")
            self.logger.info(f"  └─ Scores: {score_breakdown}")
    
    def process_all_videos(self, progress_callback=None):
        """Process all videos in the input folder"""
        video_files = []
        for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.mxf']:
            video_files.extend(Path(self.config.video_folder).glob(f"*{ext}"))
            video_files.extend(Path(self.config.video_folder).glob(f"*{ext.upper()}"))
        
        video_files = list(set(video_files))  # Remove duplicates
        total_videos = len(video_files)
        
        self.logger.info(f"Found {total_videos} videos to process")
        
        if progress_callback:
            progress_callback(('max', total_videos))
        
        for idx, video_path in enumerate(video_files):
            if self.state.is_cancelled():
                break
            
            self.state.current_video = str(video_path)
            
            if progress_callback:
                progress_callback(('video', f"Processing {video_path.name}..."))
            
            # Process video
            frames_data = self.process_video(str(video_path), progress_callback)
            
            if frames_data:
                # Get video duration for per-minute mode
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                # Select best frames
                selected = self.select_best_frames(frames_data, duration)
                
                # Save frames
                self.save_frames(selected, video_path.name)
                
                self.state.processed_videos.append(str(video_path))
            
            if progress_callback:
                progress_callback(('progress', idx + 1))
        
        self.logger.info(f"Processing complete. Processed {len(self.state.processed_videos)} videos")


# ===== GUI =====
class EnhancedGUI:
    """Enhanced GUI with all features"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Smart Screengrabber Pro - Enhanced Edition")
        self.root.geometry("900x750")
        
        # State
        self.config = Config()
        self.state = ProcessingState()
        self.logger = None
        self.processor = None
        self.config_file = os.path.expanduser("~/.screengrabber_config.json")
        
        # Load last config if exists
        self.load_last_config()
        
        # Variables
        self.video_folder = tk.StringVar(value=self.config.video_folder)
        self.output_folder = tk.StringVar(value=self.config.output_folder)
        self.fps_rate = tk.DoubleVar(value=self.config.fps_rate)
        self.top_k = tk.IntVar(value=self.config.top_k)
        self.max_frames = tk.IntVar(value=self.config.max_frames)
        self.batch_size = tk.IntVar(value=self.config.batch_size)
        self.model_choice = tk.StringVar(value=self.config.model_name)
        self.selection_mode = tk.StringVar(value=self.config.selection_mode)
        self.use_lightweight = tk.BooleanVar(value=self.config.use_lightweight_mode)
        self.enable_duplicates = tk.BooleanVar(value=self.config.enable_duplicate_filter)
        self.enable_scenes = tk.BooleanVar(value=self.config.enable_scene_detection)
        self.blur_threshold = tk.DoubleVar(value=self.config.blur_threshold)
        self.custom_prompts = tk.StringVar(value=", ".join(self.config.prompts))
        
        self.build_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def load_last_config(self):
        """Load last used configuration"""
        if os.path.exists(self.config_file):
            try:
                self.config = Config.load(self.config_file)
            except:
                pass
    
    def save_current_config(self):
        """Save current configuration"""
        self.update_config_from_gui()
        try:
            self.config.save(self.config_file)
        except:
            pass
    
    def update_config_from_gui(self):
        """Update config object from GUI values"""
        self.config.video_folder = self.video_folder.get()
        self.config.output_folder = self.output_folder.get()
        self.config.fps_rate = self.fps_rate.get()
        self.config.top_k = self.top_k.get()
        self.config.max_frames = self.max_frames.get()
        self.config.batch_size = self.batch_size.get()
        self.config.model_name = self.model_choice.get()
        self.config.selection_mode = self.selection_mode.get()
        self.config.use_lightweight_mode = self.use_lightweight.get()
        self.config.enable_duplicate_filter = self.enable_duplicates.get()
        self.config.enable_scene_detection = self.enable_scenes.get()
        
        # Parse custom prompts
        prompts_text = self.custom_prompts.get().strip()
        if prompts_text:
            self.config.prompts = [p.strip() for p in prompts_text.split(",") if p.strip()]
    
    def build_gui(self):
        """Build the GUI interface"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        row = 0
        
        # ===== FOLDER SELECTION =====
        ttk.Label(main_frame, text="📁 Input/Output", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0,5))
        row += 1
        
        ttk.Label(main_frame, text="Video Folder:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.video_folder, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E))
        ttk.Button(main_frame, text="Browse", command=self.select_video_folder).grid(row=row, column=2, padx=5)
        row += 1
        
        ttk.Label(main_frame, text="Output Folder:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.output_folder, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E))
        ttk.Button(main_frame, text="Browse", command=self.select_output_folder).grid(row=row, column=2, padx=5)
        row += 1
        
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # ===== PROCESSING OPTIONS =====
        ttk.Label(main_frame, text="⚙️ Processing Options", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0,5))
        row += 1
        
        ttk.Label(main_frame, text="Model:").grid(row=row, column=0, sticky=tk.W)
        models = ["ViT-B-32", "ViT-B-32-quickgelu", "ViT-L-14"]
        ttk.Combobox(main_frame, textvariable=self.model_choice, values=models, width=47).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(main_frame, text="FPS Rate:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.fps_rate, width=20).grid(row=row, column=1, sticky=tk.W)
        ttk.Label(main_frame, text="(frames per second to sample)").grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        ttk.Label(main_frame, text="Top-K Frames:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.top_k, width=20).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(main_frame, text="Max Frames/Video:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.max_frames, width=20).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(main_frame, text="Selection Mode:").grid(row=row, column=0, sticky=tk.W)
        ttk.Radiobutton(main_frame, text="Best Overall", variable=self.selection_mode, value="overall").grid(row=row, column=1, sticky=tk.W)
        row += 1
        ttk.Radiobutton(main_frame, text="Best Per Minute", variable=self.selection_mode, value="per_minute").grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # ===== ADVANCED OPTIONS =====
        ttk.Label(main_frame, text="🔧 Advanced Features", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0,5))
        row += 1
        
        ttk.Checkbutton(main_frame, text="Enable Duplicate Detection", variable=self.enable_duplicates).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        ttk.Checkbutton(main_frame, text="Enable Scene Detection", variable=self.enable_scenes).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        ttk.Checkbutton(main_frame, text="Lightweight Mode (no CLIP model)", variable=self.use_lightweight).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        ttk.Label(main_frame, text="Custom Prompts:").grid(row=row, column=0, sticky=tk.W, pady=(5,0))
        ttk.Label(main_frame, text="(comma-separated, only used if CLIP enabled)", font=('Arial', 9, 'italic')).grid(row=row, column=1, sticky=tk.W, pady=(5,0))
        row += 1
        ttk.Entry(main_frame, textvariable=self.custom_prompts, width=50).grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0,5))
        row += 1
        
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # ===== LOG OUTPUT =====
        ttk.Label(main_frame, text="📋 Processing Log", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0,5))
        row += 1
        
        self.log_output = scrolledtext.ScrolledText(main_frame, width=80, height=12, state='disabled')
        self.log_output.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        row += 1
        
        # ===== PROGRESS BAR =====
        self.progress_label = ttk.Label(main_frame, text="Ready")
        self.progress_label.grid(row=row, column=0, columnspan=3, sticky=tk.W)
        row += 1
        
        self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal", length=600, mode="determinate")
        self.progress_bar.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        # ===== CONTROL BUTTONS =====
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=10)
        
        self.start_btn = ttk.Button(button_frame, text="▶ Start", command=self.on_start, width=15)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.pause_btn = ttk.Button(button_frame, text="⏸ Pause", command=self.on_pause, width=15, state='disabled')
        self.pause_btn.grid(row=0, column=1, padx=5)
        
        self.cancel_btn = ttk.Button(button_frame, text="⏹ Cancel", command=self.on_cancel, width=15, state='disabled')
        self.cancel_btn.grid(row=0, column=2, padx=5)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(row-2, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def select_video_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.video_folder.set(folder)
    
    def select_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder.set(folder)
    
    def log(self, message: str):
        """Thread-safe logging to GUI"""
        def _log():
            self.log_output.config(state='normal')
            self.log_output.insert(tk.END, message + "\n")
            self.log_output.yview(tk.END)
            self.log_output.config(state='disabled')
        self.root.after(0, _log)
    
    def update_progress(self, data):
        """Update progress bar"""
        def _update():
            if isinstance(data, tuple):
                cmd, value = data
                if cmd == 'max':
                    self.progress_bar['maximum'] = value
                elif cmd == 'progress':
                    self.progress_bar['value'] = value
                elif cmd == 'video':
                    self.progress_label['text'] = value
            else:
                self.progress_label['text'] = str(data)
        self.root.after(0, _update)
    
    def on_start(self):
        """Start processing"""
        self.update_config_from_gui()
        self.save_current_config()
        
        # Validate
        valid, error = self.config.validate()
        if not valid:
            messagebox.showerror("Configuration Error", error)
            return
        
        # Setup logging
        self.logger = setup_logging(self.config.output_folder, self.log)
        
        # Create processor
        self.processor = VideoProcessor(self.config, self.logger, self.state)
        
        # Update UI
        self.start_btn.config(state='disabled')
        self.pause_btn.config(state='normal')
        self.cancel_btn.config(state='normal')
        
        # Start processing in thread
        import threading
        threading.Thread(target=self.run_processing, daemon=True).start()
    
    def run_processing(self):
        """Run the processing pipeline"""
        try:
            self.state.start()
            
            # Initialize
            self.log("Initializing...")
            if not self.processor.initialize(self.update_progress):
                self.log("ERROR: Initialization failed")
                self.on_complete()
                return
            
            # Process videos
            self.log("Starting video processing...")
            self.processor.process_all_videos(self.update_progress)
            
            if self.state.is_cancelled():
                self.log("Processing cancelled by user")
            else:
                self.log("✅ Processing complete!")
            
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            self.log(f"ERROR: {e}")
        finally:
            self.on_complete()
    
    def on_pause(self):
        """Pause/Resume processing"""
        if self.state.is_paused:
            self.state.resume()
            self.pause_btn.config(text="⏸ Pause")
            self.log("Resumed")
        else:
            self.state.pause()
            self.pause_btn.config(text="▶ Resume")
            self.log("Paused")
    
    def on_cancel(self):
        """Cancel processing"""
        if messagebox.askyesno("Confirm", "Are you sure you want to cancel?"):
            self.state.cancel()
            self.log("Cancelling...")
    
    def on_complete(self):
        """Called when processing completes"""
        def _update():
            self.start_btn.config(state='normal')
            self.pause_btn.config(state='disabled')
            self.cancel_btn.config(state='disabled')
            self.pause_btn.config(text="⏸ Pause")
        self.root.after(0, _update)
    
    def on_closing(self):
        """Handle window close"""
        if self.state.is_running:
            if messagebox.askyesno("Confirm", "Processing is running. Are you sure you want to exit?"):
                self.state.cancel()
                self.root.destroy()
        else:
            self.root.destroy()


# ===== MAIN ENTRY POINT =====
if __name__ == "__main__":
    try:
        app = EnhancedGUI()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
