"""
gesture_recognizer.py
---------------------
Supports two modes:
  - "rules"  : original hardcoded geometry (your baseline for the paper)
  - "model"  : trained MLP loaded from gesture_model.pth

Set USE_MODEL = True after you have trained a model with train.py.
"""

import math
import json
import numpy as np

USE_MODEL  = True          # ← flip to True after training
MODEL_PATH = "gesture_model.pth"
LABEL_MAP_PATH = "label_map.json"

# ── Lazy-load torch only when model mode is requested ─────────────────────────
_model      = None
_label_map  = None
_inv_label  = None   # int → gesture string

def _load_model():
    global _model, _label_map, _inv_label
    import torch
    import torch.nn as nn

    class GestureMLP(nn.Module):
        def __init__(self, input_size=42, hidden_size=128, num_classes=5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, num_classes),
            )
        def forward(self, x):
            return self.net(x)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    _model = GestureMLP(
        checkpoint["input_size"],
        checkpoint["hidden_size"],
        checkpoint["num_classes"],
    )
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()
    _label_map = checkpoint["label_map"]          # {int: str}
    _inv_label = {int(k): v for k, v in _label_map.items()}
    print(f"[GestureRecognizer] ML model loaded from {MODEL_PATH}")
    return _model


def _normalize_landmarks(landmarks):
    """Mirror the normalization done in train.py."""
    coords = []
    for (x, y) in landmarks:
        coords += [x, y]
    arr = np.array(coords, dtype=np.float32)
    wrist_x, wrist_y = arr[0], arr[1]
    arr[0::2] -= wrist_x
    arr[1::2] -= wrist_y
    scale = np.max(np.abs(arr)) + 1e-6
    arr /= scale
    return arr


# ── Gesture positions (used by both modes) ────────────────────────────────────
CURSOR_GESTURES = {"MOVE", "CLICK", "SCROLL_UP", "SCROLL_DOWN"}


class GestureRecognizer:
    def __init__(self, click_threshold=30, use_model=None):
        self.click_threshold = click_threshold
        self.use_model = USE_MODEL if use_model is None else use_model

        if self.use_model:
            _load_model()

    def get_distance(self, point1, point2):
        return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

    # ── Rule-based (baseline) ─────────────────────────────────────────────────
    def _rules_recognize(self, landmarks):
        index_tip   = landmarks[8]
        index_mcp   = landmarks[5]
        thumb_tip   = landmarks[4]
        middle_tip  = landmarks[12]
        middle_mcp  = landmarks[9]
        pinky_tip   = landmarks[20]
        pinky_mcp   = landmarks[17]

        cursor_x = index_tip[0]
        cursor_y = index_tip[1]

        pinch_dist      = self.get_distance(thumb_tip, index_tip)
        index_extended  = index_tip[1]  < index_mcp[1]
        middle_extended = middle_tip[1] < middle_mcp[1]
        pinky_extended  = pinky_tip[1]  < pinky_mcp[1]

        if pinch_dist < self.click_threshold:
            return "CLICK", (cursor_x, cursor_y)
        elif pinky_extended and not index_extended and not middle_extended:
            return "SCROLL_DOWN", (cursor_x, cursor_y)
        elif index_extended and middle_extended:
            return "SCROLL_UP", (cursor_x, cursor_y)
        elif index_extended:
            return "MOVE", (cursor_x, cursor_y)
        else:
            return "NO_GESTURE", None

    # ── ML model ──────────────────────────────────────────────────────────────
    def _model_recognize(self, landmarks):
        import torch
        features = _normalize_landmarks(landmarks)
        tensor   = torch.tensor(features).unsqueeze(0)   # (1, 42)
        with torch.no_grad():
            logits  = _model(tensor)
            pred_idx = int(logits.argmax(1).item())
        gesture = _inv_label[pred_idx]
        # Cursor position is always the index fingertip
        cursor_x, cursor_y = landmarks[8]
        pos = (cursor_x, cursor_y) if gesture in CURSOR_GESTURES else None
        return gesture, pos

    # ── Public API ────────────────────────────────────────────────────────────
    def recognize_gesture(self, landmarks, config):
        if not landmarks or len(landmarks) < 21:
            return "NO_HAND", None

        if self.use_model:
            return self._model_recognize(landmarks)
        else:
            return self._rules_recognize(landmarks)