"""
data_collector.py
-----------------
Collects hand landmark data for gesture classification training.

HOW TO USE:
  1. Run: python data_collector.py
  2. Press a number key to select the gesture you want to record:
       1 = MOVE
       2 = CLICK
       3 = SCROLL_UP
       4 = SCROLL_DOWN
       5 = NO_GESTURE
  3. Hold the gesture in front of the camera. Samples collect automatically.
  4. Press SPACE to pause/resume collection.
  5. Press 'q' to quit and save the CSV.

OUTPUT:
  gesture_data.csv  — one row per frame, columns: x0,y0,x1,y1,...,x20,y20,label
"""

import cv2
import csv
import os
import time
from hand_tracker import HandTracker
from config_loader import load_config

# ── Settings ──────────────────────────────────────────────────────────────────
OUTPUT_CSV   = "gesture_data.csv"
SAMPLE_DELAY = 0.05   # seconds between saved samples (≈20 fps max capture rate)

GESTURE_MAP = {
    ord('1'): "MOVE",
    ord('2'): "CLICK",
    ord('3'): "SCROLL_UP",
    ord('4'): "SCROLL_DOWN",
    ord('5'): "NO_GESTURE",
}

GESTURE_COLORS = {
    "MOVE":        (0,   255, 0),
    "CLICK":       (0,   0,   255),
    "SCROLL_UP":   (255, 200, 0),
    "SCROLL_DOWN": (0,   200, 255),
    "NO_GESTURE":  (150, 150, 150),
    None:          (80,  80,  80),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def landmarks_to_row(landmarks):
    """Flatten 21 (x, y) landmark tuples into a single list of 42 floats."""
    row = []
    for (x, y) in landmarks:
        row.append(x)
        row.append(y)
    return row  # length 42


def load_existing_counts(path):
    """Count how many samples per label already exist in the CSV."""
    counts = {}
    if not os.path.exists(path):
        return counts
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                label = row[-1]
                counts[label] = counts.get(label, 0) + 1
    return counts


def draw_ui(frame, current_gesture, collecting, counts, last_saved_label):
    h, w = frame.shape[:2]
    color = GESTURE_COLORS.get(current_gesture, (80, 80, 80))

    # Semi-transparent dark overlay at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Current gesture label
    status = "● REC" if collecting else "❚❚ PAUSED"
    status_color = (0, 60, 255) if collecting else (200, 200, 0)
    cv2.putText(frame, status, (w - 140, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    label_text = current_gesture if current_gesture else "--- SELECT GESTURE ---"
    cv2.putText(frame, label_text, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Key guide
    guide = "1:MOVE  2:CLICK  3:SCROLL_UP  4:SCROLL_DOWN  5:NO_GESTURE  SPACE:pause  Q:quit"
    cv2.putText(frame, guide, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # Sample counts
    count_str = "  |  ".join(
        f"{g}: {counts.get(g, 0)}" for g in
        ["MOVE", "CLICK", "SCROLL_UP", "SCROLL_DOWN", "NO_GESTURE"]
    )
    cv2.putText(frame, count_str, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 220, 140), 1)

    # Flash confirmation when a sample is saved
    if last_saved_label:
        cv2.putText(frame, f"Saved: {last_saved_label}", (10, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)

    return frame


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config = load_config()
    hand_tracker = HandTracker()

    cap = cv2.VideoCapture(config['camera']['device_id'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config['camera']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])

    # Prepare CSV
    write_header = not os.path.exists(OUTPUT_CSV)
    csv_file = open(OUTPUT_CSV, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}"]
        header.append("label")
        writer.writerow(header)

    counts = load_existing_counts(OUTPUT_CSV)

    current_gesture = None
    collecting = False
    last_sample_time = 0
    last_saved_label = None
    last_saved_flash = 0

    print("\n=== Gesture Data Collector ===")
    print("Press 1-5 to select a gesture, SPACE to toggle recording, Q to quit.\n")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = hand_tracker.find_hands(frame, draw=True)
        landmarks = hand_tracker.get_landmarks(frame)

        now = time.time()

        # Auto-collect if recording and hand visible
        if collecting and current_gesture and landmarks:
            if now - last_sample_time >= SAMPLE_DELAY:
                row = landmarks_to_row(landmarks) + [current_gesture]
                writer.writerow(row)
                csv_file.flush()
                counts[current_gesture] = counts.get(current_gesture, 0) + 1
                last_sample_time = now
                last_saved_label = current_gesture
                last_saved_flash = now

        # Clear flash after 0.4 s
        if now - last_saved_flash > 0.4:
            last_saved_label = None

        frame = draw_ui(frame, current_gesture, collecting, counts, last_saved_label)
        cv2.imshow("Gesture Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if current_gesture:
                collecting = not collecting
                print(f"{'Recording' if collecting else 'Paused'} — {current_gesture}")
            else:
                print("Select a gesture first (press 1-5).")
        elif key in GESTURE_MAP:
            current_gesture = GESTURE_MAP[key]
            collecting = True
            print(f"Now recording: {current_gesture}  (samples so far: {counts.get(current_gesture, 0)})")

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\n=== Collection complete ===")
    for g in ["MOVE", "CLICK", "SCROLL_UP", "SCROLL_DOWN", "NO_GESTURE"]:
        print(f"  {g:15s}: {counts.get(g, 0)} samples")
    print(f"\nSaved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()