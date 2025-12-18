import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque

# ------------------ CONFIG ------------------

CAM_W, CAM_H = 640, 480
SIM_W, SIM_H = 640, 480
TOTAL_W, TOTAL_H = CAM_W + SIM_W, CAM_H

DRONE_SPEED = 0.05        # units per frame
WORLD_RADIUS = 5.0        # how far drone can move from origin

GESTURE_STABLE_FRAMES = 5 # how many frames to confirm a gesture

# ------------------ MEDIAPIPE SETUP ------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ------------------ DRONE STATE ------------------

drone_x = 0.0   # left/right
drone_z = 0.0   # forward/back (towards screen)
drone_state = "LANDED"

current_gesture = "NONE"
stable_gesture = "NONE"
gesture_history = deque(maxlen=GESTURE_STABLE_FRAMES)

# ------------------ UTIL FUNCTIONS ------------------

def fingers_up(hand):
    lm = hand.landmark
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    out = []
    for tip, pip in zip(tips, pips):
        out.append(lm[tip].y < lm[pip].y)
    return out  # [index, middle, ring, pinky]


def classify_gesture(hand):
    """Return a gesture string based on which fingers are up."""
    index, middle, ring, pinky = fingers_up(hand)
    up_count = sum([index, middle, ring, pinky])

    # ROCK: none -> LAND
    if up_count == 0:
        return "LAND"

    # PAPER: all -> HOVER
    if index and middle and ring and pinky:
        return "HOVER"

    # SCISSORS: index + middle -> MOVE_FORWARD
    if index and middle and not ring and not pinky:
        return "MOVE_FORWARD"

    # INDEX only -> MOVE_BACKWARD
    if index and not middle and not ring and not pinky:
        return "MOVE_BACKWARD"

    # INDEX + MIDDLE + RING -> MOVE_LEFT
    if index and middle and ring and not pinky:
        return "MOVE_LEFT"

    return "NONE"


def update_stable_gesture(new_gesture):
    """Use history to smooth noisy gesture detection."""
    global stable_gesture
    gesture_history.append(new_gesture)
    if len(gesture_history) == GESTURE_STABLE_FRAMES:
        unique = set(gesture_history)
        if len(unique) == 1 and list(unique)[0] != "NONE":
            stable_gesture = list(unique)[0]


def update_drone():
    """Move drone based on stable_gesture."""
    global drone_x, drone_z, drone_state

    if stable_gesture == "LAND":
        drone_state = "LANDED"
        drone_x *= 0.9
        drone_z *= 0.9
        return

    if stable_gesture == "HOVER":
        drone_state = "HOVERING"
        return

    drone_state = "FLYING"

    if stable_gesture == "MOVE_FORWARD":
        drone_z -= DRONE_SPEED
    elif stable_gesture == "MOVE_BACKWARD":
        drone_z += DRONE_SPEED
    elif stable_gesture == "MOVE_LEFT":
        drone_x -= DRONE_SPEED

    # clamp within world
    r = math.sqrt(drone_x**2 + drone_z**2)
    if r > WORLD_RADIUS:
        scale = WORLD_RADIUS / r
        drone_x *= scale
        drone_z *= scale


def draw_sim(sim):
    """Draw 3D-ish grid, blue quad-drone and green radar widget."""
    sim[:] = 0
    h, w, _ = sim.shape
    cx, ground_y = w // 2, int(h * 0.75)
    horizon_y = int(h * 0.25)

    # ---------- perspective ground grid ----------
    num_lines = 20
    for i in range(-num_lines, num_lines + 1):
        x_bottom = cx + i * 30
        x_top = cx + int(i * 10)
        cv2.line(sim, (x_bottom, ground_y), (x_top, horizon_y), (50, 50, 50), 1)

    for j in range(1, 10):
        t = j / 10.0
        y = int(horizon_y + t * (ground_y - horizon_y))
        x_span = int((1 - t) * (w * 0.45))
        cv2.line(sim, (cx - x_span, y), (cx + x_span, y), (40, 40, 40), 1)

    # ---------- map drone world position ----------
    span_x = w * 0.4
    sx = int(cx + (drone_x / WORLD_RADIUS) * span_x)

    z_norm = (drone_z + WORLD_RADIUS) / (2 * WORLD_RADIUS)
    sy = int(horizon_y + z_norm * (ground_y - horizon_y))

    # ---------- blue quad-drone ----------
    body_w, body_h = 40, 18
    body_color = (255, 0, 0)  # blue in BGR

    # body
    cv2.rectangle(sim,
                  (sx - body_w // 2, sy - body_h // 2),
                  (sx + body_w // 2, sy + body_h // 2),
                  body_color, -1)
    cv2.rectangle(sim,
                  (sx - body_w // 2, sy - body_h // 2),
                  (sx + body_w // 2, sy + body_h // 2),
                  (255, 255, 255), 2)

    # arms + rotors
    arm_len = 28
    rotor_r = 6
    arm_color = (255, 255, 255)

    # front, back, left, right arms
    arms = [
        ((sx, sy - body_h // 2), (sx, sy - body_h // 2 - arm_len)),  # front
        ((sx, sy + body_h // 2), (sx, sy + body_h // 2 + arm_len)),  # back
        ((sx - body_w // 2, sy), (sx - body_w // 2 - arm_len, sy)),  # left
        ((sx + body_w // 2, sy), (sx + body_w // 2 + arm_len, sy)),  # right
    ]
    rotors = []

    for (x0, y0), (x1, y1) in arms:
        cv2.line(sim, (x0, y0), (x1, y1), arm_color, 2)
        rotors.append((x1, y1))

    for (rx, ry) in rotors:
        cv2.circle(sim, (rx, ry), rotor_r, (255, 255, 255), 2)
        cv2.circle(sim, (rx, ry), rotor_r - 2, (255, 255, 255), 1)

    # ---------- green radar widget (top-right) ----------
    radar_cx, radar_cy = w - 100, 90
    radar_r = 60
    t = time.time()

    # outer circle
    cv2.circle(sim, (radar_cx, radar_cy), radar_r, (0, 255, 0), 2)

    # inner fading rings
    for i in range(1, 4):
        r = int(radar_r * (i / 4.0))
        cv2.circle(sim, (radar_cx, radar_cy), r, (0, 120, 0), 1)

    # spinning spokes
    num_spokes = 18
    for i in range(num_spokes):
        angle = (i / num_spokes) * 2 * math.pi + t * 0.8
        x_end = int(radar_cx + radar_r * math.cos(angle))
        y_end = int(radar_cy + radar_r * math.sin(angle))
        cv2.line(sim, (radar_cx, radar_cy), (x_end, y_end), (0, 200, 0), 1)

    # radial sweep line (brighter)
    sweep_angle = t * 1.5
    sx2 = int(radar_cx + radar_r * math.cos(sweep_angle))
    sy2 = int(radar_cy + radar_r * math.sin(sweep_angle))
    cv2.line(sim, (radar_cx, radar_cy), (sx2, sy2), (0, 255, 0), 2)

    # ---------- HUD text ----------
    cv2.putText(sim, "Gesture-Driven 3D Drone Simulator",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(sim, f"Gesture: {stable_gesture}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.putText(sim, f"Drone: {drone_state}",
                (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.putText(sim, f"Pos: x={drone_x:.2f}  z={drone_z:.2f}",
                (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.putText(sim, "Controls:",
                (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    cv2.putText(sim, "Rock = LAND, Open palm = HOVER",
                (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    cv2.putText(sim, "Scissors = FORWARD, Index = BACKWARD, 3 fingers = LEFT",
                (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    cv2.putText(sim, "Press 'q' to quit",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)


def main():
    global current_gesture

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (CAM_W, CAM_H))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        current_gesture = "NONE"

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            current_gesture = classify_gesture(hand)
            update_stable_gesture(current_gesture)

        update_drone()

        cv2.putText(frame, f"Detected: {current_gesture}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        sim = np.zeros((SIM_H, SIM_W, 3), dtype=np.uint8)
        draw_sim(sim)

        combined = np.zeros((TOTAL_H, TOTAL_W, 3), dtype=np.uint8)
        combined[:, :SIM_W] = sim
        combined[:, SIM_W:] = frame

        cv2.imshow("Gesture-Driven Drone Simulator", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 
