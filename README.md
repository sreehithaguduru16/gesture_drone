# Gesture-Driven Drone Simulator 🚁✋

A real-time **gesture-controlled drone simulator** built using **OpenCV** and **MediaPipe**.  
Hand gestures captured via webcam are used to control a simulated quadcopter in a 3D-style environment.

This project demonstrates concepts of **computer vision**, **gesture recognition**, and **human–computer interaction**.

---

## ✨ Features

- Real-time hand tracking using MediaPipe
- Stable gesture detection using frame smoothing
- 3D-style drone simulation with perspective grid
- Radar-style HUD for drone awareness
- Gesture-based drone state control (Land, Hover, Fly)
- Side-by-side webcam feed and simulation view

---

## 🛠️ Technologies Used

- Python 3.x
- OpenCV
- MediaPipe
- NumPy

---

## 🎮 Gesture Controls

| Gesture | Action |
|------|------|
| ✊ Fist (Rock) | Land drone |
| ✋ Open Palm (Paper) | Hover |
| ✌️ Index + Middle (Scissors) | Move Forward |
| ☝️ Index finger only | Move Backward |
| ☝️☝️☝️ Three fingers | Move Left |

> Gestures are stabilized over multiple frames to reduce noise.

---

## 📷 Output View

- **Left side**: 3D drone simulation with grid and radar
- **Right side**: Live webcam feed with hand landmarks
- HUD displays current gesture, drone state, and position

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install opencv-python mediapipe numpy
