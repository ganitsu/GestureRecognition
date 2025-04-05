import argparse
import sys
import time
import cv2
import numpy as np
import mediapipe as mp
import threading
from flask import Flask, jsonify

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# === Flask App Setup ===
app = Flask(__name__)
shared_hand_data = []

@app.route('/hands', methods=['GET'])
def get_hand_data():
    return jsonify(shared_hand_data)

def start_flask_server():
    app.run(host="0.0.0.0", port=5000)

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

COUNTER, FPS = 0, 0
START_TIME = time.time()

def run(model: str, num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float, min_tracking_confidence: float,
        stream_url: str, frame_width: int, frame_height: int,
        proc_width: int, proc_height: int,
        processing_fps: float, disable_image: bool, silence: bool) -> None:

    global shared_hand_data

    cap = None
    if not disable_image:
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        if not cap.isOpened():
            sys.exit(f'ERROR: Unable to open video stream from {stream_url}')

    fps_avg_frame_count = 10
    last_processed_time = 0
    processing_interval = 1.0 / processing_fps
    last_result = None

    def save_result(result, unused_output_image: mp.Image, timestamp_ms: int):
        nonlocal last_result
        global FPS, COUNTER, START_TIME, shared_hand_data
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()
        last_result = result
        COUNTER += 1

        output = []
        for i, landmarks in enumerate(result.hand_landmarks):
            x_center = np.mean([lmk.x for lmk in landmarks])
            y_center = np.mean([lmk.y for lmk in landmarks])
            gesture_name = result.gestures[i][0].category_name if result.gestures else None
            gesture_score = result.gestures[i][0].score if result.gestures else None

            hand_data = {
                "id": i,
                "gesture": gesture_name,
                "score": round(gesture_score, 2) if gesture_score is not None else None,
                "center": {"x": round(x_center, 3), "y": round(y_center, 3)}
            }
            output.append(hand_data)

        shared_hand_data = output

        if disable_image and result.gestures and not silence:
            for gesture in result.gestures:
                label = gesture[0].category_name
                score = gesture[0].score
                print(f"[HAND] Detected: {label} ({score:.2f})")

    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        result_callback=save_result
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    while True:
        current_time = time.time()
        if current_time - last_processed_time >= processing_interval:
            last_processed_time = current_time
            if not silence:
                print("[INFO] Processing frame...")

            if disable_image:
                temp_cap = cv2.VideoCapture(stream_url)
                temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
                temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
                success, image = temp_cap.read()
                temp_cap.release()
                if not success:
                    print('WARNING: Unable to read from stream.')
                    continue
            else:
                success, image = cap.read()
                if not success:
                    print('WARNING: Unable to read from stream.')
                    break

            image = cv2.flip(image, 1)
            resized_image = cv2.resize(image, (proc_width, proc_height))
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)

        if not disable_image and last_result:
            success, image = cap.read()
            if not success:
                print('WARNING: Unable to read from stream.')
                break

            image = cv2.flip(image, 1)
            for hand_index, hand_landmarks in enumerate(last_result.hand_landmarks):
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lmk.x, y=lmk.y, z=lmk.z) for lmk in hand_landmarks
                ])
                mp_drawing.draw_landmarks(image, hand_landmarks_proto, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())

                if last_result.gestures:
                    gesture = last_result.gestures[hand_index]
                    label = gesture[0].category_name
                    score = round(gesture[0].score, 2)
                    center_x = int(np.mean([lmk.x for lmk in hand_landmarks]) * image.shape[1])
                    center_y = int(np.mean([lmk.y for lmk in hand_landmarks]) * image.shape[0])
                    text = f'{label} ({score})'
                    cv2.putText(image, text, (center_x, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Gesture Recognition", image)
            if cv2.waitKey(1) == 27:
                break
        else:
            time.sleep(0.01)

    recognizer.close()
    if cap:
        cap.release()
    if not disable_image:
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gesture_recognizer.task')
    parser.add_argument('--numHands', type=int, default=2)
    parser.add_argument('--minHandDetectionConfidence', type=float, default=0.5)
    parser.add_argument('--minHandPresenceConfidence', type=float, default=0.5)
    parser.add_argument('--minTrackingConfidence', type=float, default=0.5)
    parser.add_argument('--streamUrl', default="http://nitsuga:putoelquelee@192.168.0.127:30141/videostream.cgi")
    parser.add_argument('--frameWidth', type=int, default=128)
    parser.add_argument('--frameHeight', type=int, default=128)
    parser.add_argument('--procWidth', type=int, default=64)
    parser.add_argument('--procHeight', type=int, default=64)
    parser.add_argument('--processingFps', type=float, default=1.0)
    parser.add_argument('--disableImage', action='store_true', help='Disable image display and capture')
    parser.add_argument('--silence', action='store_true', help='Suppress output logs')
    args = parser.parse_args()

    flask_thread = threading.Thread(target=start_flask_server, daemon=True)
    flask_thread.start()

    run(args.model, args.numHands, args.minHandDetectionConfidence, args.minHandPresenceConfidence,
        args.minTrackingConfidence, args.streamUrl, args.frameWidth, args.frameHeight,
        args.procWidth, args.procHeight, args.processingFps, args.disableImage, args.silence)

if __name__ == '__main__':
    main()
