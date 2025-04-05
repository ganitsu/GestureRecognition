# Importaciones igual que antes
import argparse
import sys
import time
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

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
        processing_fps: float, disable_image: bool) -> None:

    cap = None
    if not disable_image:
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        if not cap.isOpened():
            sys.exit(f'ERROR: Unable to open video stream from {stream_url}')

    row_size = 50
    left_margin = 24
    text_color = (0, 0, 0)
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    last_processed_time = 0
    processing_interval = 1.0 / processing_fps
    last_result = None

    def save_result(result, unused_output_image: mp.Image, timestamp_ms: int):
        nonlocal last_result
        global FPS, COUNTER, START_TIME
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()
        last_result = result
        COUNTER += 1

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

            if not disable_image:
                success, image = cap.read()
                if not success:
                    print('WARNING: Unable to read from stream.')
                    break

                image = cv2.flip(image, 1)
                resized_image = cv2.resize(image, (proc_width, proc_height))
                rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            else:
                # Create a blank image just to trigger processing
                rgb_image = (255 * np.ones((proc_height, proc_width, 3), dtype=np.uint8))

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)

        if not disable_image:
            success, image = cap.read()
            if not success:
                print('WARNING: Unable to read from stream.')
                break

            image = cv2.flip(image, 1)

            fps_text = 'FPS = {:.1f}'.format(FPS)
            text_location = (left_margin, row_size)
            cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness, cv2.LINE_AA)

            if last_result:
                for hand_index, hand_landmarks in enumerate(last_result.hand_landmarks):
                    x_min = min([landmark.x for landmark in hand_landmarks])
                    y_min = min([landmark.y for landmark in hand_landmarks])
                    y_max = max([landmark.y for landmark in hand_landmarks])
                    frame_height_disp, frame_width_disp = image.shape[:2]
                    x_min_px = int(x_min * frame_width_disp)
                    y_min_px = int(y_min * frame_height_disp)
                    y_max_px = int(y_max * frame_height_disp)

                    if last_result.gestures:
                        gesture = last_result.gestures[hand_index]
                        category_name = gesture[0].category_name
                        score = round(gesture[0].score, 2)
                        result_text = f'{category_name} ({score})'
                        text_x, text_y = x_min_px, y_min_px - 10
                        if text_y < 0:
                            text_y = y_max_px + 20
                        cv2.putText(image, result_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                    ])
                    mp_drawing.draw_landmarks(image, hand_landmarks_proto, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing_styles.get_default_hand_landmarks_style(),
                                              mp_drawing_styles.get_default_hand_connections_style())

            cv2.imshow('gesture_recognition', image)
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
    parser.add_argument('--procWidth', type=int, default=128)
    parser.add_argument('--procHeight', type=int, default=128)
    parser.add_argument('--processingFps', type=float, default=1.0)
    parser.add_argument('--disableImage', action='store_true', help='Disable image display and capture')
    args = parser.parse_args()

    run(args.model, args.numHands, args.minHandDetectionConfidence, args.minHandPresenceConfidence,
        args.minTrackingConfidence, args.streamUrl, args.frameWidth, args.frameHeight,
        args.procWidth, args.procHeight, args.processingFps, args.disableImage)

if __name__ == '__main__':
    main()
