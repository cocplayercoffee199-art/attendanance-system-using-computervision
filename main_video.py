import cv2 # type: ignore
import argparse
from simple_facerec import SimpleFacerec # type: ignore
from datetime import datetime
from anti_spoofing_utils import anti_spoofing, anti_spoofing_score_on_frame

def silentface_check_on_frame(frame_bgr, bbox_xyxy, threshold: float = 0.8) -> tuple[bool, float]:
    """Return (is_live, score) using SilentFace on the original frame.

    bbox_xyxy: (x1, y1, x2, y2) in frame coordinates.
    threshold: probability threshold for the Real class.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return False, 0.0
    try:
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        score = anti_spoofing_score_on_frame(frame_bgr, [x1, y1, w, h])
        return bool(score >= threshold), float(score)
    except Exception:
        return False, 0.0

def markAttendance(names):
    if names.strip().lower() == "unknown":   # skip Unknown
        return
    
    with open(r"/Users/macbook/Developer/face-recognition/Attendance.csv",'r+') as f:
        MyDataList = f.readlines()
        nameList = []
        for line in MyDataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if names not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{names}, {dtString}')

def run_video(max_frames: int | None = None, headless: bool = False):
    # Encode faces from a folder
    sfr = SimpleFacerec()
    sfr.load_encoding_images(r"/Users/macbook/Developer/face-recognition/images")

    # Load Camera
    cap = cv2.VideoCapture(0)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            # Evaluate liveness on the full frame using this bbox
            is_live, live_score = silentface_check_on_frame(frame, (x1, y1, x2, y2))

            if is_live:
                cv2.putText(frame, f"{name} ({live_score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 4)
                names = str(name)
                markAttendance(names)
            else:
                cv2.putText(frame, f"Spoof ({live_score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

        if not headless:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=None, help="Run for N frames then exit")
    parser.add_argument("--headless", action="store_true", help="Do not show window")
    args = parser.parse_args()
    run_video(max_frames=args.frames, headless=args.headless)
