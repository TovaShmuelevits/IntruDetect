import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Lucas-Kanade Optical Flow Tracker')
parser.add_argument('video', type=str, help='path to video file')
args = parser.parse_args()

cap = cv.VideoCapture(args.video)

feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=5, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                  criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (500, 3))

ret, old_frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask = np.zeros_like(old_frame)

older_points = None  # אחסון את נקודות המעקב מהפריים הקודם
frame_count = 0      # מעקב אחרי מספר הפריים הנוכחי שנקרא
prev_avg_speed = 0  # שומר את המהירות הממוצעת של הפрейм קודם
fast_threshold = 9   # סף מהירות - נסה להגדיל אם מקבלים זיהויים שגויים
sharp_change_threshold = 30.0 # סף להבדל בין המהירות הממוצעת - נסה להגדיל אם מקבלים זיהויים שגויים
min_speed_for_turn_detection = 5  # סף למהירות המינימלית לזיהוי סיבוב חד - נסה להגדיל אם מקבלים זיהויים שגויים

while True:
    ret, frame = cap.read()
    frame_count += 1
    if not ret:
        print("No more frames!")
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None and st is not None and np.count_nonzero(st) > 0:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        motion_vectors = good_new - good_old
        speeds = np.linalg.norm(motion_vectors, axis=1)
        avg_speed = np.mean(speeds)

        # תנאים להדפסה של שינוי חד במהירות
        if avg_speed > fast_threshold:
            print(f"Tracking {len(good_new)} points in frame {frame_count}")
            print(f":תעצוממ תוריהמ {avg_speed:.2f}")
            print("!הריהמ העונת")
        elif abs(avg_speed - prev_avg_speed) > sharp_change_threshold:
            print(f"Tracking {len(good_new)} points in frame {frame_count}")
            print(f":תעצוממ תוריהמ {avg_speed:.2f}")
            print("!הדח העונת התהוז (שינוי מהירות)")
            print(f":הדח העונת תוריהמ {abs(avg_speed - prev_avg_speed):.2f}")

        # גילוי סיבובים חדים
        if older_points is not None and len(older_points) == len(good_old):
            vec1 = good_old - older_points
            vec2 = good_new - good_old
            dot = np.sum(vec1 * vec2, axis=1)
            norms1 = np.linalg.norm(vec1, axis=1)
            norms2 = np.linalg.norm(vec2, axis=1)
            norms = norms1 * norms2

            # מניעת חלוקה באפס
            valid = norms > 1e-6
            cos_angles = np.zeros_like(dot)
            cos_angles[valid] = np.clip(dot[valid] / norms[valid], -1.0, 1.0)

            angles = np.arccos(cos_angles)
            sharp_turns = angles > np.deg2rad(45) # נסה להגדיל את הזווית אם מקבלים זיהויים שגויים

            if np.any(sharp_turns) and avg_speed > min_speed_for_turn_detection:
                print(f"Tracking {len(good_new)} points in frame {frame_count}")
                print(f":תעצוממ תוריהמ {avg_speed:.2f}")
                print("!הדח העונת התהוז (סיבוב)")
                print(f"(בדיקת סיבוב) זוויות (במעלות): {np.degrees(angles[sharp_turns])}")
                print(f"(בדיקת סיבוב) מהירות ממוצעת: {avg_speed:.2f}, סף מהירות לסיבוב: {min_speed_for_turn_detection}")

        # ויזואליזציה: ציור תנועה
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i % len(color)].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i % len(color)].tolist(), -1)

        img = cv.add(frame, mask)
        cv.imshow('frame', img)

        older_points = good_old.copy()
        p0 = good_new.reshape(-1, 1, 2)
        old_gray = frame_gray.copy()
        prev_avg_speed = avg_speed

    else:
        print(f"No valid optical flow points found in frame {frame_count}")
        img = frame.copy()
        cv.imshow('frame', img)

    if cv.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
#python optical_flow.py "C:\Users\moiam\Documents\פרויקט\rr.avi"