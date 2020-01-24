#!/usr/bin/env python3

import cv2
from pandas import DataFrame
from datetime import timedelta

# cv2 can detect it but it's easier this way.
FPS = 25
VIDEO_FILE_PATH = "video.mp4"
# In each second, number of detections considered to be noise
NOISE_PER_SECOND = 3
# detected area size less than this considered to be noise
NOISE_AREA_UPPER_SIZE = 1000

# distance considered to be same log
LOG_NEIGHBOR_DISTANCE = 3
LOG_INDEX_ENABLED = False

UPPER_LEFT_COORDINATES = (24, 53)
BOTTOM_RIGHT_COORDINATES = (967, 806)
X1, X2, Y1, Y2 = UPPER_LEFT_COORDINATES[0], BOTTOM_RIGHT_COORDINATES[0], \
    UPPER_LEFT_COORDINATES[1], BOTTOM_RIGHT_COORDINATES[1]


def detect_movement(frame1, frame2, detections):
    diff = cv2.absdiff(frame1[Y1:Y2, X1:X2], frame2[Y1:Y2, X1:X2])
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # to see region of interest, uncomment the next line
    # cv2.rectangle(frame1, (X1, Y1), (X2, Y2), (255, 0, 0), 2)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < NOISE_AREA_UPPER_SIZE:
            continue
        detections += 1

        # cv2.rectangle(frame1[Y1:Y2, X1:X2], (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.putText(frame1, "Status: {}".format('Movement'),
        #             (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    return detections


def log_movement(df, log_time):
    end_time = str(timedelta(seconds=log_time["end"]))
    if df["End"].iloc[-1] != end_time:
        df = df.append({'Start': str(
            timedelta(seconds=log_time["start"])), 'End': end_time},
            ignore_index=True)
        global new_log
        new_log = True
    return df


df = DataFrame(columns=["Start", "End"])
df = df.append({'Start': "placeholder", 'End': "-1 day, 23:59:59"},
               ignore_index=True)

cap = cv2.VideoCapture(VIDEO_FILE_PATH)
ret, frame1 = cap.read()
ret, frame2 = cap.read()
frame_num = 2
log_time = {"start": -1, "end": -1}
new_log = True  # global variable
detections = 0
log_second = int(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)

while(cap.isOpened()):
    if frame1 is None or frame2 is None:
        print("Empty frame occured at ", frame_num//FPS)
        break
    detections = detect_movement(frame1, frame2, detections)

    this_second = int(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
    if log_second < this_second:
        if detections > NOISE_PER_SECOND:
            if new_log:
                log_time["start"] = log_second
                new_log = False
            detections = 0
            log_time["end"] = log_second
        log_second = this_second
        print("Clock", this_second, end="\r")
    if log_time["end"] < this_second - LOG_NEIGHBOR_DISTANCE:
        df = log_movement(df, log_time)

    # cv2.imshow("feed", frame1)
    # if cv2.waitKey(20) & 0xFF == ord('q'):
    #     break

    frame1 = frame2
    ret, frame2 = cap.read()
    frame_num += 1

df = log_movement(df, log_time)  # to ensure the last log is written.
df = df.drop(df.index[0])
print("-----------End----------\n", df)
df.to_csv('Times.csv', index=LOG_INDEX_ENABLED, encoding='utf-8')
cap.release()
cv2.destroyAllWindows()
