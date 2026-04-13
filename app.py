import streamlit as st
import cv2
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

st.title("👷 PPE Detection Dashboard")

uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    model = YOLO("best.pt")
    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    workers = {}
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        detections = []
        helmets = []
        vests = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls == 0:
                    detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))
                elif cls == 1:
                    helmets.append((x1,y1,x2,y2))
                elif cls == 2:
                    vests.append((x1,y1,x2,y2))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, w, h = map(int, track.to_ltrb())

            px1, py1, px2, py2 = l, t, l+w, t+h

            if track_id not in workers:
                workers[track_id] = {
                    "helmet_time": 0,
                    "no_helmet_time": 0,
                    "vest_time": 0,
                    "no_vest_time": 0
                }

            has_helmet = any(hx1>px1 and hx2<px2 for hx1,_,hx2,_ in helmets)
            has_vest = any(vx1>px1 and vx2<px2 for vx1,_,vx2,_ in vests)

            time_increment = 1 / fps

            if has_helmet:
                workers[track_id]["helmet_time"] += time_increment
            else:
                workers[track_id]["no_helmet_time"] += time_increment

            if has_vest:
                workers[track_id]["vest_time"] += time_increment
            else:
                workers[track_id]["no_vest_time"] += time_increment

            cv2.rectangle(frame, (px1,py1), (px2,py2), (255,0,0), 2)
            cv2.putText(frame, f"ID: {track_id}", (px1,py1-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            if has_helmet:
                cv2.putText(frame, "Helmet", (px1,py1-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                cv2.putText(frame, "No Helmet", (px1,py1-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            if has_vest:
                cv2.putText(frame, "Vest", (px1,py1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame)

    cap.release()

    df = pd.DataFrame(workers).T

    st.subheader("📊 Worker PPE Time (seconds)")
    st.dataframe(df.round(2))

    st.subheader("📈 PPE Compliance Graph")
    df.plot(kind="bar")
    st.pyplot(plt)
