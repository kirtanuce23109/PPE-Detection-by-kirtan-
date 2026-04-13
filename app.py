import streamlit as st
import cv2
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

st.title("👷 PPE Detection Dashboard")

uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    model = YOLO("best.pt")

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    workers = {"helmet":0, "no_helmet":0, "vest":0, "no_vest":0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls == 1:
                    workers["helmet"] += 1
                elif cls == 2:
                    workers["vest"] += 1

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame)

    cap.release()

    df = pd.DataFrame([workers])

    st.subheader("📊 PPE Stats")
    st.dataframe(df)

    df.plot(kind="bar")
    st.pyplot(plt)
