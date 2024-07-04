import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import mediapipe as mp

model = tf.keras.models.load_model('new_model.h5')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def create_heatmap(image_shape, landmarks, radius=5):
    heatmap = np.zeros((image_shape[0], image_shape[1]), dtype=np.float32)
    for landmark in landmarks:
        if landmark is not None:
            x, y = int(landmark[0]), int(landmark[1])
            heatmap = cv2.circle(heatmap, (x, y), radius, 1, thickness=-1)
    return heatmap

def preprocess_input_image(image):
    resized_image = cv2.resize(image, (256, 256))
    if resized_image.shape[-1] == 4:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2RGB)
    else:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    normalized_image = resized_image / 255.0
    if len(normalized_image.shape) == 2:
        normalized_image = np.expand_dims(normalized_image, axis=-1)
    if normalized_image.shape[-1] == 1:
        normalized_image = np.repeat(normalized_image, 3, axis=-1)
    input_data = np.expand_dims(normalized_image, axis=0)
    return input_data


def detect_yoga_pose(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = [[lm.x * image.shape[1], lm.y * image.shape[0]] for lm in results.pose_landmarks.landmark]
        
        heatmap = create_heatmap(image.shape[:2], landmarks)

        input_image = preprocess_input_image(image)

        predictions = model.predict(input_image)

        pred = np.argmax(predictions)
        if 0 <= pred < len(yoga_poses):
            return yoga_poses[pred]
        else:
            return "Unknown pose index"
        
def capture_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '): 
            cv2.imwrite('captured_image.jpg', frame)
            break
    cap.release()
    cv2.destroyAllWindows()
    return 'captured_image.jpg'

yoga_poses = [
    'adho mukha svanasana',
    'adho mukha virasana',
    'agnistambhasana',
    'ananda balasana',
    'anantasana',
    'anjaneyasana',
    'ardha bhekasana',
    'ardha chandrasana',
    'ardha matsyendrasana',
    'ardha pincha mayurasana',
    'ardha uttanasana',
    'ashtanga namaskara',
    'astavakrasana',
    'baddha konasana',
    'bakasana',
]

def main():
    st.title('Yoga Pose Detection App')

    page = st.sidebar.radio("Navigation", ('Upload Image', 'Webcam Capture'))

    if page == 'Upload Image':
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            if st.button('Detect Pose'):
                image_cv = np.array(image)
                pose_result = detect_yoga_pose(image_cv)
                st.write("Detected Pose:", pose_result)

    elif page == 'Webcam Capture':
        st.write('Press Spacebar to capture image from webcam')
        if st.button('Capture Image'):
            captured_image_path = capture_image()
            captured_image = Image.open(captured_image_path)
            st.image(captured_image, caption='Captured Image', use_column_width=True)
            pose_result = detect_yoga_pose(cv2.imread(captured_image_path))
            st.write("Detected Pose:", pose_result)

if __name__ == "__main__":
    main()