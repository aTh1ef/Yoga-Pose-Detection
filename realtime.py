import cv2
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
    normalized_image = resized_image / 255.0
    if len(normalized_image.shape) == 2:
        normalized_image = np.expand_dims(normalized_image, axis=-1)
    if normalized_image.shape[-1] == 1:
        normalized_image = np.repeat(normalized_image, 3, axis=-1)
    input_data = np.expand_dims(normalized_image, axis=0)
    return input_data

cap = cv2.VideoCapture(0)
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

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = [[lm.x * image.shape[1], lm.y * image.shape[0]] for lm in results.pose_landmarks.landmark]
        
        heatmap = create_heatmap(image.shape[:2], landmarks)

        input_image = preprocess_input_image(image)

        predictions = model.predict(input_image)

        pred = np.argmax(predictions)
        print(f"Prediction: {pred}")

        if 0 <= pred < len(yoga_poses):
            print("The predicted Yoga Pose is:", yoga_poses[pred])
        else:
            print("Unknown pose index:", pred)

    cv2.imshow('Yoga Pose Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()