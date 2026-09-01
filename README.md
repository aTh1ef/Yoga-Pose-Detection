# Yoga Pose Detection

Classifies yoga poses from images and from a live webcam feed, using a CNN trained on pose imagery.

## What's in it

| File | Role |
|---|---|
| `yoga_cnn2.ipynb` | Notebook that builds and trains the CNN — data loading, augmentation, training, evaluation |
| `app.py` | Streamlit app for classifying an uploaded image |
| `realtime.py` | Live webcam classification |
| `mnb.py` | Multinomial Naive Bayes baseline to compare the CNN against |

Keeping a simple baseline beside the CNN makes the improvement measurable rather than assumed.

## Tech stack

| Concern | Choice |
|---|---|
| Deep learning | TensorFlow / Keras |
| Video | `av`, OpenCV for webcam capture |
| UI | Streamlit |
| Classical ML | scikit-learn (Naive Bayes baseline) |
| Charts | Altair |

## Running it

Prerequisites: Python 3.9+. A GPU is not required for inference.

```bash
pip install -r requirements.txt
```

Classify an uploaded image:

```bash
streamlit run app.py
```

Live webcam classification:

```bash
python realtime.py
```

Train from scratch — open the notebook:

```bash
jupyter notebook yoga_cnn2.ipynb
```

## Known gaps

- `requirements.txt` is a full pinned freeze of the development environment, so it installs far more than the app needs and may conflict on other Python versions.
- The trained model weights are not committed, so `app.py` and `realtime.py` need the notebook run first.
- Accuracy varies with lighting and camera angle; there is no pose-landmark preprocessing step.

## Related

[`Yoga-Pose-Live-Detection`](https://github.com/aTh1ef/Yoga-Pose-Live-Detection) is the standalone real-time version.
