# Production Deployment & Real-Time Inference

This document outlines the deployment architecture of OncoSense and explains how pre-trained deep learning weights are managed to enable live, real-time image classification.

---

## 🏗️ 1. Deployment Architecture

OncoSense is structured as a decoupled, dual-service web application:

```
┌─────────────────────────────────┐
│     Vercel Static Hosting       │
│  (React Frontend Client Portal) │
└────────────────┬────────────────┘
                 │
                 │ HTTP POST Requests
                 ▼
┌─────────────────────────────────┐
│       Render Web Service        │
│  (FastAPI Backend Python API)   │
└─────────────────────────────────┘
```

*   **Frontend (Vercel)**: Hosts the static React application. The environment variable `VITE_API_BASE_URL` is set to point to the live Render backend.
*   **Backend (Render)**: Runs the FastAPI server (`uvicorn`). The server listens for tabular diagnosis request payloads and histopathology image file uploads.

---

## 📦 2. Production Model Weights Management

Deep learning weights (`best_model.pth`) are binary files of about 17MB. To keep the Git repository lightweight, these weights are ignored by Git.

### The Self-Healing Download Flow:
To make the live version work on Render without committing large files to GitHub, the API incorporates an automatic startup downloader:

```
          [ Render Server Starts Up ]
                      │
                      ▼
        [ Checks for best_model.pth ]
         /                         \
    (Exists)                    (Missing)
      /                               \
[Ready to Serve]          [Checks DL_MODEL_WEIGHTS_URL]
                           /                         \
                       (Set)                      (Not Set)
                        /                               \
          [Downloads file from URL]           [Disables Image Tab]
                        │
                        ▼
             [Ready to Serve Requests]
```

1.  **Configure URL**: The developer uploads `best_model.pth` to a public storage folder (such as GitHub Releases) and configures the direct URL in Render under `DL_MODEL_WEIGHTS_URL`.
2.  **Download on Startup**: When Render starts up, `api.py` checks if `models/deep_learning/best_model.pth` exists. If missing, it uses `urllib.request.urlretrieve` to download the weights from `DL_MODEL_WEIGHTS_URL` and saves it locally.
3.  **Graceful Fallback**: If the URL is not set, the server still launches and classical ML tabs work, but image prediction requests return a clean warning.

---

## ⏱️ 3. Real-Time Inference Lifecycle

When a user uploads a biopsy image on Vercel and clicks **Analyze**:
1.  **File Upload**: The image is sent as a `multipart/form-data` payload via a `POST /api/predict/image` request.
2.  **Memory Load**: The backend reads the image bytes into memory and opens it using the Python Imaging Library (`PIL`).
3.  **Instantiation**: It instantiates an `EfficientNet-B0` architecture on the CPU (or GPU/MPS if available) and loads the downloaded `best_model.pth` state dict.
4.  **Forward Pass**: The normalized image tensor is passed through the network, producing logits for the Benign and Malignant classes.
5.  **Softmax Probabilities**: Softmax is applied to transform logits into confidence percentages:
    ```json
    {
        "model_used": "EfficientNet-B0 CNN",
        "prediction": 1,
        "class_label": "Malignant",
        "confidence": 96.44,
        "probabilities": {
            "Benign": 3.56,
            "Malignant": 96.44
        }
    }
    ```
6.  **UI Update**: Vercel receives the JSON response in under 1 second, updating the gauges dynamically for the user.
