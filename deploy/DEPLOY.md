# OncoSense Deployment Guide (Vercel + Render)

Follow these step-by-step instructions to host the **FastAPI Backend** on Render (free tier) and the **React Frontend** on Vercel (free tier).

---

## 🛠️ Step 1: Deploy the FastAPI Backend to Render

Render is a cloud hosting platform that makes it simple to host Python servers.

1. **Sign Up/Log In**: Go to [Render](https://render.com/) and register using your GitHub account.
2. **Create Web Service**:
   - In the Render Dashboard, click **New +** and select **Web Service**.
   - Connect your GitHub account (if not already connected) and select the `breast-cancer-diagnosis` repository.
3. **Configure Service Settings**:
   - **Name**: `oncosense-backend` (or a name of your choice)
   - **Region**: Select the closest region to your users (e.g., `Oregon (US West)`)
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: 
     ```bash
     pip install -r requirements.txt && python src/pipeline.py
     ```
     > [!NOTE]
     > Running `python src/pipeline.py` during the build phase generates the raw dataset features and trained `.joblib` model/scaler files, ensuring they are present when the server boots.
   - **Start Command**: 
     ```bash
     python -m uvicorn src.api:app --host 0.0.0.0 --port $PORT
     ```
   - **Plan**: Select **Free** (or Starter).
4. **Deploy**: Click **Deploy Web Service**.
   - Render will start building the project and training the ML models.
   - Once completed, you will see a live URL in the top left corner (e.g., `https://oncosense-backend.onrender.com`). Copy this URL.

---

## 🎨 Step 2: Deploy the React Frontend to Vercel

Vercel provides blazing-fast free hosting for frontend applications built with Vite/React.

1. **Sign Up/Log In**: Go to [Vercel](https://vercel.com/) and log in using your GitHub account.
2. **Import Project**:
   - Click **Add New > Project**.
   - Locate and select the `breast-cancer-diagnosis` repository, then click **Import**.
3. **Configure Build Settings**:
   - **Framework Preset**: `Vite` (Vercel auto-detects this)
   - **Root Directory**: Click *Edit* and select the **`frontend`** directory.
4. **Configure Environment Variables**:
   - Expand the **Environment Variables** accordion.
   - Add the following environment variable to link your React frontend to your live Render backend:
     - **Key**: `VITE_API_BASE_URL`
     - **Value**: `https://your-render-backend-url.onrender.com` (paste the URL you copied in Step 1, without a trailing slash)
5. **Deploy**: Click **Deploy**.
   - Vercel will install dependencies, compile the production bundles, and output a live public URL (e.g., `https://oncosense.vercel.app`).

---

## ⚙️ How It Connects Under the Hood
We set up a dynamic environment check in `frontend/src/App.jsx`:
```javascript
const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
```
- **Local Dev Server**: Defaults to `http://localhost:8000` because no environment variable is present.
- **Production Server**: Reads `VITE_API_BASE_URL` on build time to route all API calls to your live Render service.
