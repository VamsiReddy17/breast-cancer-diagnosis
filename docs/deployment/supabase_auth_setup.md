# Supabase & Google Sign-In Step-by-Step Setup Guide

This guide describes how to configure Google OAuth login using Supabase for user authentication, protected routes, and audit tracking in OncoSense.

---

## 🛠️ Step 1: Set Up Your Supabase Project

1.  **Create an Account**: Go to [supabase.com](https://supabase.com/) and sign up or sign in.
2.  **Create a New Project**:
    *   Click **New Project**.
    *   Select your organization, give it a name (e.g., `OncoSense-Auth`), and set a secure database password.
    *   Choose a hosting region closest to your users.
    *   Click **Create New Project** and wait a couple of minutes for provisioning.
3.  **Collect API Credentials**:
    *   Once provisioned, go to **Settings** (gear icon) ➔ **API**.
    *   Copy the **Project URL** (e.g., `https://xxxxxx.supabase.co`).
    *   Copy the **anon public API Key**.
    *   Copy the **JWT Secret** (you will need this for the backend FastAPI verification).

---

## 🔑 Step 2: Configure Google Cloud OAuth Credentials

To allow users to log in with their Google accounts, you must register your app on Google Cloud:

1.  **Go to Google Developer Console**: Open [console.cloud.google.com](https://console.cloud.google.com/).
2.  **Create a Project**: Select the project dropdown in the top left and click **New Project** (e.g., `OncoSense`).
3.  **Configure OAuth Consent Screen**:
    *   Go to **APIs & Services** ➔ **OAuth consent screen**.
    *   Select **External** user type and click **Create**.
    *   Fill out the **App Information** (App Name, User Support Email, Developer Contact Email). Click **Save and Continue**.
    *   Under **Scopes**, click **Save and Continue** (default scopes are sufficient).
    *   Under **Test users**, add your own email address so you can test it before publishing. Click **Save and Continue**.
4.  **Create Credentials (OAuth Client ID)**:
    *   Go to the **Credentials** tab on the left.
    *   Click **+ Create Credentials** at the top ➔ **OAuth client ID**.
    *   Select **Web application** as the application type.
    *   Name it `OncoSense Client`.
5.  **Get the Redirect URI from Supabase**:
    *   Open your **Supabase Dashboard** in another tab.
    *   Go to **Authentication** ➔ **Providers** ➔ **Google**.
    *   Copy the URL displayed under **Callback URL (for OAuth)** (it will look like `https://xxxxxx.supabase.co/auth/v1/callback`).
6.  **Add Authorized Redirect URIs in Google**:
    *   Go back to the Google Developer Console where you are creating the credentials.
    *   Under **Authorized redirect URIs**, click **+ Add URI** and paste the callback URL you copied from Supabase.
    *   Click **Create**.
7.  **Save Client IDs**:
    *   Copy the generated **Client ID** and **Client Secret**.

---

## 🔌 Step 3: Enable Google Provider in Supabase

1.  Go to your **Supabase Dashboard** ➔ **Authentication** ➔ **Providers**.
2.  Expand the **Google** accordion.
3.  Toggle **Enable Google Provider** to active.
4.  Paste your **Client ID** and **Client Secret** copied from Google Cloud Console.
5.  Click **Save**.

---

## 💻 Step 4: Configure Local Project Environment Variables

Add these variables to local files (which are ignored by Git to prevent leaking secrets):

### Frontend Settings
Create a file named `frontend/.env.local`:
```env
VITE_SUPABASE_URL=https://your-project-id.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-public-key
```

### Backend Settings
Create a file named `.env` in your project root folder:
```env
SUPABASE_JWT_SECRET=your-supabase-jwt-secret-from-api-settings
```

---

## 🛠️ Step 5: Implementation Reference

### Frontend Client Authentication (React)
Install the Supabase JS SDK:
```bash
cd frontend
npm install @supabase/supabase-js
```

Initialize the client and trigger Google sign-in:
```javascript
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Trigger Google OAuth Flow
export async function signInWithGoogle() {
  const { error } = await supabase.auth.signInWithOAuth({
    provider: 'google',
  });
  if (error) console.error('Error logging in:', error.message);
}

// Log Out
export async function signOut() {
  await supabase.auth.signOut();
}
```

### Backend JWT Authorization Header Verification (FastAPI)
Verify incoming requests containing the authorization token in `src/api.py`:

```python
import jwt
import os
from fastapi import Header, HTTPException, Depends

JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET")

def verify_supabase_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token.")
    
    token = authorization.split(" ")[1]
    try:
        # Decode token using your Supabase JWT Secret
        payload = jwt.decode(
            token, 
            JWT_SECRET, 
            algorithms=["HS256"], 
            options={"verify_aud": False}
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid authorization token.")
```
