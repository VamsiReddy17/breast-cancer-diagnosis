import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

let supabaseInstance = null;

// Only initialize Supabase if credentials are valid URLs/keys
if (supabaseUrl && supabaseAnonKey && supabaseUrl.startsWith('https://')) {
  try {
    supabaseInstance = createClient(supabaseUrl, supabaseAnonKey);
  } catch (error) {
    console.error("Failed to initialize Supabase client:", error);
  }
}

// Graceful stub fallback to prevent UI white-screens when credentials are not yet set up
if (!supabaseInstance) {
  console.warn("Supabase credentials missing or invalid. Initializing fallback mock client.");
  supabaseInstance = {
    auth: {
      getSession: async () => ({ data: { session: null }, error: null }),
      onAuthStateChange: () => ({ data: { subscription: { unsubscribe: () => {} } } }),
      signInWithOAuth: async () => {
        alert("Authentication is currently unavailable (Supabase configuration missing). Please configure VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY in your Vercel/local env.");
        return { error: new Error("Supabase environment variables missing.") };
      },
      signOut: async () => {},
    },
    from: () => ({
      insert: async () => ({ error: new Error("Supabase not configured.") })
    })
  };
}

export const supabase = supabaseInstance;

