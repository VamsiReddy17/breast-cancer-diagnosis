-- OncoSense Admin Visibility & Guaranteed Upload Storage Migration
-- Run this idempotent script in the Supabase SQL Editor.

-- 1. Ensure private biopsy-images Storage bucket exists
INSERT INTO storage.buckets (id, name, public)
VALUES ('biopsy-images', 'biopsy-images', false)
ON CONFLICT (id) DO NOTHING;

-- 2. Admin whitelist table (controls who can view ALL users' uploads, values, and logs in the app)
CREATE TABLE IF NOT EXISTS public.admin_users (
    email TEXT PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

ALTER TABLE public.admin_users ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can check own admin status" ON public.admin_users;
CREATE POLICY "Users can check own admin status"
    ON public.admin_users FOR SELECT
    TO authenticated
    USING (lower(email) = lower(COALESCE(auth.jwt()->>'email', '')));

-- Helper function used by RLS policies
CREATE OR REPLACE FUNCTION public.is_admin()
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1
        FROM public.admin_users
        WHERE lower(email) = lower(COALESCE(auth.jwt()->>'email', ''))
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;

-- 3. Update RLS on audit_logs & user_usage so admins can see ALL users' rows
DROP POLICY IF EXISTS "Users can view own audit logs" ON public.audit_logs;
CREATE POLICY "Users can view own audit logs"
    ON public.audit_logs FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id OR public.is_admin());

DROP POLICY IF EXISTS "Users can view own usage" ON public.user_usage;
CREATE POLICY "Users can view own usage"
    ON public.user_usage FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id OR public.is_admin());

-- 4. Update Storage policies so admins can view/sign ALL users' uploaded biopsy images
DROP POLICY IF EXISTS "Users can view own biopsy images" ON storage.objects;
CREATE POLICY "Users can view own biopsy images"
    ON storage.objects FOR SELECT
    TO authenticated
    USING (
        bucket_id = 'biopsy-images'
        AND (
            (storage.foldername(name))[1] = auth.uid()::text
            OR public.is_admin()
        )
    );

-- IMPORTANT: Replace with your Google sign-in email and run once to enable the Admin view in the app:
-- INSERT INTO public.admin_users (email) VALUES ('your-google-email@gmail.com') ON CONFLICT DO NOTHING;
