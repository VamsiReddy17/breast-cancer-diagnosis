-- OncoSense Full User Profile, Detector Audit, Feature Values & Image Tracking Migration
-- Run this idempotent script in the Supabase SQL Editor (after 01_schema.sql).

-- 1. Extend user_usage with profile & sign-in tracking columns
ALTER TABLE public.user_usage
    ADD COLUMN IF NOT EXISTS full_name TEXT,
    ADD COLUMN IF NOT EXISTS avatar_url TEXT,
    ADD COLUMN IF NOT EXISTS last_sign_in_at TIMESTAMP WITH TIME ZONE,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL;

ALTER TABLE public.user_usage
    ALTER COLUMN last_prediction_at DROP NOT NULL;

-- 2. Extend audit_logs to capture detector/model, confidence, posted values, and uploaded image info
ALTER TABLE public.audit_logs
    ADD COLUMN IF NOT EXISTS model_used TEXT,
    ADD COLUMN IF NOT EXISTS confidence NUMERIC(5,2),
    ADD COLUMN IF NOT EXISTS input_values JSONB,
    ADD COLUMN IF NOT EXISTS prediction_details JSONB,
    ADD COLUMN IF NOT EXISTS image_path TEXT,
    ADD COLUMN IF NOT EXISTS image_url TEXT;

CREATE INDEX IF NOT EXISTS idx_audit_logs_user_created
    ON public.audit_logs (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_logs_prediction_type
    ON public.audit_logs (prediction_type);

-- 3. Sync auth.users -> public.user_usage on signup and every login
CREATE OR REPLACE FUNCTION public.handle_auth_user_sync()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_usage (
        user_id,
        user_email,
        full_name,
        avatar_url,
        last_sign_in_at,
        total_predictions,
        image_predictions,
        last_prediction_at,
        created_at
    )
    VALUES (
        NEW.id,
        COALESCE(NEW.email, ''),
        COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.raw_user_meta_data->>'name', ''),
        NEW.raw_user_meta_data->>'avatar_url',
        COALESCE(NEW.last_sign_in_at, timezone('utc'::text, now())),
        0,
        0,
        NULL,
        COALESCE(NEW.created_at, timezone('utc'::text, now()))
    )
    ON CONFLICT (user_id) DO UPDATE
    SET
        user_email = COALESCE(EXCLUDED.user_email, public.user_usage.user_email),
        full_name = COALESCE(NULLIF(EXCLUDED.full_name, ''), public.user_usage.full_name),
        avatar_url = COALESCE(EXCLUDED.avatar_url, public.user_usage.avatar_url),
        last_sign_in_at = COALESCE(EXCLUDED.last_sign_in_at, public.user_usage.last_sign_in_at);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_sync ON auth.users;
CREATE TRIGGER on_auth_user_sync
AFTER INSERT OR UPDATE OF email, raw_user_meta_data, last_sign_in_at ON auth.users
FOR EACH ROW
EXECUTE FUNCTION public.handle_auth_user_sync();

-- Backfill any existing authenticated users who signed in before this migration ran
INSERT INTO public.user_usage (
    user_id,
    user_email,
    full_name,
    avatar_url,
    last_sign_in_at,
    total_predictions,
    image_predictions,
    last_prediction_at,
    created_at
)
SELECT
    id,
    COALESCE(email, ''),
    COALESCE(raw_user_meta_data->>'full_name', raw_user_meta_data->>'name', ''),
    raw_user_meta_data->>'avatar_url',
    COALESCE(last_sign_in_at, timezone('utc'::text, now())),
    0,
    0,
    NULL,
    COALESCE(created_at, timezone('utc'::text, now()))
FROM auth.users
ON CONFLICT (user_id) DO UPDATE
SET
    user_email = COALESCE(EXCLUDED.user_email, public.user_usage.user_email),
    full_name = COALESCE(NULLIF(EXCLUDED.full_name, ''), public.user_usage.full_name),
    avatar_url = COALESCE(EXCLUDED.avatar_url, public.user_usage.avatar_url),
    last_sign_in_at = COALESCE(EXCLUDED.last_sign_in_at, public.user_usage.last_sign_in_at);

-- 4. Keep usage counters updated whenever an audit log is inserted
CREATE OR REPLACE FUNCTION public.update_user_usage_stats()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_usage (
        user_id,
        user_email,
        total_predictions,
        image_predictions,
        last_prediction_at
    )
    VALUES (
        NEW.user_id,
        NEW.user_email,
        1,
        CASE WHEN NEW.prediction_type = 'Histopathology Image Classification' THEN 1 ELSE 0 END,
        NEW.created_at
    )
    ON CONFLICT (user_id) DO UPDATE
    SET
        total_predictions = public.user_usage.total_predictions + 1,
        image_predictions = public.user_usage.image_predictions + (
            CASE WHEN NEW.prediction_type = 'Histopathology Image Classification' THEN 1 ELSE 0 END
        ),
        last_prediction_at = NEW.created_at,
        user_email = COALESCE(NULLIF(NEW.user_email, ''), public.user_usage.user_email);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS after_audit_log_insert ON public.audit_logs;
CREATE TRIGGER after_audit_log_insert
AFTER INSERT ON public.audit_logs
FOR EACH ROW
EXECUTE FUNCTION public.update_user_usage_stats();

-- 5. Enable Row Level Security (RLS) so users can only view/write their own tracking rows
ALTER TABLE public.audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_usage ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own audit logs" ON public.audit_logs;
CREATE POLICY "Users can view own audit logs"
    ON public.audit_logs FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own audit logs" ON public.audit_logs;
CREATE POLICY "Users can insert own audit logs"
    ON public.audit_logs FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can view own usage" ON public.user_usage;
CREATE POLICY "Users can view own usage"
    ON public.user_usage FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own usage" ON public.user_usage;
CREATE POLICY "Users can insert own usage"
    ON public.user_usage FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update own usage" ON public.user_usage;
CREATE POLICY "Users can update own usage"
    ON public.user_usage FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- 6. Private Storage bucket & policies for uploaded biopsy images
INSERT INTO storage.buckets (id, name, public)
VALUES ('biopsy-images', 'biopsy-images', false)
ON CONFLICT (id) DO NOTHING;

DROP POLICY IF EXISTS "Users can upload own biopsy images" ON storage.objects;
CREATE POLICY "Users can upload own biopsy images"
    ON storage.objects FOR INSERT
    TO authenticated
    WITH CHECK (
        bucket_id = 'biopsy-images'
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

DROP POLICY IF EXISTS "Users can view own biopsy images" ON storage.objects;
CREATE POLICY "Users can view own biopsy images"
    ON storage.objects FOR SELECT
    TO authenticated
    USING (
        bucket_id = 'biopsy-images'
        AND (storage.foldername(name))[1] = auth.uid()::text
    );
