-- OncoSense Production Database Schema & SaaS Usage Triggers
-- This schema configures user audit logging and automatically tracks usage counts (quotas) for each user.

-- 1. Create Audit Logs Table (Tracks every API request)
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    user_email TEXT NOT NULL,
    prediction_type TEXT NOT NULL, -- 'Wisconsin Cytology', 'SEER Clinical Prognosis', or 'Histopathology Image'
    result TEXT NOT NULL,          -- 'Benign' or 'Malignant' (or survival probability)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 2. Create User Usage Table (Tracks SaaS quotas/limits per user)
CREATE TABLE IF NOT EXISTS user_usage (
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    user_email TEXT NOT NULL,
    total_predictions INTEGER DEFAULT 0 NOT NULL,
    image_predictions INTEGER DEFAULT 0 NOT NULL,
    last_prediction_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. Create Trigger Function to Automatically Update Usage Stats
-- Every time a user executes a prediction (resulting in a new audit log row),
-- this function updates their profile usage counts automatically.
CREATE OR REPLACE FUNCTION update_user_usage_stats()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO user_usage (user_id, user_email, total_predictions, image_predictions, last_prediction_at)
    VALUES (
        NEW.user_id,
        NEW.user_email,
        1,
        CASE WHEN NEW.prediction_type = 'Histopathology Image Classification' THEN 1 ELSE 0 END,
        NEW.created_at
    )
    ON CONFLICT (user_id) DO UPDATE
    SET 
        total_predictions = user_usage.total_predictions + 1,
        image_predictions = user_usage.image_predictions + (CASE WHEN NEW.prediction_type = 'Histopathology Image Classification' THEN 1 ELSE 0 END),
        last_prediction_at = NEW.created_at,
        user_email = NEW.user_email;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 4. Attach Trigger to Audit Logs Table
DROP TRIGGER IF EXISTS after_audit_log_insert ON audit_logs;
CREATE TRIGGER after_audit_log_insert
AFTER INSERT ON audit_logs
FOR EACH ROW
EXECUTE FUNCTION update_user_usage_stats();
