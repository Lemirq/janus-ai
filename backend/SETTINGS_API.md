# Settings API Documentation

## Overview

The settings system provides global configuration for playback speed, verbose mode, and audio clip management for AI-generated responses.

## Settings File

Settings are persisted in `/backend/data/settings.json`:

```json
{
  "playbackSpeed": "medium",
  "verbose": false,
  "lastUpdated": "2025-10-26T12:34:56.789Z"
}
```

## API Endpoints

### Get Current Settings

```http
GET /api/settings
```

**Response:**

```json
{
  "playbackSpeed": "medium",
  "speedMultiplier": 1.0,
  "verbose": false,
  "lastUpdated": "2025-10-26T12:34:56.789Z"
}
```

### Update Settings

```http
POST /api/settings
```

Directly update settings values in the JSON file.

**Request Body:**

```json
{
  "playbackSpeed": "fast",
  "verbose": true
}
```

Both fields are optional - include only what you want to update.

**Valid playbackSpeed values:** `"slow"`, `"medium"`, `"fast"`

**Response:**

```json
{
  "playbackSpeed": "fast",
  "speedMultiplier": 1.5,
  "verbose": true,
  "lastUpdated": "2025-10-26T12:34:56.789Z"
}
```

**Error Response (invalid speed):**

```json
{
  "error": "invalid playbackSpeed, must be one of: slow, medium, fast"
}
```

### Increase Playback Speed

```http
POST /api/settings/speed/increase
```

Cycles through: `slow` → `medium` → `fast` → `slow`

**Response:**

```json
{
  "playbackSpeed": "fast",
  "speedMultiplier": 1.5,
  "verbose": false,
  "lastUpdated": "2025-10-26T12:34:56.789Z"
}
```

### Decrease Playback Speed

```http
POST /api/settings/speed/decrease
```

Cycles through: `fast` → `medium` → `slow` → `fast`

**Response:**

```json
{
  "playbackSpeed": "slow",
  "speedMultiplier": 0.75,
  "verbose": false,
  "lastUpdated": "2025-10-26T12:34:56.789Z"
}
```

### Toggle Verbose Mode

```http
POST /api/settings/verbose
```

Toggles verbose mode on/off. When enabled, detailed logs are printed for settings changes.

**Response:**

```json
{
  "playbackSpeed": "medium",
  "speedMultiplier": 1.0,
  "verbose": true,
  "lastUpdated": "2025-10-26T12:34:56.789Z"
}
```

### Repeat Last Audio Clip

```http
POST /api/sessions/<session_id>/repeat
```

Returns the last AI-generated audio clip for the specified session.

**Response:**

- `200 OK` - Returns the audio file (WAV format)
- `404 Not Found` - Session doesn't exist or no audio generated yet

**Error Response:**

```json
{
  "error": "session not found"
}
```

or

```json
{
  "error": "no audio generated yet"
}
```

## Speed Multipliers

- **slow**: 0.75x speed
- **medium**: 1.0x speed (normal)
- **fast**: 1.5x speed

The client should use the `speedMultiplier` value to adjust playback rate.

## Audio Storage

AI-generated audio clips are stored in:

```
/backend/data/sessions/<session_id>/audio/
```

Format: `response_<timestamp>_<sequence>.wav`

## Usage in AI Audio Generation

When generating AI audio responses, use the helper functions from `app.routes.sessions`:

```python
from app.routes.sessions import get_session_audio_dir, set_last_audio_clip

# Get the audio directory for a session
audio_dir = get_session_audio_dir(session_id)

# Save your generated audio
audio_path = os.path.join(audio_dir, f"response_{timestamp}_{sequence}.wav")
# ... save audio to audio_path ...

# Track this as the last audio clip
set_last_audio_clip(session_id, audio_path)
```

## Integration Notes

1. **Client Playback**: The client should fetch current settings on startup and apply the `speedMultiplier` to audio playback.

2. **Verbose Logging**: When `verbose` is true, the backend will print detailed logs for:
   - Speed changes
   - Settings updates
   - Audio clip tracking

3. **Persistence**: Settings are automatically persisted to disk. The last audio clip per session is stored both in-memory and in the session JSON file for recovery after restarts.

4. **Session Audio Directory**: The audio directory is automatically created when `get_session_audio_dir()` is called. No manual setup required.
