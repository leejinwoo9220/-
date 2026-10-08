# Public social video capture

Minimal evidence collector for the AI opportunity radar.

- Uses TikTok's official public embed/player URL.
- Does not log in, bypass CAPTCHA, spoof identity/region, or use cookies from a user account.
- A successful MP4 fetch is only marked `full_video` after ffprobe can read a positive duration.
- Player screenshots without a complete MP4 remain `partial_frames`.
- No API keys or secrets are stored in this public repository.

## Road-view time-lapse

See [`roadview-timelapse/`](roadview-timelapse/README.md): fisheye removal and era alignment of road-view captures, MiniMax H3 (Sogni) generation plan, and sign-exact finishing with frozen-subject repair.
