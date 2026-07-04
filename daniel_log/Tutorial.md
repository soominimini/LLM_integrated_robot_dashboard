# Important
Always power off robot and pull out cord

# Accessing Repo
password: qtrobot
repo: /home/qtrobot/tutorials-master/demos/version_1_llm_gemini

# Run web app
source .venv/bin/activate
python src/web_user_server.py
http://127.0.0.1:8080

# Object detection
Generate games > Object detection to customize list

# Test design
Position
- Place toys at different angles (upside down, sideways etc.)
- Place camera at different angles (?)
- Place toys near/ far the robot's camera
Concealment
- Cover toys (50% from bottom, 100% etc. Think about how a kid would hold the toy)
Lighting levels (Sometimes color might not be recognized properly)
Prompt is a strawberry. I showed a strawberry but it recognizes it

# Optimize prompts
- We want the robot to better recognize these toys

# Files to work on
gemini_analyze_image.py
gemini_validate_spatial.py
gemini_validate_spatial_video.py

# Bonus: Clean up repo
- Analyze the whole project (Ex. where human_tracking.py is injected)
- TTS (Cleaner version of Polly/ qt services)
- Whisper (Speech recognition)