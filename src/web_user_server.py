#!/usr/bin/env python3

import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response, send_from_directory, send_file
from user_management import UserManager
from story_generator import StoryGenerator
from tts_helper import TTSHelper
from image_generator import ImageGenerator
from flask_cors import CORS
import json
import re

app = Flask(__name__, template_folder="../templates")
app.secret_key = os.urandom(24)
CORS(app)

user_manager = UserManager()
story_generator = StoryGenerator()
tts_helper = TTSHelper()
image_generator = ImageGenerator()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA_DIR = os.path.join(BASE_DIR, 'user_data')

@app.route("/")
def index():
    # If user is logged in, show main dashboard with two options
    if 'username' in session:
        username = session['username']
        user = user_manager.users.get(username)
        return render_template("dashboard.html", logged_in=True, user=user)
    return render_template("index.html", logged_in=False)

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json()
    username = data.get("username", "").strip()
    age = data.get("age")
    if not username or age is None:
        return jsonify({"error": "Username and age are required"}), 400
    try:
        age = int(age)
    except Exception:
        return jsonify({"error": "Invalid age"}), 400
    # No password or email
    if user_manager.register_user(username, age):
        return jsonify({"success": True}), 200
    else:
        return jsonify({"error": "Registration failed. Username might already exist or invalid age."}), 400

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json()
    username = data.get("username", "").strip()
    if not username:
        return jsonify({"error": "Username is required"}), 400
    # No password
    if user_manager.authenticate_user(username):
        session['username'] = username
        user = user_manager.users[username]
        return jsonify({"success": True, "user": {
            "username": user["username"],
            "age": user["age"],
            "created_at": user["created_at"],
            "last_login": user["last_login"]
        }}), 200
    else:
        return jsonify({"error": "Invalid username"}), 401

@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.pop('username', None)
    user_manager.logout()
    return jsonify({"success": True})

@app.route("/api/current_user")
def api_current_user():
    username = session.get('username')
    if username and username in user_manager.users:
        user = user_manager.users[username]
        return jsonify({"user": {
            "username": user["username"],
            "age": user["age"],
            "created_at": user["created_at"],
            "last_login": user["last_login"]
        }})
    return jsonify({"user": None})

@app.route("/api/users")
def api_users():
    users = [
        {
            "username": u["username"],
            "age": u["age"],
            "created_at": u["created_at"],
            "last_login": u["last_login"]
        }
        for u in user_manager.users.values()
    ]
    return jsonify({"users": users})

@app.route("/api/user_stats")
def api_user_stats():
    username = session.get('username')
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    stats = user_manager.get_user_stats(username)
    return jsonify(stats)

@app.route("/api/generate_story", methods=["POST"])
def api_generate_story():
    """Generate a therapeutic story for the logged-in user"""
    username = session.get('username')
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    user = user_manager.users.get(username)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    data = request.get_json() or {}
    child_name = data.get("child_name", username)  # Use username as default child name
    age = data.get("age", user.get("age", 4))  # Use user's age as default
    custom_prompt = data.get("custom_prompt")
    
    try:
        result = story_generator.generate_story(
            child_name=child_name,
            age=age,
            custom_prompt=custom_prompt
        )
        
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify({"error": result["error"]}), 500
            
    except Exception as e:
        return jsonify({"error": f"Story generation failed: {str(e)}"}), 500

@app.route("/api/generate_story_stream", methods=["POST"])
def api_generate_story_stream():
    """Generate a therapeutic story with streaming response"""
    username = session.get('username')
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    user = user_manager.users.get(username)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    data = request.get_json() or {}
    child_name = data.get("child_name", username)  # Use username as default child name
    age = data.get("age", user.get("age", 4))  # Use user's age as default
    custom_prompt = data.get("custom_prompt")
    
    def generate():
        try:
            for chunk in story_generator.generate_story_stream(
                child_name=child_name,
                age=age,
                custom_prompt=custom_prompt
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/plain')

@app.route("/start_assistant")
def start_assistant():
    # This endpoint can be used to redirect to the main assistant app
    # For now, just show a message
    return "<h2>QTrobot AI Assistant will start here (integration point).</h2>"

@app.route("/api/save_story", methods=["POST"])
def api_save_story():
    username = session.get('username')
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    user = user_manager.users.get(username)
    if not user:
        return jsonify({"error": "User not found"}), 404
    data = request.get_json() or {}
    story = data.get("story")
    metadata = data.get("metadata")
    if not story or not metadata:
        return jsonify({"error": "Missing story or metadata"}), 400
    
    # Prepare user stories directory
    user_dir = os.path.join(USER_DATA_DIR, username, "stories")
    os.makedirs(user_dir, exist_ok=True)
    
    # Use timestamp for unique filename
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"story_{ts}.json"
    fpath = os.path.join(user_dir, fname)
    
    # Save story and metadata
    with open(fpath, "w") as f:
        json.dump({"story": story, "metadata": metadata}, f, indent=2)
    
    # Generate images for all sentences in the story
    if image_generator.is_available():
        try:
            # Split story into sentences
            sentences = re.split(r'(?<=[.!?])\s+', story.strip())
            sentences = [s for s in sentences if s.strip()]
            
            # Create user-specific image directory
            user_images_dir = os.path.join(USER_DATA_DIR, username, 'story_images', fname.replace('.json', ''))
            os.makedirs(user_images_dir, exist_ok=True)
            
            # Generate images for each sentence
            image_paths = []
            for i, sentence in enumerate(sentences):
                image_path = image_generator.generate_story_scene_image(
                    sentence, 
                    story_context=f"Story about {metadata.get('child_name', 'a child')}",
                    output_dir=user_images_dir,
                    filename_prefix=f"story_scene_{i:03d}"
                )
                image_paths.append(image_path)
            
            print(f"Generated {len(image_paths)} images for story {fname}")
            
        except Exception as e:
            print(f"Error generating images for story {fname}: {str(e)}")
            # Continue even if image generation fails
    
    return jsonify({"success": True, "filename": fname})

@app.route("/generate")
def generate_games():
    """Game generation page - shows the original game selection interface"""
    if 'username' not in session:
        return redirect(url_for('index'))
    username = session['username']
    user = user_manager.users.get(username)
    return render_template("index.html", logged_in=True, user=user, show_game_selection=True)

@app.route("/play")
def play_games():
    """Play games page - shows the interactive games interface"""
    if 'username' not in session:
        return redirect(url_for('index'))
    username = session['username']
    user = user_manager.users.get(username)
    return render_template("play_games.html", logged_in=True, user=user)

@app.route("/api/read_user_stories", methods=["POST"])
def api_read_user_stories():
    """Read user's saved stories aloud using robot TTS"""
    username = session.get('username')
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    user = user_manager.users.get(username)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    try:
        # Get user's stories directory
        user_stories_dir = os.path.join(USER_DATA_DIR, username, "stories")
        
        if not os.path.exists(user_stories_dir):
            return jsonify({"error": "No stories found for this user"}), 404
        
        # Get all story files
        story_files = [f for f in os.listdir(user_stories_dir) if f.endswith('.json')]
        
        if not story_files:
            return jsonify({"error": "No stories found for this user"}), 404
        
        # Read the most recent story (or a random one)
        story_files.sort(reverse=True)  # Most recent first
        latest_story_file = story_files[0]
        story_path = os.path.join(user_stories_dir, latest_story_file)
        
        with open(story_path, 'r') as f:
            story_data = json.load(f)
        
        story_text = story_data.get('story', '')
        metadata = story_data.get('metadata', {})
        
        if not story_text:
            return jsonify({"error": "Story content is empty"}), 400
        
        # Make the robot speak the story
        if tts_helper.is_available():
            # Determine language from metadata or use default
            language = metadata.get('language', 'en-US')
            success = tts_helper.speak_story(story_text, language)
            
            if success:
                return jsonify({
                    "success": True,
                    "story": story_text,
                    "metadata": metadata,
                    "filename": latest_story_file,
                    "message": "Story is being read aloud by QTrobot!"
                }), 200
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to make robot speak the story",
                    "story": story_text,
                    "metadata": metadata,
                    "filename": latest_story_file
                }), 500
        else:
            # TTS not available, return story without speaking
            return jsonify({
                "success": True,
                "story": story_text,
                "metadata": metadata,
                "filename": latest_story_file,
                "message": "TTS not available. Story content provided.",
                "tts_available": False
            }), 200
        
    except Exception as e:
        return jsonify({"error": f"Error reading stories: {str(e)}"}), 500

@app.route("/api/read_specific_story", methods=["POST"])
def api_read_specific_story():
    """Read a specific story by filename"""
    username = session.get('username')
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json() or {}
    filename = data.get("filename")
    
    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    
    try:
        # Get user's stories directory
        user_stories_dir = os.path.join(USER_DATA_DIR, username, "stories")
        story_path = os.path.join(user_stories_dir, filename)
        
        if not os.path.exists(story_path):
            return jsonify({"error": "Story file not found"}), 404
        
        with open(story_path, 'r') as f:
            story_data = json.load(f)
        
        story_text = story_data.get('story', '')
        metadata = story_data.get('metadata', {})
        
        if not story_text:
            return jsonify({"error": "Story content is empty"}), 400
        
        # Make the robot speak the story
        if tts_helper.is_available():
            # Determine language from metadata or use default
            language = metadata.get('language', 'en-US')
            success = tts_helper.speak_story(story_text, language)
            
            if success:
                return jsonify({
                    "success": True,
                    "story": story_text,
                    "metadata": metadata,
                    "filename": filename,
                    "message": "Story is being read aloud by QTrobot!"
                }), 200
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to make robot speak the story",
                    "story": story_text,
                    "metadata": metadata,
                    "filename": filename
                }), 500
        else:
            # TTS not available, return story without speaking
            return jsonify({
                "success": True,
                "story": story_text,
                "metadata": metadata,
                "filename": filename,
                "message": "TTS not available. Story content provided.",
                "tts_available": False
            }), 200
        
    except Exception as e:
        return jsonify({"error": f"Error reading story: {str(e)}"}), 500

@app.route("/api/get_user_stories", methods=["GET"])
def api_get_user_stories():
    """Get list of user's saved stories"""
    username = session.get('username')
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    try:
        # Get user's stories directory
        user_stories_dir = os.path.join(USER_DATA_DIR, username, "stories")
        
        if not os.path.exists(user_stories_dir):
            return jsonify({"stories": []}), 200
        
        # Debug print statements
        print(f"Looking for stories in: {user_stories_dir}")
        print(f"Files found: {os.listdir(user_stories_dir)}")
        
        # Get all story files with metadata
        stories = []
        for filename in os.listdir(user_stories_dir):
            if filename.endswith('.json'):
                print(f"Processing file: {filename}")
                story_path = os.path.join(user_stories_dir, filename)
                try:
                    with open(story_path, 'r') as f:
                        story_data = json.load(f)
                    
                    print(f"Successfully loaded {filename}")
                    metadata = story_data.get('metadata', {})
                    # Use filename timestamp as fallback if generated_at is null
                    created_at = metadata.get('generated_at')
                    if not created_at and filename.startswith('story_'):
                        # Extract timestamp from filename like 'story_20250702_230233.json'
                        try:
                            timestamp_part = filename.replace('story_', '').replace('.json', '')
                            created_at = f"{timestamp_part[:8]} {timestamp_part[8:10]}:{timestamp_part[10:12]}:{timestamp_part[12:14]}"
                        except:
                            created_at = 'Unknown'
                    
                    stories.append({
                        "filename": filename,
                        "title": f"Story for {metadata.get('child_name', 'Unknown')}",
                        "age": metadata.get('age', 'Unknown'),
                        "word_count": metadata.get('word_count', 0),
                        "created_at": created_at or 'Unknown',
                        "preview": story_data.get('story', '')[:100] + "..." if len(story_data.get('story', '')) > 100 else story_data.get('story', '')
                    })
                    print(f"Added story: {filename}")
                except Exception as e:
                    # Skip corrupted files
                    print(f"Error processing {filename}: {str(e)}")
                    continue
        
        # Sort by creation date (newest first)
        stories.sort(key=lambda x: x.get('created_at', '') or '', reverse=True)
        
        # Debug print: show what we're returning
        print(f"Returning {len(stories)} stories: {stories}")
        
        return jsonify({"stories": stories}), 200
        
    except Exception as e:
        return jsonify({"error": f"Error getting stories: {str(e)}"}), 500

@app.route('/read_story/<filename>')
def read_story_page(filename):
    return render_template('read_story.html')

@app.route('/api/get_story_sentences')
def api_get_story_sentences():
    username = session.get('username')
    filename = request.args.get('filename')
    if not username or not filename:
        return jsonify({'success': False, 'error': 'Missing username or filename'})
    user_stories_dir = os.path.join(USER_DATA_DIR, username, 'stories')
    story_path = os.path.join(user_stories_dir, filename)
    if not os.path.exists(story_path):
        return jsonify({'success': False, 'error': 'Story not found'})
    try:
        with open(story_path, 'r') as f:
            story_data = json.load(f)
        story_text = story_data.get('story', '')
        metadata = story_data.get('metadata', {})
        # Split into sentences (simple split, can be improved)
        sentences = re.split(r'(?<=[.!?])\s+', story_text.strip())
        sentences = [s for s in sentences if s.strip()]
        
        return jsonify({
            'success': True, 
            'sentences': sentences, 
            'metadata': metadata,
            'images_available': image_generator.is_available()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_sentence_image', methods=['POST'])
def api_get_sentence_image():
    """Get image for a specific sentence"""
    username = session.get('username')
    data = request.get_json() or {}
    filename = data.get('filename', '')
    sentence_index = data.get('sentence_index', 0)
    
    print(f"Getting image for user: {username}, filename: {filename}, sentence_index: {sentence_index}")
    
    if not username or not filename:
        return jsonify({'success': False, 'error': 'Missing username or filename'})
    
    try:
        # Check for existing images
        user_images_dir = os.path.join(USER_DATA_DIR, username, 'story_images', filename.replace('.json', ''))
        print(f"Looking for images in: {user_images_dir}")
        
        if not os.path.exists(user_images_dir):
            print(f"Images directory does not exist: {user_images_dir}")
            return jsonify({'success': False, 'error': 'No images directory found'})
        
        print(f"Images directory exists. Files found: {os.listdir(user_images_dir)}")
        
        # Look for existing image files for this sentence
        # Try both patterns: with sentence index and without
        pattern_with_index = f'story_scene_{sentence_index:03d}_'
        pattern_without_index = 'story_scene_'
        
        existing_images = [f for f in os.listdir(user_images_dir) if f.startswith(pattern_with_index)]
        print(f"Looking for pattern '{pattern_with_index}', found: {existing_images}")
        
        # If no images found with index, try without index (for older images)
        if not existing_images:
            all_story_images = [f for f in os.listdir(user_images_dir) if f.startswith(pattern_without_index) and f.endswith('.png')]
            print(f"Looking for pattern '{pattern_without_index}', found: {all_story_images}")
            # Sort by creation time and take the sentence_index-th image
            all_story_images.sort()
            if sentence_index < len(all_story_images):
                existing_images = [all_story_images[sentence_index]]
                print(f"Using image {sentence_index} from sorted list: {existing_images}")
        
        if existing_images:
            # Use the first existing image
            image_path = f"/images/{username}/story_images/{os.path.basename(user_images_dir)}/{existing_images[0]}"
            print(f"Returning image path: {image_path}")
            return jsonify({
                'success': True, 
                'image_path': image_path
            })
        else:
            print(f"No image found for sentence {sentence_index}")
            return jsonify({'success': False, 'error': 'No image found for this sentence'})
            
    except Exception as e:
        print(f"Error getting image: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve generated images robustly"""
    full_path = os.path.join(USER_DATA_DIR, filename)
    print(f"Serving image: {filename}")
    print(f"Full path: {full_path}")
    print(f"File exists: {os.path.exists(full_path)}")
    if os.path.exists(full_path):
        return send_file(full_path)
    else:
        return "Image not found", 404

@app.route('/api/speak_sentence', methods=['POST'])
def api_speak_sentence():
    username = session.get('username')
    data = request.get_json() or {}
    sentence = data.get('sentence', '')
    filename = data.get('filename', '')
    if not username or not sentence:
        return jsonify({'success': False, 'error': 'Missing username or sentence'})
    # Optionally get language from story metadata
    language = 'en-US'
    if filename:
        user_stories_dir = os.path.join(USER_DATA_DIR, username, 'stories')
        story_path = os.path.join(user_stories_dir, filename)
        if os.path.exists(story_path):
            try:
                with open(story_path, 'r') as f:
                    story_data = json.load(f)
                metadata = story_data.get('metadata', {})
                language = metadata.get('language', 'en-US')
            except:
                pass
    tts_helper.speak_story(sentence, language)
    return jsonify({'success': True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True) 