#!/usr/bin/env python3.9

# Copyright (c) 2024 LuxAI S.A.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import json
from flask import Flask, render_template_string, request, redirect, url_for, flash, session
from user_management import UserManager
import rospy


class UserWebInterface:
    """Web-based user interface for user registration and login."""
    
    def __init__(self, user_manager: UserManager, port: int = 6061):
        """
        Initialize the UserWebInterface.
        
        Args:
            user_manager: UserManager instance
            port: Port for the web interface
        """
        self.user_manager = user_manager
        self.port = port
        self.app = Flask(__name__)
        self.app.secret_key = os.urandom(24)
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main page - redirect to login or dashboard."""
            if self.user_manager.get_current_user():
                return redirect(url_for('dashboard'))
            return redirect(url_for('login'))
        
        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            """Login page."""
            if request.method == 'POST':
                username = request.form.get('username', '').strip()
                password = request.form.get('password', '')
                
                if self.user_manager.authenticate_user(username, password):
                    flash(f'Welcome back, {username}!', 'success')
                    return redirect(url_for('dashboard'))
                else:
                    flash('Invalid username or password', 'error')
            
            return render_template_string(self._get_login_template())
        
        @self.app.route('/register', methods=['GET', 'POST'])
        def register():
            """Registration page."""
            if request.method == 'POST':
                username = request.form.get('username', '').strip()
                age = request.form.get('age', '')
                password = request.form.get('password', '')
                confirm_password = request.form.get('confirm_password', '')
                
                # Validation
                if not username:
                    flash('Username is required', 'error')
                elif not age or not age.isdigit():
                    flash('Valid age is required', 'error')
                elif int(age) < 0 or int(age) > 150:
                    flash('Age must be between 0 and 150', 'error')
                elif password != confirm_password:
                    flash('Passwords do not match', 'error')
                else:
                    if self.user_manager.register_user(username, int(age), password):
                        flash(f'User {username} registered successfully! Please login.', 'success')
                        return redirect(url_for('login'))
                    else:
                        flash('Registration failed. Username might already exist.', 'error')
            
            return render_template_string(self._get_register_template())
        
        @self.app.route('/dashboard')
        def dashboard():
            """User dashboard."""
            current_user = self.user_manager.get_current_user()
            if not current_user:
                return redirect(url_for('login'))
            
            stats = self.user_manager.get_user_stats()
            return render_template_string(self._get_dashboard_template(), 
                                        user=current_user, 
                                        stats=stats)
        
        @self.app.route('/logout')
        def logout():
            """Logout user."""
            self.user_manager.logout()
            flash('You have been logged out', 'info')
            return redirect(url_for('login'))
        
        @self.app.route('/api/current_user')
        def api_current_user():
            """API endpoint to get current user info."""
            current_user = self.user_manager.get_current_user()
            if current_user:
                return json.dumps({
                    'username': current_user['username'],
                    'age': current_user['age'],
                    'chat_dir': current_user['chat_dir']
                })
            return json.dumps({'error': 'No user logged in'}), 401
    
    def _get_login_template(self):
        """Get the login page HTML template."""
        return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>QTrobot AI Assistant - Login</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0;
                    padding: 0;
                    height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .container {
                    background: white;
                    padding: 2rem;
                    border-radius: 10px;
                    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
                    width: 100%;
                    max-width: 400px;
                }
                .header {
                    text-align: center;
                    margin-bottom: 2rem;
                }
                .header h1 {
                    color: #333;
                    margin: 0;
                    font-size: 2rem;
                }
                .header p {
                    color: #666;
                    margin: 0.5rem 0 0 0;
                }
                .form-group {
                    margin-bottom: 1rem;
                }
                label {
                    display: block;
                    margin-bottom: 0.5rem;
                    color: #333;
                    font-weight: 500;
                }
                input[type="text"], input[type="password"] {
                    width: 100%;
                    padding: 0.75rem;
                    border: 2px solid #e1e5e9;
                    border-radius: 5px;
                    font-size: 1rem;
                    box-sizing: border-box;
                    transition: border-color 0.3s ease;
                }
                input[type="text"]:focus, input[type="password"]:focus {
                    outline: none;
                    border-color: #667eea;
                }
                .btn {
                    width: 100%;
                    padding: 0.75rem;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-size: 1rem;
                    cursor: pointer;
                    transition: transform 0.2s ease;
                }
                .btn:hover {
                    transform: translateY(-2px);
                }
                .links {
                    text-align: center;
                    margin-top: 1rem;
                }
                .links a {
                    color: #667eea;
                    text-decoration: none;
                }
                .links a:hover {
                    text-decoration: underline;
                }
                .flash {
                    padding: 0.75rem;
                    margin-bottom: 1rem;
                    border-radius: 5px;
                    text-align: center;
                }
                .flash.success {
                    background-color: #d4edda;
                    color: #155724;
                    border: 1px solid #c3e6cb;
                }
                .flash.error {
                    background-color: #f8d7da;
                    color: #721c24;
                    border: 1px solid #f5c6cb;
                }
                .flash.info {
                    background-color: #d1ecf1;
                    color: #0c5460;
                    border: 1px solid #bee5eb;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 QTrobot AI Assistant</h1>
                    <p>Welcome back! Please login to continue.</p>
                </div>
                
                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                            <div class="flash {{ category }}">{{ message }}</div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}
                
                <form method="POST">
                    <div class="form-group">
                        <label for="username">Username:</label>
                        <input type="text" id="username" name="username" required>
                    </div>
                    <div class="form-group">
                        <label for="password">Password (optional):</label>
                        <input type="password" id="password" name="password">
                    </div>
                    <button type="submit" class="btn">Login</button>
                </form>
                
                <div class="links">
                    <p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a></p>
                </div>
            </div>
        </body>
        </html>
        '''
    
    def _get_register_template(self):
        """Get the registration page HTML template."""
        return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>QTrobot AI Assistant - Register</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0;
                    padding: 0;
                    height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .container {
                    background: white;
                    padding: 2rem;
                    border-radius: 10px;
                    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
                    width: 100%;
                    max-width: 400px;
                }
                .header {
                    text-align: center;
                    margin-bottom: 2rem;
                }
                .header h1 {
                    color: #333;
                    margin: 0;
                    font-size: 2rem;
                }
                .header p {
                    color: #666;
                    margin: 0.5rem 0 0 0;
                }
                .form-group {
                    margin-bottom: 1rem;
                }
                label {
                    display: block;
                    margin-bottom: 0.5rem;
                    color: #333;
                    font-weight: 500;
                }
                input[type="text"], input[type="password"], input[type="number"] {
                    width: 100%;
                    padding: 0.75rem;
                    border: 2px solid #e1e5e9;
                    border-radius: 5px;
                    font-size: 1rem;
                    box-sizing: border-box;
                    transition: border-color 0.3s ease;
                }
                input[type="text"]:focus, input[type="password"]:focus, input[type="number"]:focus {
                    outline: none;
                    border-color: #667eea;
                }
                .btn {
                    width: 100%;
                    padding: 0.75rem;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-size: 1rem;
                    cursor: pointer;
                    transition: transform 0.2s ease;
                }
                .btn:hover {
                    transform: translateY(-2px);
                }
                .links {
                    text-align: center;
                    margin-top: 1rem;
                }
                .links a {
                    color: #667eea;
                    text-decoration: none;
                }
                .links a:hover {
                    text-decoration: underline;
                }
                .flash {
                    padding: 0.75rem;
                    margin-bottom: 1rem;
                    border-radius: 5px;
                    text-align: center;
                }
                .flash.success {
                    background-color: #d4edda;
                    color: #155724;
                    border: 1px solid #c3e6cb;
                }
                .flash.error {
                    background-color: #f8d7da;
                    color: #721c24;
                    border: 1px solid #f5c6cb;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 QTrobot AI Assistant</h1>
                    <p>Create your account to get started</p>
                </div>
                
                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                            <div class="flash {{ category }}">{{ message }}</div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}
                
                <form method="POST">
                    <div class="form-group">
                        <label for="username">Username:</label>
                        <input type="text" id="username" name="username" required>
                    </div>
                    <div class="form-group">
                        <label for="age">Age:</label>
                        <input type="number" id="age" name="age" min="0" max="150" required>
                    </div>
                    <div class="form-group">
                        <label for="password">Password (optional):</label>
                        <input type="password" id="password" name="password">
                    </div>
                    <div class="form-group">
                        <label for="confirm_password">Confirm Password:</label>
                        <input type="password" id="confirm_password" name="confirm_password">
                    </div>
                    <button type="submit" class="btn">Register</button>
                </form>
                
                <div class="links">
                    <p>Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
                </div>
            </div>
        </body>
        </html>
        '''
    
    def _get_dashboard_template(self):
        """Get the dashboard page HTML template."""
        return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>QTrobot AI Assistant - Dashboard</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0;
                    padding: 0;
                    min-height: 100vh;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 2rem;
                }
                .header {
                    background: white;
                    padding: 2rem;
                    border-radius: 10px;
                    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
                    margin-bottom: 2rem;
                    text-align: center;
                }
                .header h1 {
                    color: #333;
                    margin: 0;
                    font-size: 2.5rem;
                }
                .header p {
                    color: #666;
                    margin: 0.5rem 0 0 0;
                    font-size: 1.1rem;
                }
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 1rem;
                    margin-bottom: 2rem;
                }
                .stat-card {
                    background: white;
                    padding: 1.5rem;
                    border-radius: 10px;
                    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
                    text-align: center;
                }
                .stat-card h3 {
                    color: #333;
                    margin: 0 0 0.5rem 0;
                }
                .stat-card p {
                    color: #666;
                    margin: 0;
                    font-size: 1.1rem;
                }
                .actions {
                    background: white;
                    padding: 2rem;
                    border-radius: 10px;
                    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
                    text-align: center;
                }
                .btn {
                    display: inline-block;
                    padding: 0.75rem 1.5rem;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    font-size: 1rem;
                    margin: 0.5rem;
                    transition: transform 0.2s ease;
                }
                .btn:hover {
                    transform: translateY(-2px);
                }
                .btn.logout {
                    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
                }
                .user-info {
                    background: #f8f9fa;
                    padding: 1rem;
                    border-radius: 5px;
                    margin-bottom: 1rem;
                }
                .user-info p {
                    margin: 0.25rem 0;
                    color: #666;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 QTrobot AI Assistant</h1>
                    <p>Welcome, {{ user.username }}!</p>
                </div>
                
                <div class="user-info">
                    <p><strong>Username:</strong> {{ user.username }}</p>
                    <p><strong>Age:</strong> {{ user.age }}</p>
                    <p><strong>Member since:</strong> {{ user.created_at.split('T')[0] }}</p>
                    {% if user.last_login %}
                        <p><strong>Last login:</strong> {{ user.last_login.split('T')[0] }}</p>
                    {% endif %}
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>📁 Chat Files</h3>
                        <p>{{ stats.total_files }}</p>
                    </div>
                    <div class="stat-card">
                        <h3>💾 Storage Used</h3>
                        <p>{{ "%.2f"|format(stats.total_size / 1024) }} KB</p>
                    </div>
                    <div class="stat-card">
                        <h3>📂 Chat Directory</h3>
                        <p>{{ stats.user_dir.split('/')[-2] }}/{{ stats.user_dir.split('/')[-1] }}</p>
                    </div>
                </div>
                
                <div class="actions">
                    <h3>Ready to start chatting?</h3>
                    <p>Your chat history will be automatically saved and restored for future sessions.</p>
                    <a href="/qt_ai_data_assistant" class="btn">🚀 Start AI Assistant</a>
                    <a href="{{ url_for('logout') }}" class="btn logout">Logout</a>
                </div>
            </div>
        </body>
        </html>
        '''
    
    def run(self, debug=False):
        """Run the Flask application."""
        rospy.loginfo(f"Starting user interface on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)
    
    def get_current_user_info(self):
        """Get current user information for the main application."""
        return self.user_manager.get_current_user()
    
    def get_user_mem_store_path(self):
        """Get the memory store path for the current user."""
        return self.user_manager.get_user_mem_store_path() 