#!/usr/bin/env python3.9

# Copyright (c) 2024 LuxAI S.A.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import json
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime
import rospy


class UserManager:
    """Manages user registration, authentication, and user-specific data directories."""
    
    def __init__(self, users_file: str = "users.json", base_dir: str = "user_data"):
        """
        Initialize the UserManager.
        
        Args:
            users_file: Path to the JSON file storing user information
            base_dir: Base directory for user-specific data
        """
        self.users_file = users_file
        self.base_dir = base_dir
        self.current_user = None
        
        # Create base directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Load existing users
        self.users = self._load_users()
    
    def _load_users(self) -> Dict[str, Dict[str, Any]]:
        """Load users from the JSON file."""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                rospy.logwarn(f"Error loading users file: {e}")
                return {}
        return {}
    
    def _save_users(self):
        """Save users to the JSON file."""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
        except IOError as e:
            rospy.logerr(f"Error saving users file: {e}")
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, username: str, age: int, password: str = "") -> bool:
        """
        Register a new user.
        
        Args:
            username: User's name
            age: User's age
            password: Optional password for authentication
            
        Returns:
            True if registration successful, False otherwise
        """
        if not username or not username.strip():
            rospy.logwarn("Username cannot be empty")
            return False
        
        if age < 0 or age > 150:
            rospy.logwarn("Invalid age")
            return False
        
        username = username.strip().lower()
        
        if username in self.users:
            rospy.logwarn(f"User '{username}' already exists")
            return False
        
        # Create user directory
        user_dir = os.path.join(self.base_dir, username)
        os.makedirs(user_dir, exist_ok=True)
        
        # Create chat history directory
        chat_dir = os.path.join(user_dir, "chat_history")
        os.makedirs(chat_dir, exist_ok=True)
        
        # Store user information
        self.users[username] = {
            "username": username,
            "age": age,
            "password_hash": self._hash_password(password) if password else "",
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "user_dir": user_dir,
            "chat_dir": chat_dir
        }
        
        self._save_users()
        rospy.loginfo(f"User '{username}' registered successfully")
        return True
    
    def authenticate_user(self, username: str, password: str = "") -> bool:
        """
        Authenticate a user.
        
        Args:
            username: User's name
            password: User's password (optional if no password was set)
            
        Returns:
            True if authentication successful, False otherwise
        """
        username = username.strip().lower()
        
        if username not in self.users:
            rospy.logwarn(f"User '{username}' not found")
            return False
        
        user = self.users[username]
        
        # If no password was set during registration, allow login without password
        if not user["password_hash"]:
            self._login_user(username)
            return True
        
        # Check password
        if self._hash_password(password) == user["password_hash"]:
            self._login_user(username)
            return True
        
        rospy.logwarn(f"Invalid password for user '{username}'")
        return False
    
    def _login_user(self, username: str):
        """Set the current user and update last login time."""
        self.current_user = username
        self.users[username]["last_login"] = datetime.now().isoformat()
        self._save_users()
        rospy.loginfo(f"User '{username}' logged in successfully")
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently logged-in user."""
        if self.current_user and self.current_user in self.users:
            return self.users[self.current_user]
        return None
    
    def get_user_chat_dir(self, username: str = None) -> Optional[str]:
        """Get the chat history directory for a user."""
        if username is None:
            username = self.current_user
        
        if username and username in self.users:
            return self.users[username]["chat_dir"]
        return None
    
    def get_user_mem_store_path(self, username: str = None) -> Optional[str]:
        """Get the memory store file path for a user."""
        chat_dir = self.get_user_chat_dir(username)
        if chat_dir:
            return os.path.join(chat_dir, "chat_memory.json")
        return None
    
    def logout(self):
        """Logout the current user."""
        if self.current_user:
            rospy.loginfo(f"User '{self.current_user}' logged out")
            self.current_user = None
    
    def list_users(self) -> list:
        """Get a list of all registered users."""
        return list(self.users.keys())
    
    def delete_user(self, username: str) -> bool:
        """
        Delete a user and their data.
        
        Args:
            username: Name of the user to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        username = username.strip().lower()
        
        if username not in self.users:
            rospy.logwarn(f"User '{username}' not found")
            return False
        
        # Remove user directory
        user_dir = self.users[username]["user_dir"]
        try:
            import shutil
            shutil.rmtree(user_dir)
        except OSError as e:
            rospy.logwarn(f"Error removing user directory: {e}")
        
        # Remove from users dict
        del self.users[username]
        
        # If this was the current user, logout
        if self.current_user == username:
            self.current_user = None
        
        self._save_users()
        rospy.loginfo(f"User '{username}' deleted successfully")
        return True
    
    def get_user_stats(self, username: str = None) -> Optional[Dict[str, Any]]:
        """Get statistics about a user's chat history."""
        if username is None:
            username = self.current_user
        
        if not username or username not in self.users:
            return None
        
        chat_dir = self.get_user_chat_dir(username)
        if not chat_dir or not os.path.exists(chat_dir):
            return {"total_files": 0, "total_size": 0}
        
        total_files = 0
        total_size = 0
        
        for filename in os.listdir(chat_dir):
            filepath = os.path.join(chat_dir, filename)
            if os.path.isfile(filepath):
                total_files += 1
                total_size += os.path.getsize(filepath)
        
        return {
            "total_files": total_files,
            "total_size": total_size,
            "user_dir": chat_dir
        } 