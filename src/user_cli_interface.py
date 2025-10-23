#!/usr/bin/env python3.9

# Copyright (c) 2024 LuxAI S.A.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import sys
import getpass
from user_management import UserManager
import rospy


class UserCLIInterface:
    """Command-line user interface for user registration and login."""
    
    def __init__(self, user_manager: UserManager):
        """
        Initialize the UserCLIInterface.
        
        Args:
            user_manager: UserManager instance
        """
        self.user_manager = user_manager
    
    def show_welcome_screen(self):
        """Display the welcome screen."""
        print("\n" + "="*60)
        print("🤖 Welcome to QTrobot AI Assistant")
        print("="*60)
        print("Please login or register to continue.")
        print("="*60)
    
    def show_menu(self):
        """Display the main menu."""
        print("\nMain Menu:")
        print("1. Login")
        print("2. Register")
        print("3. List Users")
        print("4. Exit")
        print("-" * 30)
    
    def get_user_choice(self, prompt="Enter your choice: ", valid_choices=None):
        """Get user input with validation."""
        while True:
            try:
                choice = input(prompt).strip()
                if valid_choices is None or choice in valid_choices:
                    return choice
                else:
                    print(f"Please enter one of: {', '.join(valid_choices)}")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
            except EOFError:
                print("\nExiting...")
                sys.exit(0)
    
    def register_user(self):
        """Handle user registration."""
        print("\n" + "="*40)
        print("📝 User Registration")
        print("="*40)
        
        # Get username
        while True:
            username = input("Enter username: ").strip()
            if username:
                if username.lower() in self.user_manager.list_users():
                    print("❌ Username already exists. Please choose a different one.")
                    continue
                break
            else:
                print("❌ Username cannot be empty.")
        
        # Get age
        while True:
            try:
                age = int(input("Enter age: ").strip())
                if 0 <= age <= 150:
                    break
                else:
                    print("❌ Age must be between 0 and 150.")
            except ValueError:
                print("❌ Please enter a valid number for age.")
        
        # Get password (optional)
        password = ""
        use_password = self.get_user_choice("Do you want to set a password? (y/n): ", ['y', 'n', 'Y', 'N'])
        if use_password.lower() == 'y':
            while True:
                password = getpass.getpass("Enter password: ")
                confirm_password = getpass.getpass("Confirm password: ")
                if password == confirm_password:
                    break
                else:
                    print("❌ Passwords do not match. Please try again.")
        
        # Register the user
        if self.user_manager.register_user(username, age, password):
            print(f"✅ User '{username}' registered successfully!")
            return True
        else:
            print("❌ Registration failed.")
            return False
    
    def login_user(self):
        """Handle user login."""
        print("\n" + "="*40)
        print("🔐 User Login")
        print("="*40)
        
        username = input("Enter username: ").strip()
        if not username:
            print("❌ Username cannot be empty.")
            return False
        
        # Check if user exists
        if username.lower() not in self.user_manager.list_users():
            print("❌ User not found.")
            return False
        
        # Get password if required
        user_info = self.user_manager.users.get(username.lower())
        if user_info and user_info.get("password_hash"):
            password = getpass.getpass("Enter password: ")
        else:
            password = ""
        
        # Authenticate
        if self.user_manager.authenticate_user(username, password):
            print(f"✅ Welcome back, {username}!")
            return True
        else:
            print("❌ Invalid username or password.")
            return False
    
    def list_users(self):
        """List all registered users."""
        print("\n" + "="*40)
        print("👥 Registered Users")
        print("="*40)
        
        users = self.user_manager.list_users()
        if not users:
            print("No users registered yet.")
            return
        
        for i, username in enumerate(users, 1):
            user_info = self.user_manager.users[username]
            print(f"{i}. {username} (Age: {user_info['age']})")
            if user_info.get('last_login'):
                print(f"   Last login: {user_info['last_login'].split('T')[0]}")
            print()
    
    def show_user_dashboard(self):
        """Show user dashboard after successful login."""
        current_user = self.user_manager.get_current_user()
        if not current_user:
            return
        
        stats = self.user_manager.get_user_stats()
        
        print("\n" + "="*50)
        print(f"🎉 Welcome, {current_user['username']}!")
        print("="*50)
        print(f"Age: {current_user['age']}")
        print(f"Member since: {current_user['created_at'].split('T')[0]}")
        if current_user.get('last_login'):
            print(f"Last login: {current_user['last_login'].split('T')[0]}")
        
        print("\n📊 Chat Statistics:")
        print(f"  • Chat files: {stats['total_files']}")
        print(f"  • Storage used: {stats['total_size'] / 1024:.2f} KB")
        print(f"  • Chat directory: {stats['user_dir']}")
        
        print("\n✅ You are now logged in!")
        print("Your chat history will be automatically saved and restored.")
        print("="*50)
    
    def run(self):
        """Run the CLI interface."""
        self.show_welcome_screen()
        
        while True:
            self.show_menu()
            choice = self.get_user_choice(valid_choices=['1', '2', '3', '4'])
            
            if choice == '1':
                # Login
                if self.login_user():
                    self.show_user_dashboard()
                    return True  # Successfully logged in
            elif choice == '2':
                # Register
                if self.register_user():
                    print("\nRegistration successful! You can now login.")
            elif choice == '3':
                # List users
                self.list_users()
            elif choice == '4':
                # Exit
                print("\n👋 Goodbye!")
                sys.exit(0)
            
            input("\nPress Enter to continue...")
    
    def get_current_user_info(self):
        """Get current user information for the main application."""
        return self.user_manager.get_current_user()
    
    def get_user_mem_store_path(self):
        """Get the memory store path for the current user."""
        return self.user_manager.get_user_mem_store_path() 