#!/usr/bin/env python3.9

# Copyright (c) 2024 LuxAI S.A.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
Test script for the user management system.
This script demonstrates how to use the user management functionality.
"""

import os
import sys
import rospy

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from user_management import UserManager
from user_cli_interface import UserCLIInterface


def test_user_management():
    """Test the user management functionality."""
    print("🧪 Testing User Management System")
    print("=" * 50)
    
    # Initialize user manager
    user_manager = UserManager()
    
    # Test user registration
    print("\n1. Testing user registration...")
    success = user_manager.register_user("alice", 25, "password123")
    print(f"Registration result: {success}")
    
    success = user_manager.register_user("bob", 30, "")
    print(f"Registration result (no password): {success}")
    
    # Test duplicate registration
    success = user_manager.register_user("alice", 26, "different_password")
    print(f"Duplicate registration result: {success}")
    
    # Test user authentication
    print("\n2. Testing user authentication...")
    success = user_manager.authenticate_user("alice", "password123")
    print(f"Authentication result (correct password): {success}")
    
    success = user_manager.authenticate_user("alice", "wrong_password")
    print(f"Authentication result (wrong password): {success}")
    
    success = user_manager.authenticate_user("bob", "")
    print(f"Authentication result (no password): {success}")
    
    # Test user stats
    print("\n3. Testing user statistics...")
    current_user = user_manager.get_current_user()
    if current_user:
        print(f"Current user: {current_user['username']}")
        stats = user_manager.get_user_stats()
        print(f"User stats: {stats}")
    
    # Test user listing
    print("\n4. Testing user listing...")
    users = user_manager.list_users()
    print(f"Registered users: {users}")
    
    # Test user logout
    print("\n5. Testing user logout...")
    user_manager.logout()
    current_user = user_manager.get_current_user()
    print(f"Current user after logout: {current_user}")
    
    print("\n✅ User management test completed!")


def test_cli_interface():
    """Test the CLI interface."""
    print("\n🧪 Testing CLI Interface")
    print("=" * 50)
    
    # Initialize user manager and CLI interface
    user_manager = UserManager()
    cli_interface = UserCLIInterface(user_manager)
    
    # Show welcome screen
    cli_interface.show_welcome_screen()
    
    # Show menu
    cli_interface.show_menu()
    
    # Test user choice validation
    print("\nTesting user choice validation...")
    choice = cli_interface.get_user_choice("Enter choice (1-4): ", ['1', '2', '3', '4'])
    print(f"User choice: {choice}")
    
    print("\n✅ CLI interface test completed!")


if __name__ == "__main__":
    # Initialize ROS node for testing
    rospy.init_node('test_user_management', anonymous=True)
    
    try:
        # Test user management functionality
        test_user_management()
        
        # Test CLI interface
        test_cli_interface()
        
        print("\n🎉 All tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        rospy.signal_shutdown("Test completed") 