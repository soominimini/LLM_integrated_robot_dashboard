#!/usr/bin/env python3

# Copyright (c) 2024 LuxAI S.A.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
Standalone user management system runner.
This script allows you to test the user management functionality independently.
"""

import os
import sys
import rospy

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from user_management import UserManager
from user_cli_interface import UserCLIInterface


def main():
    """Main function to run the user management system."""
    print("🚀 Starting QTrobot AI Assistant User Management System")
    print("=" * 60)
    
    # Initialize ROS node
    rospy.init_node('user_management_standalone', anonymous=True)
    
    try:
        # Initialize user manager
        user_manager = UserManager()
        
        # Initialize CLI interface
        cli_interface = UserCLIInterface(user_manager)
        
        # Run the user interface
        success = cli_interface.run()
        
        if success:
            print("\n✅ User authentication successful!")
            print("You can now start the main QTrobot AI Assistant.")
            print("\nTo start the main application, run:")
            print("python src/qt_ai_data_assistant.py")
        else:
            print("\n❌ User authentication failed or cancelled.")
            
    except KeyboardInterrupt:
        print("\n\n👋 User management system stopped by user.")
    except Exception as e:
        print(f"\n❌ Error in user management system: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rospy.signal_shutdown("User management system stopped")


if __name__ == "__main__":
    main() 