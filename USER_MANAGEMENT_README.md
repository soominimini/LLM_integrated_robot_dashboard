# User Management System for QTrobot AI Assistant

This document describes the user management system that has been integrated into the QTrobot AI Assistant to provide user-specific chat history and personalized interactions.

## Overview

The user management system allows multiple users to register and login to the QTrobot AI Assistant, with each user having their own:
- User profile (username, age)
- Chat history directory
- Memory store for conversation persistence
- Personalized interactions

## Features

### User Registration
- **Username**: Required, must be unique
- **Age**: Required, must be between 0-150
- **Password**: Optional, for additional security
- **Automatic directory creation**: Each user gets their own chat history directory

### User Authentication
- **Login with username**: Required
- **Password authentication**: Optional (if set during registration)
- **Session management**: Tracks current user and login times

### User-Specific Data Management
- **Chat history**: Stored in `user_data/{username}/chat_history/`
- **Memory store**: User-specific conversation memory in `chat_memory.json`
- **User statistics**: Track chat files and storage usage

## File Structure

```
project_root/
├── src/
│   ├── user_management.py          # Core user management functionality
│   ├── user_cli_interface.py       # Command-line interface
│   └── qt_ai_data_assistant.py     # Modified main application
├── user_data/                      # User data directory (created automatically)
│   ├── alice/
│   │   └── chat_history/
│   │       └── chat_memory.json
│   └── bob/
│       └── chat_history/
│           └── chat_memory.json
├── users.json                      # User registry file
└── test_user_management.py         # Test script
```

## Usage

### Starting the Application

When you start the QTrobot AI Assistant, you'll now see a user authentication screen:

```bash
cd qt_ai_data_assistant
source venv/bin/activate
python src/qt_ai_data_assistant.py
```

### User Interface

The application will display a welcome screen with options:

```
🤖 Welcome to QTrobot AI Assistant
============================================================
Please login or register to continue.
============================================================

Main Menu:
1. Login
2. Register
3. List Users
4. Exit
------------------------------
```

### Registration Process

1. Choose option 2 (Register)
2. Enter username (must be unique)
3. Enter age (0-150)
4. Choose whether to set a password (optional)
5. If setting password, enter and confirm it

### Login Process

1. Choose option 1 (Login)
2. Enter username
3. Enter password (if required)
4. View user dashboard with statistics

### User Dashboard

After successful login, you'll see:

```
🎉 Welcome, alice!
==================================================
Age: 25
Member since: 2024-01-15
Last login: 2024-01-15

📊 Chat Statistics:
  • Chat files: 0
  • Storage used: 0.00 KB
  • Chat directory: user_data/alice/chat_history

✅ You are now logged in!
Your chat history will be automatically saved and restored.
==================================================
```

## Technical Details

### UserManager Class

The core user management functionality is provided by the `UserManager` class:

```python
from user_management import UserManager

# Initialize user manager
user_manager = UserManager()

# Register a new user
success = user_manager.register_user("alice", 25, "password123")

# Authenticate user
success = user_manager.authenticate_user("alice", "password123")

# Get current user info
current_user = user_manager.get_current_user()

# Get user-specific memory store path
mem_store_path = user_manager.get_user_mem_store_path()
```

### UserCLIInterface Class

The command-line interface is provided by the `UserCLIInterface` class:

```python
from user_cli_interface import UserCLIInterface

# Initialize CLI interface
cli_interface = UserCLIInterface(user_manager)

# Run the interface
cli_interface.run()
```

### Integration with Main Application

The user management system is integrated into the main `QTAIDataAssistant` class:

1. **Initialization**: User manager and CLI interface are created in `__init__`
2. **Authentication**: User authentication is handled in `setup()` before starting the chat engine
3. **Memory Store**: User-specific memory store is used in `_reset_chat_engine()`
4. **User Context**: User information is included in chat interactions in `_asr_callback()`

## Testing

You can test the user management system using the provided test script:

```bash
python test_user_management.py
```

This script will:
- Test user registration and authentication
- Test user statistics and listing
- Test the CLI interface components

## Configuration

### User Data Directory

By default, user data is stored in the `user_data/` directory. You can change this by modifying the `UserManager` initialization:

```python
user_manager = UserManager(base_dir="custom_user_data_path")
```

### User Registry File

User information is stored in `users.json` by default. You can change this by modifying the `UserManager` initialization:

```python
user_manager = UserManager(users_file="custom_users.json")
```

## Security Features

- **Password hashing**: Passwords are hashed using SHA-256 before storage
- **Optional passwords**: Users can register without passwords for convenience
- **User isolation**: Each user's data is stored in separate directories
- **Input validation**: Username and age validation to prevent invalid data

## Error Handling

The system includes comprehensive error handling for:
- Duplicate usernames
- Invalid age values
- Missing user files
- Authentication failures
- File system errors

## Future Enhancements

Potential improvements for the user management system:
- Web-based interface using Flask
- User roles and permissions
- User preferences and settings
- Backup and restore functionality
- User activity logging
- Multi-factor authentication

## Troubleshooting

### Common Issues

1. **"User not found" error**: Make sure the user is registered first
2. **"Username already exists"**: Choose a different username
3. **"Invalid age"**: Age must be between 0 and 150
4. **Permission errors**: Check file system permissions for the user data directory

### Debug Information

Enable debug logging by setting the ROS log level:

```bash
rosservice call /qt_robot/log_level "level: DEBUG"
```

## Support

For issues related to the user management system, please check:
1. The user data directory permissions
2. The users.json file format
3. The ROS logs for error messages

For additional support, contact the development team. 