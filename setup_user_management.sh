#!/bin/bash

# Copyright (c) 2024 LuxAI S.A.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Setup script for QTrobot AI Assistant User Management System

set -e

echo "🤖 Setting up QTrobot AI Assistant User Management System"
echo "========================================================"

# Check if we're in the right directory
if [ ! -f "src/qt_ai_data_assistant.py" ]; then
    echo "❌ Error: Please run this script from the qt_ai_data_assistant directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found. Please run the installation steps first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install Flask if not already installed
echo "📦 Checking Flask installation..."
if ! python -c "import flask" 2>/dev/null; then
    echo "📦 Installing Flask..."
    pip install flask==2.3.3
else
    echo "✅ Flask already installed"
fi

# Create user data directory
echo "📁 Creating user data directory..."
mkdir -p user_data

# Set permissions
echo "🔐 Setting permissions..."
chmod 755 user_data

# Test the user management system
echo "🧪 Testing user management system..."
if python test_user_management.py; then
    echo "✅ User management system test passed"
else
    echo "❌ User management system test failed"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To start the QTrobot AI Assistant with user management:"
echo "   source venv/bin/activate"
echo "   python src/qt_ai_data_assistant.py"
echo ""
echo "To test the user management system independently:"
echo "   python run_user_management.py"
echo ""
echo "For more information, see USER_MANAGEMENT_README.md" 