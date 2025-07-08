#!/usr/bin/env python3
"""
Startup script for the RAG Document Q&A Application
"""

import os
import sys
from app import app

if __name__ == '__main__':
    print("🧠 Starting RAG Document Q&A Application...")
    print("📁 Make sure you have set up your .env file with OPENAI_API_KEY")
    print("🌐 The application will be available at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1) 