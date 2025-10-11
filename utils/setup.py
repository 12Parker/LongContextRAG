"""
Setup script for the Long Context RAG research project.
"""

import os
import subprocess
import sys
from pathlib import Path

def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_file = Path(".env")
    env_template = Path("env.template")
    
    if not env_file.exists():
        if env_template.exists():
            # Copy template to .env
            with open(env_template, 'r') as f:
                content = f.read()
            with open(env_file, 'w') as f:
                f.write(content)
            print("Created .env file from template. Please update with your API keys.")
        else:
            # Create basic .env file
            env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview

# Vector Store Configuration
VECTOR_STORE_TYPE=chroma
EMBEDDING_MODEL=text-embedding-3-large

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
MAX_CONTEXT_LENGTH=8000

# Research Settings
ENABLE_LONG_CONTEXT=True
CONTEXT_WINDOW_SIZE=32000
"""
            with open(env_file, 'w') as f:
                f.write(env_content)
            print("Created .env file. Please update with your API keys.")
    else:
        print(".env file already exists.")

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        # Check if we're in a virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("Virtual environment detected. Installing dependencies...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        else:
            print("No virtual environment detected. Creating one...")
            subprocess.check_call([sys.executable, "-m", "venv", "venv"])
            print("Virtual environment created. Please activate it and run:")
            print("source venv/bin/activate")
            print("pip install -r requirements.txt")
            return False
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories."""
    directories = ["data", "vector_store", "logs", "results"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")

def setup_project():
    """Main setup function."""
    print("Setting up Long Context RAG research project...")
    
    # Create .env file
    create_env_file()
    
    # Install dependencies
    if not install_dependencies():
        print("Setup failed during dependency installation.")
        return False
    
    # Create directories
    create_directories()
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Update the .env file with your OpenAI API key")
    print("2. Run 'python index.py' to test the basic functionality")
    print("3. Run 'python examples.py' to see various usage examples")
    print("4. Add your own documents to the 'data' directory")
    
    return True

if __name__ == "__main__":
    setup_project()
