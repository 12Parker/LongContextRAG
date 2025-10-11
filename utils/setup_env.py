#!/usr/bin/env python3
"""
Environment setup script for Long Context RAG project.
This script helps you create and configure your .env file securely.
"""

import os
import shutil
from pathlib import Path

def setup_environment():
    """Set up the environment configuration."""
    print("🔧 Setting up environment configuration...")
    
    env_file = Path(".env")
    template_file = Path("env.template")
    
    # Check if .env already exists
    if env_file.exists():
        print("⚠️  .env file already exists.")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response != 'y':
            print("Keeping existing .env file.")
            return
    
    # Copy template to .env
    if template_file.exists():
        shutil.copy(template_file, env_file)
        print("✅ Created .env file from template")
    else:
        print("❌ Template file not found. Creating basic .env file...")
        create_basic_env()
        return
    
    print("\n📝 Next steps:")
    print("1. Edit the .env file and add your API keys:")
    print("   - OPENAI_API_KEY=your_actual_api_key_here")
    print("   - Add any other API keys you need")
    print("\n2. The .env file is already in .gitignore, so your keys will be safe")
    print("\n3. Run 'python test_setup.py' to verify everything is working")

def create_basic_env():
    """Create a basic .env file if template is not available."""
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
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Created basic .env file")

def check_gitignore():
    """Check if .gitignore is properly configured."""
    gitignore_file = Path(".gitignore")
    
    if not gitignore_file.exists():
        print("⚠️  .gitignore file not found. Creating one...")
        create_gitignore()
    else:
        with open(gitignore_file, 'r') as f:
            content = f.read()
        
        if '.env' in content:
            print("✅ .gitignore properly configured for .env files")
        else:
            print("⚠️  .gitignore doesn't include .env files. Adding it...")
            with open(gitignore_file, 'a') as f:
                f.write('\n# Environment variables\n.env\n.env.local\n.env.*.local\n')
            print("✅ Updated .gitignore")

def create_gitignore():
    """Create a basic .gitignore file."""
    gitignore_content = """# Environment variables and secrets
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# Vector stores and data
vector_store/
data/
logs/
results/
*.db
*.sqlite
*.sqlite3

# API keys and secrets
secrets.json
config.json
api_keys.txt

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Temporary files
*.tmp
*.temp
*.log

# Model files (if downloading large models)
models/
*.bin
*.safetensors
*.pt
*.pth

# Cache
.cache/
.pytest_cache/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("✅ Created .gitignore file")

def main():
    """Main setup function."""
    print("🚀 Long Context RAG Environment Setup")
    print("=" * 40)
    
    # Check gitignore
    check_gitignore()
    print()
    
    # Setup environment
    setup_environment()
    
    print("\n🎉 Environment setup complete!")
    print("\nYour API keys will be safe because:")
    print("- .env file is in .gitignore")
    print("- Only the template file (env.template) is tracked by git")
    print("- Your actual keys stay local to your machine")

if __name__ == "__main__":
    main()
