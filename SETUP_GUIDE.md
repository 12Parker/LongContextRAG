# 🔧 Setup Guide for Long Context RAG

This guide will help you set up your environment securely with proper API key management.

## 🚀 Quick Setup

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment (Secure API Key Setup)
```bash
python setup_env.py
```

This script will:
- ✅ Create a `.env` file from the template
- ✅ Ensure `.gitignore` is properly configured
- ✅ Keep your API keys secure (never committed to git)

### 4. Add Your API Keys
Edit the `.env` file and replace the placeholder values:
```bash
# Edit .env file
nano .env  # or use your preferred editor
```

Replace:
```
OPENAI_API_KEY=your_actual_api_key_here
```

### 5. Test Your Setup
```bash
python test_setup.py
```

### 6. Run Examples
```bash
python index.py          # Basic functionality
python examples.py       # Advanced examples
```

## 🔒 Security Features

Your API keys are protected by:

- **`.gitignore`** - Prevents `.env` from being committed to git
- **Template System** - Only `env.template` is tracked (no real keys)
- **Local Storage** - Your actual keys stay on your machine only

## 📁 File Structure

```
LongContextRAG/
├── .env                  # Your API keys (NOT in git)
├── env.template          # Template file (safe to commit)
├── .gitignore           # Protects your secrets
├── setup_env.py         # Environment setup script
└── ... (other files)
```

## 🆘 Troubleshooting

### "Module not found" errors
Make sure you're in the virtual environment:
```bash
source venv/bin/activate
```

### "API key not found" errors
1. Run `python setup_env.py`
2. Edit `.env` file with your actual API key
3. Run `python test_setup.py` to verify

### Virtual environment issues
```bash
# Remove and recreate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🎯 Next Steps

Once setup is complete:

1. **Add Documents**: Put your research documents in the `data/` directory
2. **Experiment**: Try different prompts in `prompts.py`
3. **Research**: Use the examples in `examples.py` as starting points
4. **Customize**: Modify `config.py` for your specific needs

## 📚 Additional Resources

- See `README.md` for detailed project information
- Check `examples.py` for usage examples
- Review `prompts.py` for different prompt strategies
