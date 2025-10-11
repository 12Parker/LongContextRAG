#!/bin/bash
# Activation script for Long Context RAG project

echo "Activating Long Context RAG environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
source venv/bin/activate

echo "Virtual environment activated!"
echo "You can now run:"
echo "  python index.py          # Test basic functionality"
echo "  python examples.py       # Run examples"
echo "  python -c 'from index import LongContextRAG; print(\"Setup complete!\")'  # Quick test"
echo ""
echo "To deactivate, run: deactivate"
