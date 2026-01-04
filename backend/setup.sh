#!/bin/bash

echo "🚀 Setting up Grow AI Attendance System"
echo "========================================="

cd "$(dirname "$0")"

echo "📦 Creating virtual environment..."
python3 -m venv venv

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 NEXT STEPS:"
echo "1. Test Google Drive:   python test_google.py"
echo "2. Start server:        source venv/bin/activate && python main.py"
echo "3. Force train:         POST http://localhost:8000/force-train"
echo "4. Health check:        GET http://localhost:8000/health"