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
echo "1. Set BASE_URL: export BASE_URL=http://YOUR_SERVER_IP:8000"
echo "2. Test Google Sheets:   python test_google.py (if exists)"
echo "3. Start server:        source venv/bin/activate && python main.py"
echo "4. Train faces:         POST http://localhost:8000/train-from-drive"
echo "5. Health check:        GET http://localhost:8000/health"