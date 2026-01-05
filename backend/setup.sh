#!/bin/bash

echo "🚀 Setting up Grow AI Attendance System (ArcFace Edition)"
echo "=========================================================="

cd "$(dirname "$0")"

echo "📦 Creating virtual environment..."
python3 -m venv venv

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "📥 Installing dependencies..."
pip install --upgrade pip

# Remove old face_recognition/dlib
pip uninstall face_recognition dlib -y

# Install new requirements
pip install -r requirements.txt

echo ""
echo "✅ ArcFace setup complete!"
echo ""
echo "🎯 ARCFACE ADVANTAGES:"
echo "   • 512-D embeddings (vs 128-D in dlib)"
echo "   • Better for CCTV/IR/low-light"
echo "   • Higher accuracy (92-96%)"
echo "   • Faster on CPU (8-15 FPS)"
echo ""
echo "📋 NEXT STEPS:"
echo "1. Set BASE_URL: export BASE_URL=http://YOUR_SERVER_IP:8000"
echo "2. Create baseline config: python run_optimization.py --create-baseline"
echo "3. Start server: source venv/bin/activate && python main.py"
echo "4. Train faces: POST http://localhost:8000/train-from-drive"
echo "5. Test: GET http://localhost:8000/system-status"