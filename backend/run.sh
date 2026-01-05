#!/bin/bash

echo "🚀 Starting Grow AI Attendance System (ArcFace Edition)"
echo "=========================================================="

# Set BASE_URL to your actual IP address
export BASE_URL=http://$(hostname -I | awk '{print $1}'):8000

echo "🔗 Base URL: $BASE_URL"
echo "📡 This is accessible from other devices at: $BASE_URL"

# Create optimal config if not exists
if [ ! -f "optimal_config.json" ]; then
    echo "📝 Creating default optimal_config.json..."
    python run_optimization.py --create-baseline
fi

# Start the server
python main.py