"""
🔥 METAHEURISTIC OPTIMIZATION SETUP SCRIPT

Run this ONCE to find optimal parameters for your system.
Then the system will use these parameters automatically.
"""

import sys
import json
from metaheuristic_optimizer import PSO_HillClimbing_Optimizer, ParticleConfig


def create_baseline_config():
    """Create a baseline configuration file with MULTI-TRACKING defaults"""
    baseline = {
        "tolerance": 0.54,
        "downscale_factor": 0.35,
        "recognition_interval": 0.75,
        "num_jitters": 1,
        "top_k": 5
    }
    
    with open("optimal_config.json", "w") as f:
        json.dump(baseline, f, indent=2)
    
    print("✅ Created MULTI-TRACKING baseline optimal_config.json")
    print("   (Optimized for multiple bounding boxes)")
    print(json.dumps(baseline, indent=2))
    
    # Also create a config file for attendance_worker settings
    worker_settings = {
        "note": "These settings are hardcoded in AttendanceWorker class",
        "recommended_settings": {
            "iou_threshold": 0.18,
            "min_face_width": 75,
            "min_face_height": 75,
            "aspect_ratio_min": 0.65,
            "aspect_ratio_max": 1.5
        }
    }
    
    with open("worker_settings.json", "w") as f:
        json.dump(worker_settings, f, indent=2)


def run_full_optimization():
    """Run full PSO + Hill Climbing optimization"""
    print("\n" + "🔥"*30)
    print("METAHEURISTIC OPTIMIZATION")
    print("This will take a few minutes...")
    print("🔥"*30 + "\n")
    
    # Create optimizer
    optimizer = PSO_HillClimbing_Optimizer()
    
    # TODO: In production, you would:
    # 1. Start a camera worker
    # 2. Pass it to optimizer.optimize(worker)
    # 3. Let it test configurations on live feed
    
    # For now, we'll use the simulated version
    print("⚠️  NOTE: Running in SIMULATION mode")
    print("   For production, integrate with live camera worker\n")
    
    # Run optimization (pass None since we're simulating)
    optimal_config = optimizer.optimize(None)
    
    print("\n✅ Optimization complete!")
    print("   The system will now use these optimal parameters automatically.")


def print_current_config():
    """Print current configuration"""
    try:
        with open("optimal_config.json", "r") as f:
            config = json.load(f)
        
        print("\n📊 Current Optimal Configuration:")
        print("="*50)
        for key, value in config.items():
            print(f"   {key:25s}: {value}")
        print("="*50)
        
        print("\n🎯 Multi-Tracking Settings (hardcoded):")
        print("="*50)
        print("   iou_threshold: 0.18")
        print("   min_face_width: 75")
        print("   min_face_height: 75")
        print("   aspect_ratio: 0.65 - 1.5")
        print("="*50)
        
    except FileNotFoundError:
        print("❌ No configuration file found")
        print("   Run: python run_optimization.py --create-baseline")


def main():
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--create-baseline":
            create_baseline_config()
        
        elif command == "--optimize":
            run_full_optimization()
        
        elif command == "--show":
            print_current_config()
        
        else:
            print(f"❌ Unknown command: {command}")
            print_usage()
    
    else:
        print_usage()


def print_usage():
    print("\n🔥 METAHEURISTIC OPTIMIZATION TOOL")
    print("="*60)
    print("\nUsage:")
    print("  python run_optimization.py --create-baseline")
    print("     → Create baseline config for MULTI-TRACKING")
    print()
    print("  python run_optimization.py --optimize")
    print("     → Run full PSO + Hill Climbing optimization")
    print()
    print("  python run_optimization.py --show")
    print("     → Show current configuration")
    print()
    print("="*60)
    print("\n📝 Recommended workflow:")
    print("   1. python run_optimization.py --create-baseline")
    print("   2. Set BASE_URL: export BASE_URL=http://YOUR_IP:8000")
    print("   3. Start server: uvicorn main:app --host 0.0.0.0 --port 8000")
    print("   4. Camera will show multiple bounding boxes with clear labels")
    print()


if __name__ == "__main__":
    main()