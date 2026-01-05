"""
🔥 OPTIMIZATION TOOL (ArcFace Edition)
Now uses: SHADE → L-SHADE → Local Refinement (security-first)

Commands:
  python run_optimization.py --create-baseline
  python run_optimization.py --optimize
  python run_optimization.py --show
"""

import sys
import json

from metaheuristic_optimizer import SHADE_LSHADE_SecurityOptimizer, ParticleConfig


def create_arcface_baseline_config():
    baseline = {
        "tolerance": 0.47,            # good CCTV starting point for ArcFace cosine
        "downscale_factor": 0.35,
        "recognition_interval": 0.60,
        "num_jitters": 0,             # unused in ArcFace pipeline
        "top_k": 1                    # unused in ArcFace pipeline
    }

    with open("optimal_config.json", "w") as f:
        json.dump(baseline, f, indent=2)

    print("✅ Created baseline optimal_config.json (ArcFace + Security bias)")
    print(json.dumps(baseline, indent=2))
    print("\n🎯 Notes:")
    print("   • ArcFace tolerance is cosine similarity threshold")
    print("   • Typical CCTV range: 0.44–0.52")
    print("   • Lower = stricter / fewer false matches")


def run_shade_optimization():
    print("\n" + "🔒" * 30)
    print("SHADE-LSHADE SECURITY OPTIMIZATION")
    print("🔒" * 30)

    optimizer = SHADE_LSHADE_SecurityOptimizer(
        config_file="optimal_config.json",
        seed=42
    )

    # In production you can pass a live worker to evaluate real metrics.
    # For now it uses the CCTV-biased simulation in metaheuristic_optimizer.py
    best = optimizer.optimize(worker=None)

    print("\n✅ Optimization complete!")
    print("📊 Best config saved to optimal_config.json:")
    print(json.dumps(best.to_dict(), indent=2))


def print_current_config():
    try:
        with open("optimal_config.json", "r") as f:
            cfg = json.load(f)

        print("\n📊 Current Optimal Configuration:")
        print("=" * 55)
        print(f"   tolerance:            {cfg.get('tolerance')}  (ArcFace cosine threshold)")
        print(f"   downscale_factor:     {cfg.get('downscale_factor')}")
        print(f"   recognition_interval: {cfg.get('recognition_interval')}")
        print(f"   num_jitters:          {cfg.get('num_jitters')} (unused in ArcFace)")
        print(f"   top_k:                {cfg.get('top_k')} (unused in ArcFace)")
        print("=" * 55)

    except FileNotFoundError:
        print("❌ No optimal_config.json found")
        print("   Run: python run_optimization.py --create-baseline")


def main():
    if len(sys.argv) <= 1:
        print_usage()
        return

    cmd = sys.argv[1].strip()

    if cmd == "--create-baseline":
        create_arcface_baseline_config()
    elif cmd == "--optimize":
        run_shade_optimization()
    elif cmd == "--show":
        print_current_config()
    else:
        print(f"❌ Unknown command: {cmd}")
        print_usage()


def print_usage():
    print("\n🔥 ARCFACE SECURITY OPTIMIZATION TOOL (SHADE-LSHADE)")
    print("=" * 60)
    print("\nUsage:")
    print("  python run_optimization.py --create-baseline")
    print("  python run_optimization.py --optimize")
    print("  python run_optimization.py --show")
    print("\nWorkflow:")
    print("  1) Create baseline")
    print("  2) Optimize (writes optimal_config.json)")
    print("  3) Run server and workers will auto-load config")
    print("=" * 60)


if __name__ == "__main__":
    main()
