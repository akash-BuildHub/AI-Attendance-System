"""
🔥 HYBRID METAHEURISTIC OPTIMIZER
PSO (Particle Swarm Optimization) + Hill Climbing

Optimizes:
- Recognition accuracy
- False Unknown rate
- FPS stability
- CPU usage
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import threading


@dataclass
class ParticleConfig:
    """Parameter configuration (particle in PSO)"""
    tolerance: float          # 0.35 - 0.65
    downscale_factor: float   # 0.20 - 0.40
    recognition_interval: float  # 0.3 - 0.8
    num_jitters: int          # 1 - 3
    top_k: int                # 3 - 7
    
    def to_dict(self) -> Dict:
        return {
            'tolerance': self.tolerance,
            'downscale_factor': self.downscale_factor,
            'recognition_interval': self.recognition_interval,
            'num_jitters': self.num_jitters,
            'top_k': self.top_k
        }
    
    @staticmethod
    def from_dict(d: Dict) -> 'ParticleConfig':
        return ParticleConfig(
            tolerance=d['tolerance'],
            downscale_factor=d['downscale_factor'],
            recognition_interval=d['recognition_interval'],
            num_jitters=d['num_jitters'],
            top_k=d['top_k']
        )


@dataclass
class FitnessMetrics:
    """Metrics for fitness evaluation"""
    recognition_accuracy: float  # % correct recognitions
    false_unknown_rate: float    # % known shown as Unknown
    false_match_rate: float      # % wrong person matches
    avg_fps: float               # Average stream FPS
    cpu_usage: float             # CPU usage %
    
    def calculate_fitness(self) -> float:
        """
        Multi-objective fitness function
        Higher is better
        """
        fitness = (
            + 3.0 * self.recognition_accuracy
            - 2.5 * self.false_unknown_rate
            - 2.0 * self.false_match_rate
            + 1.5 * (self.avg_fps / 20.0)  # Normalize to 0-1
            - 1.0 * (self.cpu_usage / 100.0)  # Normalize to 0-1
        )
        return fitness


class PSO_HillClimbing_Optimizer:
    """
    🔥 Hybrid Metaheuristic Optimizer
    
    Phase 1: PSO explores parameter space globally
    Phase 2: Hill Climbing refines best solution locally
    """
    
    def __init__(self, config_file: str = "optimal_config.json"):
        self.config_file = config_file
        
        # PSO parameters
        self.n_particles = 15
        self.n_iterations = 30
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        
        # Hill climbing parameters
        self.hill_climb_steps = 10
        self.step_size = 0.02
        
        # 🔴 FIX #6: Corrected parameter bounds
        self.bounds = {
            'tolerance': (0.35, 0.65),
            'downscale_factor': (0.30, 0.50),  # Increased for CCTV
            'recognition_interval': (0.3, 0.8),
            'num_jitters': (1, 3),
            'top_k': (3, 7)
        }
        
        # Best solution tracking
        self.global_best_config: Optional[ParticleConfig] = None
        self.global_best_fitness: float = -float('inf')
        
        print("🧠 Metaheuristic Optimizer initialized")
        print(f"   PSO: {self.n_particles} particles × {self.n_iterations} iterations")
        print(f"   Hill Climbing: {self.hill_climb_steps} refinement steps")
    
    def initialize_particle(self) -> Tuple[ParticleConfig, np.ndarray]:
        """Initialize random particle with velocity"""
        config = ParticleConfig(
            tolerance=np.random.uniform(*self.bounds['tolerance']),
            downscale_factor=np.random.uniform(*self.bounds['downscale_factor']),
            recognition_interval=np.random.uniform(*self.bounds['recognition_interval']),
            num_jitters=np.random.randint(self.bounds['num_jitters'][0], self.bounds['num_jitters'][1] + 1),
            top_k=np.random.randint(self.bounds['top_k'][0], self.bounds['top_k'][1] + 1)
        )
        
        # Initialize velocity (small random values)
        velocity = np.array([
            np.random.uniform(-0.05, 0.05),  # tolerance
            np.random.uniform(-0.02, 0.02),  # downscale
            np.random.uniform(-0.05, 0.05),  # interval
            np.random.randint(-1, 2),         # jitters
            np.random.randint(-1, 2)          # top_k
        ])
        
        return config, velocity
    
    def config_to_vector(self, config: ParticleConfig) -> np.ndarray:
        """Convert config to numpy vector for PSO operations"""
        return np.array([
            config.tolerance,
            config.downscale_factor,
            config.recognition_interval,
            float(config.num_jitters),
            float(config.top_k)
        ])
    
    def vector_to_config(self, vector: np.ndarray) -> ParticleConfig:
        """Convert numpy vector back to config"""
        return ParticleConfig(
            tolerance=np.clip(vector[0], *self.bounds['tolerance']),
            downscale_factor=np.clip(vector[1], *self.bounds['downscale_factor']),
            recognition_interval=np.clip(vector[2], *self.bounds['recognition_interval']),
            num_jitters=int(np.clip(vector[3], *self.bounds['num_jitters'])),
            top_k=int(np.clip(vector[4], *self.bounds['top_k']))
        )
    
    def evaluate_fitness(self, config: ParticleConfig, worker) -> float:
        """
        Evaluate fitness by testing configuration on live camera feed
        
        This is a SIMULATED version - in production, you'd:
        1. Apply config to worker
        2. Run for 30-60 seconds
        3. Collect metrics
        4. Calculate fitness
        """
        # TODO: Replace with actual testing
        # For now, we'll simulate based on heuristics
        
        # Heuristic fitness (replace with real metrics)
        recognition_accuracy = 0.85 - abs(config.tolerance - 0.50) * 0.5
        false_unknown_rate = abs(config.tolerance - 0.50) * 0.4
        false_match_rate = max(0, (0.60 - config.tolerance) * 0.3)
        
        # FPS heuristic (lower interval = higher FPS, but CPU cost)
        fps_score = 1.0 / (config.recognition_interval + 0.1)
        avg_fps = min(20, fps_score * 5)
        
        # CPU heuristic
        cpu_base = 40
        cpu_base += (1 - config.downscale_factor) * 30  # Higher resolution = more CPU
        cpu_base += config.num_jitters * 5
        cpu_base -= config.recognition_interval * 20
        cpu_usage = np.clip(cpu_base, 20, 100)
        
        metrics = FitnessMetrics(
            recognition_accuracy=recognition_accuracy,
            false_unknown_rate=false_unknown_rate,
            false_match_rate=false_match_rate,
            avg_fps=avg_fps,
            cpu_usage=cpu_usage
        )
        
        return metrics.calculate_fitness()
    
    def pso_optimize(self, worker) -> ParticleConfig:
        """
        🔥 Phase 1: Particle Swarm Optimization
        Global exploration of parameter space
        """
        print("\n" + "="*60)
        print("🔄 PHASE 1: PSO Optimization")
        print("="*60)
        
        # Initialize swarm
        particles = []
        velocities = []
        personal_best_configs = []
        personal_best_fitness = []
        
        for i in range(self.n_particles):
            config, velocity = self.initialize_particle()
            fitness = self.evaluate_fitness(config, worker)
            
            particles.append(config)
            velocities.append(velocity)
            personal_best_configs.append(config)
            personal_best_fitness.append(fitness)
            
            # Update global best
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_config = config
        
        print(f"✅ Initialized {self.n_particles} particles")
        print(f"   Initial best fitness: {self.global_best_fitness:.3f}")
        
        # PSO iterations
        for iteration in range(self.n_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{self.n_iterations}")
            
            for i in range(self.n_particles):
                # Convert to vectors
                x = self.config_to_vector(particles[i])
                v = velocities[i]
                p_best = self.config_to_vector(personal_best_configs[i])
                g_best = self.config_to_vector(self.global_best_config)
                
                # Update velocity
                r1, r2 = np.random.random(5), np.random.random(5)
                v = (self.w * v + 
                     self.c1 * r1 * (p_best - x) + 
                     self.c2 * r2 * (g_best - x))
                
                # Update position
                x = x + v
                
                # Convert back and clip
                new_config = self.vector_to_config(x)
                velocities[i] = v
                particles[i] = new_config
                
                # Evaluate
                fitness = self.evaluate_fitness(new_config, worker)
                
                # Update personal best
                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_configs[i] = new_config
                
                # Update global best
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_config = new_config
                    print(f"   🎯 New best! Fitness: {fitness:.3f}")
                    print(f"      tolerance={new_config.tolerance:.3f}, "
                          f"downscale={new_config.downscale_factor:.3f}, "
                          f"interval={new_config.recognition_interval:.3f}")
        
        print("\n" + "="*60)
        print(f"✅ PSO Complete - Best fitness: {self.global_best_fitness:.3f}")
        print("="*60)
        
        return self.global_best_config
    
    def hill_climbing_refine(self, config: ParticleConfig, worker) -> ParticleConfig:
        """
        🔥 Phase 2: Hill Climbing
        Local refinement of best solution
        """
        print("\n" + "="*60)
        print("⛰️  PHASE 2: Hill Climbing Refinement")
        print("="*60)
        
        current_config = config
        current_fitness = self.evaluate_fitness(config, worker)
        
        print(f"Starting fitness: {current_fitness:.3f}")
        
        for step in range(self.hill_climb_steps):
            improved = False
            
            # Try small perturbations
            for param in ['tolerance', 'recognition_interval']:
                for delta in [-self.step_size, self.step_size]:
                    # Create neighbor
                    neighbor_dict = current_config.to_dict()
                    neighbor_dict[param] += delta
                    
                    # Clip to bounds
                    neighbor_dict[param] = np.clip(
                        neighbor_dict[param], 
                        *self.bounds[param]
                    )
                    
                    neighbor_config = ParticleConfig.from_dict(neighbor_dict)
                    neighbor_fitness = self.evaluate_fitness(neighbor_config, worker)
                    
                    # Accept if better
                    if neighbor_fitness > current_fitness:
                        current_config = neighbor_config
                        current_fitness = neighbor_fitness
                        improved = True
                        print(f"   Step {step+1}: Improved to {current_fitness:.3f} "
                              f"({param} → {neighbor_dict[param]:.3f})")
                        break
                
                if improved:
                    break
            
            if not improved:
                print(f"   Step {step+1}: No improvement (converged)")
                break
        
        print("\n" + "="*60)
        print(f"✅ Hill Climbing Complete - Final fitness: {current_fitness:.3f}")
        print("="*60)
        
        return current_config
    
    def optimize(self, worker) -> ParticleConfig:
        """
        🔥 Full hybrid optimization pipeline
        """
        print("\n" + "🔥"*30)
        print("HYBRID METAHEURISTIC OPTIMIZATION")
        print("🔥"*30)
        
        # Phase 1: Global search
        pso_best = self.pso_optimize(worker)
        
        # Phase 2: Local refinement
        final_best = self.hill_climbing_refine(pso_best, worker)
        
        # Save optimal config
        self.save_config(final_best)
        
        print("\n" + "🎉"*30)
        print("OPTIMIZATION COMPLETE!")
        print("🎉"*30)
        print(f"\n📊 Optimal Configuration:")
        print(f"   tolerance: {final_best.tolerance:.3f}")
        print(f"   downscale_factor: {final_best.downscale_factor:.3f}")
        print(f"   recognition_interval: {final_best.recognition_interval:.3f}")
        print(f"   num_jitters: {final_best.num_jitters}")
        print(f"   top_k: {final_best.top_k}")
        print(f"\n💾 Saved to: {self.config_file}")
        
        return final_best
    
    def save_config(self, config: ParticleConfig):
        """Save optimal configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        print(f"\n💾 Saved optimal config to {self.config_file}")
    
    def load_config(self) -> Optional[ParticleConfig]:
        """Load optimal configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
            config = ParticleConfig.from_dict(data)
            print(f"✅ Loaded optimal config from {self.config_file}")
            return config
        except FileNotFoundError:
            print(f"⚠️  No config file found at {self.config_file}")
            return None


class RuntimeAdaptiveController:
    """
    🔥 Lightweight runtime adaptation
    Micro-adjustments based on real-time metrics
    """
    
    def __init__(self, worker):
        self.worker = worker
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Thresholds
        self.min_fps = 10
        self.max_false_unknown_rate = 0.3
        self.max_cpu = 85
        
        # Adaptation parameters
        self.check_interval = 10  # Check every 10 seconds
        
        print("🎮 Runtime Adaptive Controller initialized")
    
    def start(self):
        """Start adaptive monitoring"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.thread.start()
        print("🎮 Runtime adaptation started")
    
    def stop(self):
        """Stop adaptive monitoring"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        print("🎮 Runtime adaptation stopped")
    
    def _adaptation_loop(self):
        """Continuous monitoring and micro-adaptation"""
        while self.running:
            time.sleep(self.check_interval)
            
            try:
                status = self.worker.get_status()
                
                # Get metrics
                fps = status.get('capture_fps', 0)
                recognition_fps = status.get('recognition_fps', 0)
                
                # Adapt recognition interval based on FPS
                if fps < self.min_fps:
                    # Stream struggling - reduce recognition load
                    old_interval = self.worker.recognition_interval_sec
                    self.worker.recognition_interval_sec = min(1.0, old_interval + 0.1)
                    print(f"⚡ FPS low ({fps:.1f}) - Increased recognition interval to {self.worker.recognition_interval_sec:.2f}s")
                
                elif fps > 18 and recognition_fps < 1.5:
                    # Stream doing well - can increase recognition
                    old_interval = self.worker.recognition_interval_sec
                    self.worker.recognition_interval_sec = max(0.3, old_interval - 0.05)
                    print(f"⚡ FPS good ({fps:.1f}) - Decreased recognition interval to {self.worker.recognition_interval_sec:.2f}s")
                
                # TODO: Add more adaptive rules
                # - Adjust tolerance based on false unknown rate
                # - Switch between HOG/CNN based on load
                # - Adjust downscale based on CPU
                
            except Exception as e:
                print(f"❌ Adaptation error: {e}")
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        # TODO: Implement real metric collection
        return {
            'fps': self.worker.fps,
            'recognition_fps': self.worker.recognition_fps,
            'false_unknown_rate': 0.0,  # Needs tracking
            'false_match_rate': 0.0,     # Needs tracking
            'cpu_usage': 0.0              # Needs psutil
        }


# Example usage
if __name__ == "__main__":
    # Create optimizer
    optimizer = PSO_HillClimbing_Optimizer()
    
    # Run optimization (pass worker instance)
    # optimal_config = optimizer.optimize(worker)
    
    # Or load existing config
    # config = optimizer.load_config()
    
    print("\n📝 To use in production:")
    print("   1. Run optimizer.optimize(worker) once offline")
    print("   2. Load config with optimizer.load_config()")
    print("   3. Apply to worker parameters")
    print("   4. Enable RuntimeAdaptiveController for micro-tuning")