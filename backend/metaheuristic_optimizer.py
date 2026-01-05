"""
🏆 SHADE → L-SHADE → Local Refinement (NelderMead-like)
Security-first parameter optimizer for CCTV face recognition.

- Keeps ParticleConfig + optimal_config.json compatibility with your current system.
- Uses Success-History based Differential Evolution (SHADE),
  then Linear population size reduction (L-SHADE),
  then a small simplex-style local refine (no SciPy).

NOTE:
For ArcFace pipeline, num_jitters and top_k are effectively unused by recognition,
but we keep them for backward compatibility and future toggles.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np


# ----------------------------
# Config dataclass (kept compatible)
# ----------------------------
@dataclass
class ParticleConfig:
    tolerance: float
    downscale_factor: float
    recognition_interval: float
    num_jitters: int
    top_k: int

    def to_dict(self) -> Dict:
        return {
            "tolerance": float(self.tolerance),
            "downscale_factor": float(self.downscale_factor),
            "recognition_interval": float(self.recognition_interval),
            "num_jitters": int(self.num_jitters),
            "top_k": int(self.top_k),
        }

    @staticmethod
    def from_dict(d: Dict) -> "ParticleConfig":
        return ParticleConfig(
            tolerance=float(d["tolerance"]),
            downscale_factor=float(d["downscale_factor"]),
            recognition_interval=float(d["recognition_interval"]),
            num_jitters=int(d["num_jitters"]),
            top_k=int(d["top_k"]),
        )


# ----------------------------
# SHADE / L-SHADE Optimizer
# ----------------------------
class SHADE_LSHADE_SecurityOptimizer:
    """
    SHADE (global) → L-SHADE (linear pop reduction) → Local simplex refinement

    Security weights prioritize:
    - false_match: biggest penalty
    - false_unknown: next biggest penalty
    """

    def __init__(
        self,
        config_file: str = "optimal_config.json",
        seed: Optional[int] = None,
    ):
        self.config_file = config_file
        self.rng = np.random.default_rng(seed)

        # Function evaluation budget
        self.max_fes = 1000

        # SHADE memory size
        self.H = 10

        # Starting/ending population for L-SHADE
        self.n_pop_init = 20
        self.n_pop_min = 6

        # SECURITY fitness weights
        self.weights = {
            "false_unknown": -5.0,
            "false_match": -10.0,   # SECURITY FIRST (as requested)
            "accuracy": 4.0,
            "fps": 1.5,
            "cpu": -0.8,
        }

        # Parameter bounds (CCTV tuned) (tolerance is ArcFace cosine threshold)
        # [min, max] per param in vector order:
        # tolerance, downscale, interval, jitters, topk
        self.bounds = np.array(
            [
                [0.42, 0.58],
                [0.28, 0.42],
                [0.35, 0.85],
                [0.0, 2.0],  # keep numeric; cast to int later
                [1.0, 8.0],  # keep numeric; cast to int later
            ],
            dtype=np.float64,
        )

        # DE control parameter ranges
        self.F_min, self.F_max = 0.1, 1.0
        self.CR_min, self.CR_max = 0.0, 1.0

        # Archive size multiplier (standard in SHADE family)
        self.archive_rate = 1.4

        # Local refinement budget fraction
        self.local_frac = 0.20  # last 20% of budget

        print("🔒 SHADE-LSHADE Security Optimizer initialized")
        print(f"   max_fes={self.max_fes}, H={self.H}, pop={self.n_pop_init}→{self.n_pop_min}")
        print(f"   weights={self.weights}")

    # ----------------------------
    # Public API
    # ----------------------------
    def optimize(self, worker=None) -> ParticleConfig:
        """
        3 phases:
        - Phase 1: SHADE (50% budget)
        - Phase 2: L-SHADE (30% budget, pop reduction)
        - Phase 3: Local refine (20% budget)
        """
        print("\n" + "=" * 70)
        print("🔒 SHADE → L-SHADE → Local Refinement (Security-First)")
        print("=" * 70)

        # budgets
        fes_phase1 = int(self.max_fes * 0.50)
        fes_phase2 = int(self.max_fes * 0.30)
        fes_phase3 = self.max_fes - (fes_phase1 + fes_phase2)

        # init population
        pop = self._initialize_population(self.n_pop_init)
        fitness = np.array([self.evaluate_security_fitness(ind, worker) for ind in pop], dtype=np.float64)
        fes_used = len(pop)

        best_idx = int(np.argmax(fitness))
        best_vec = pop[best_idx].copy()
        best_fit = float(fitness[best_idx])

        # SHADE memories
        M_F = np.full(self.H, 0.5, dtype=np.float64)
        M_CR = np.full(self.H, 0.5, dtype=np.float64)
        mem_idx = 0

        # Archive
        archive = np.empty((0, pop.shape[1]), dtype=np.float64)

        # ---------------- Phase 1: SHADE ----------------
        print(f"\nPhase 1: SHADE (Global) | budget={fes_phase1} FEs")
        pop, fitness, best_vec, best_fit, M_F, M_CR, mem_idx, archive, fes_used = self._shade_run(
            pop, fitness, best_vec, best_fit, M_F, M_CR, mem_idx, archive,
            worker=worker,
            fes_target=fes_used + fes_phase1
        )

        # ---------------- Phase 2: L-SHADE ----------------
        print(f"\nPhase 2: L-SHADE (Linear pop reduction) | budget={fes_phase2} FEs")
        pop, fitness, best_vec, best_fit, M_F, M_CR, mem_idx, archive, fes_used = self._lshade_run(
            pop, fitness, best_vec, best_fit, M_F, M_CR, mem_idx, archive,
            worker=worker,
            fes_target=fes_used + fes_phase2,
        )

        # ---------------- Phase 3: Local refine ----------------
        print(f"\nPhase 3: Local refinement (simplex-style) | budget={fes_phase3} FEs")
        best_vec, best_fit, fes_used = self._local_refine(
            best_vec, best_fit, worker, fes_used, fes_used + fes_phase3
        )

        best_cfg = self._vector_to_config(best_vec)
        self.save_config(best_cfg)

        print("\n" + "=" * 70)
        print(f"🎯 SECURITY OPTIMAL: fitness={best_fit:.6f}")
        print(f"   tolerance={best_cfg.tolerance:.4f} | downscale={best_cfg.downscale_factor:.4f} | "
              f"interval={best_cfg.recognition_interval:.4f} | jitters={best_cfg.num_jitters} | top_k={best_cfg.top_k}")
        print(f"💾 Saved to: {self.config_file}")
        print("=" * 70 + "\n")

        return best_cfg

    def save_config(self, config: ParticleConfig) -> None:
        with open(self.config_file, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

    def load_config(self) -> Optional[ParticleConfig]:
        try:
            with open(self.config_file, "r") as f:
                return ParticleConfig.from_dict(json.load(f))
        except FileNotFoundError:
            return None

    # ----------------------------
    # Fitness
    # ----------------------------
    def evaluate_security_fitness(self, config_vector: np.ndarray, worker=None) -> float:
        """
        "False match = instant death penalty" style objective.

        If you later add real metrics from worker (fps/cpu/false_match/false_unknown),
        plug them here. For now we use a CCTV-biased simulation fallback.
        """
        metrics = self._simulate_cctv_metrics(config_vector)

        fitness = (
            self.weights["accuracy"] * metrics["accuracy"]
            + self.weights["false_unknown"] * metrics["false_unknown"]
            + self.weights["false_match"] * metrics["false_match"]
            + self.weights["fps"] * min(metrics["fps"] / 25.0, 1.0)
            + self.weights["cpu"] * (metrics["cpu"] / 100.0)
        )

        # Extra hard penalty if false_match is high (security wall)
        if metrics["false_match"] > 0.03:
            fitness -= 5.0 * (metrics["false_match"] - 0.03) * 100.0

        return float(fitness)

    def _simulate_cctv_metrics(self, x: np.ndarray) -> Dict[str, float]:
        """
        Deterministic-ish simulation that prefers:
        - tolerance around ~0.47-0.50 (ArcFace sweet spot)
        - reasonable downscale ~0.33-0.38
        - reasonable interval ~0.55-0.70

        This is a placeholder until you wire real measurements.
        """
        tol, ds, interval, jit, topk = x

        # "accuracy" peak around 0.485
        accuracy = 0.93 - abs(tol - 0.485) * 0.9
        accuracy -= abs(ds - 0.35) * 0.35
        accuracy -= abs(interval - 0.62) * 0.20
        accuracy = float(np.clip(accuracy, 0.65, 0.97))

        # false_unknown decreases as tolerance becomes slightly more lenient, but too lenient hurts ID quality
        false_unknown = 0.020 + abs(tol - 0.48) * 0.22 + abs(interval - 0.62) * 0.05
        false_unknown = float(np.clip(false_unknown, 0.005, 0.12))

        # false_match increases when tolerance is too low (too lenient matching)
        false_match = max(0.0, (0.46 - tol) * 0.35)
        false_match += abs(tol - 0.485) * 0.05
        false_match = float(np.clip(false_match, 0.0, 0.20))

        # fps/cpu heuristic
        fps = 26.0 - (1.0 - ds) * 22.0 - (0.75 - interval) * 8.0
        fps = float(np.clip(fps, 6.0, 25.0))

        cpu = 35.0 + (1.0 - ds) * 55.0 + max(0.0, (0.55 - interval)) * 40.0
        cpu = float(np.clip(cpu, 15.0, 100.0))

        return {
            "accuracy": accuracy,
            "false_unknown": false_unknown,
            "false_match": false_match,
            "fps": fps,
            "cpu": cpu,
        }

    # ----------------------------
    # Core SHADE / L-SHADE
    # ----------------------------
    def _initialize_population(self, n: int) -> np.ndarray:
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        pop = self.rng.uniform(lo, hi, size=(n, len(lo)))
        return pop

    def _shade_run(
        self,
        pop: np.ndarray,
        fit: np.ndarray,
        best_vec: np.ndarray,
        best_fit: float,
        M_F: np.ndarray,
        M_CR: np.ndarray,
        mem_idx: int,
        archive: np.ndarray,
        worker,
        fes_target: int,
    ):
        n, d = pop.shape
        fes_used = int(len(pop))  # already counted initial eval externally

        while fes_used < fes_target:
            # sort by fitness descending for p-best selection
            order = np.argsort(-fit)
            pop_sorted = pop[order]
            fit_sorted = fit[order]

            p_best_rate = 0.2
            p_num = max(2, int(np.ceil(p_best_rate * n)))

            new_pop = pop.copy()
            new_fit = fit.copy()

            # success memories
            S_F, S_CR, delta_f = [], [], []

            for i in range(n):
                # choose memory slot
                r = self.rng.integers(0, self.H)
                mu_F = M_F[r]
                mu_CR = M_CR[r]

                # sample F ~ Cauchy(mu_F, 0.1), resample until (0,1]
                F = self._sample_cauchy_positive(mu_F, 0.1)
                F = float(np.clip(F, self.F_min, self.F_max))

                # sample CR ~ Normal(mu_CR, 0.1), clipped
                CR = float(np.clip(self.rng.normal(mu_CR, 0.1), self.CR_min, self.CR_max))

                xi = pop[i]

                # p-best individual
                pbest = pop_sorted[self.rng.integers(0, p_num)]

                # mutation base vectors from union(pop, archive)
                union = pop if archive.size == 0 else np.vstack([pop, archive])
                r1 = self._rand_index_excluding(len(pop), exclude=i)
                x_r1 = pop[r1]

                r2 = self._rand_index_excluding(len(union), exclude=None)
                x_r2 = union[r2]

                # current-to-pbest/1
                vi = xi + F * (pbest - xi) + F * (x_r1 - x_r2)

                # binomial crossover
                ui = self._binomial_crossover(xi, vi, CR)

                # bounds handling
                ui = self._clip_to_bounds(ui)

                # evaluate
                ui_fit = self.evaluate_security_fitness(ui, worker)
                fes_used += 1
                if fes_used >= fes_target and fes_target > 0:
                    # allow last eval to count then break later
                    pass

                # selection
                if ui_fit >= fit[i]:
                    # success: push parent into archive
                    archive = self._archive_push(archive, xi)

                    new_pop[i] = ui
                    new_fit[i] = ui_fit

                    # record success
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(ui_fit - fit[i])

                    # update best
                    if ui_fit > best_fit:
                        best_fit = float(ui_fit)
                        best_vec = ui.copy()
                # else keep

                if fes_used >= fes_target:
                    break

            pop, fit = new_pop, new_fit
            n = pop.shape[0]

            # limit archive size
            max_arch = int(np.floor(self.archive_rate * n))
            if archive.shape[0] > max_arch:
                idx = self.rng.choice(archive.shape[0], size=max_arch, replace=False)
                archive = archive[idx]

            # update memories
            if len(S_F) > 0:
                mem_idx = self._update_memory(M_F, M_CR, mem_idx, S_F, S_CR, delta_f)

            # progress prints (light)
            if (fes_used % 150) < n:
                print(f"   FEs={fes_used}/{fes_target} | best_fit={best_fit:.6f}")

        return pop, fit, best_vec, best_fit, M_F, M_CR, mem_idx, archive, fes_used

    def _lshade_run(
        self,
        pop: np.ndarray,
        fit: np.ndarray,
        best_vec: np.ndarray,
        best_fit: float,
        M_F: np.ndarray,
        M_CR: np.ndarray,
        mem_idx: int,
        archive: np.ndarray,
        worker,
        fes_target: int,
    ):
        # Linear pop reduction from current n to n_pop_min over this phase
        fes_start = 0
        n0 = pop.shape[0]

        # We don’t know absolute global fes here, so reduce by proportion of progress in this phase
        # We'll approximate using loops and shrink periodically.
        while fes_start < (fes_target):
            # run a chunk of SHADE steps
            chunk = 120
            pop, fit, best_vec, best_fit, M_F, M_CR, mem_idx, archive, fes_used = self._shade_run(
                pop, fit, best_vec, best_fit, M_F, M_CR, mem_idx, archive,
                worker=worker,
                fes_target=min(fes_target, (fes_start + chunk))
            )
            fes_start = fes_used

            # desired pop size by linear schedule
            progress = min(1.0, fes_start / max(1.0, float(fes_target)))
            n_target = int(round(n0 - progress * (n0 - self.n_pop_min)))
            n_target = max(self.n_pop_min, n_target)

            if pop.shape[0] > n_target:
                # keep best individuals
                order = np.argsort(-fit)
                keep = order[:n_target]
                pop = pop[keep]
                fit = fit[keep]

                # also trim archive accordingly
                max_arch = int(np.floor(self.archive_rate * pop.shape[0]))
                if archive.shape[0] > max_arch:
                    idx = self.rng.choice(archive.shape[0], size=max_arch, replace=False)
                    archive = archive[idx]

                print(f"   📉 L-SHADE pop reduced to {pop.shape[0]}")

            if fes_start >= fes_target:
                break

        return pop, fit, best_vec, best_fit, M_F, M_CR, mem_idx, archive, fes_start

    # ----------------------------
    # Local refinement (simplex-style, no SciPy)
    # ----------------------------
    def _local_refine(self, x_best: np.ndarray, f_best: float, worker, fes_used: int, fes_target: int):
        d = x_best.shape[0]
        # build simplex around best
        simplex = [x_best.copy()]
        for k in range(d):
            step = (self.bounds[k, 1] - self.bounds[k, 0]) * 0.03
            x = x_best.copy()
            x[k] = x[k] + step
            simplex.append(self._clip_to_bounds(x))
        simplex = np.array(simplex, dtype=np.float64)

        fvals = np.array([self.evaluate_security_fitness(x, worker) for x in simplex], dtype=np.float64)
        fes_used += len(simplex)

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        while fes_used < fes_target:
            # sort
            order = np.argsort(-fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            if fvals[0] > f_best:
                f_best = float(fvals[0])
                x_best = simplex[0].copy()

            # centroid of best d points (exclude worst)
            centroid = np.mean(simplex[:-1], axis=0)

            # reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = self._clip_to_bounds(x_r)
            f_r = self.evaluate_security_fitness(x_r, worker)
            fes_used += 1

            if f_r >= fvals[0]:
                # expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = self._clip_to_bounds(x_e)
                f_e = self.evaluate_security_fitness(x_e, worker)
                fes_used += 1
                if f_e > f_r:
                    simplex[-1], fvals[-1] = x_e, f_e
                else:
                    simplex[-1], fvals[-1] = x_r, f_r

            elif f_r >= fvals[-2]:
                simplex[-1], fvals[-1] = x_r, f_r
            else:
                # contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = self._clip_to_bounds(x_c)
                f_c = self.evaluate_security_fitness(x_c, worker)
                fes_used += 1

                if f_c > fvals[-1]:
                    simplex[-1], fvals[-1] = x_c, f_c
                else:
                    # shrink
                    best = simplex[0].copy()
                    for i in range(1, simplex.shape[0]):
                        simplex[i] = self._clip_to_bounds(best + sigma * (simplex[i] - best))
                        fvals[i] = self.evaluate_security_fitness(simplex[i], worker)
                    fes_used += (simplex.shape[0] - 1)

            # light progress
            if (fes_used % 120) < 3:
                print(f"   Local FEs={fes_used}/{fes_target} | best_fit={f_best:.6f}")

            if fes_used >= fes_target:
                break

        return x_best, f_best, fes_used

    # ----------------------------
    # Helpers
    # ----------------------------
    def _vector_to_config(self, v: np.ndarray) -> ParticleConfig:
        v = self._clip_to_bounds(v)

        tol = float(v[0])
        ds = float(v[1])
        interval = float(v[2])

        # ints
        jit = int(np.clip(int(round(v[3])), int(self.bounds[3, 0]), int(self.bounds[3, 1])))
        topk = int(np.clip(int(round(v[4])), int(self.bounds[4, 0]), int(self.bounds[4, 1])))

        return ParticleConfig(
            tolerance=round(tol, 4),
            downscale_factor=round(ds, 4),
            recognition_interval=round(interval, 4),
            num_jitters=jit,
            top_k=topk,
        )

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def _binomial_crossover(self, x: np.ndarray, v: np.ndarray, CR: float) -> np.ndarray:
        d = x.shape[0]
        jrand = self.rng.integers(0, d)
        mask = self.rng.random(d) < CR
        mask[jrand] = True
        u = np.where(mask, v, x)
        return u

    def _rand_index_excluding(self, n: int, exclude: Optional[int]) -> int:
        if exclude is None:
            return int(self.rng.integers(0, n))
        j = int(self.rng.integers(0, n - 1))
        if j >= exclude:
            j += 1
        return j

    def _archive_push(self, archive: np.ndarray, x: np.ndarray) -> np.ndarray:
        if archive.size == 0:
            return x.reshape(1, -1)
        return np.vstack([archive, x.reshape(1, -1)])

    def _sample_cauchy_positive(self, loc: float, scale: float) -> float:
        # Resample until > 0
        for _ in range(50):
            val = loc + scale * np.tan(np.pi * (self.rng.random() - 0.5))
            if val > 0:
                return float(val)
        return float(max(1e-6, loc))

    def _update_memory(
        self,
        M_F: np.ndarray,
        M_CR: np.ndarray,
        mem_idx: int,
        S_F: List[float],
        S_CR: List[float],
        delta_f: List[float],
    ) -> int:
        # weights proportional to improvement
        w = np.array(delta_f, dtype=np.float64)
        w = w / (np.sum(w) + 1e-12)

        S_F = np.array(S_F, dtype=np.float64)
        S_CR = np.array(S_CR, dtype=np.float64)

        # Lehmer mean for F
        num = np.sum(w * (S_F ** 2))
        den = np.sum(w * S_F) + 1e-12
        new_MF = num / den

        # weighted mean for CR
        new_MCR = float(np.sum(w * S_CR))

        M_F[mem_idx] = float(np.clip(new_MF, self.F_min, self.F_max))
        M_CR[mem_idx] = float(np.clip(new_MCR, self.CR_min, self.CR_max))

        mem_idx = (mem_idx + 1) % self.H
        return mem_idx
