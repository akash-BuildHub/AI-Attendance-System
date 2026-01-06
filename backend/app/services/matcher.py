import numpy as np

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))

def match_embedding(emb: np.ndarray, prototypes: dict, threshold: float):
    """
    prototypes: {person_name: prototype_embedding}
    returns: (label, score)
    """
    if not prototypes:
        return "Unknown", 0.0

    best_name = "Unknown"
    best_score = -1.0

    for name, p in prototypes.items():
        s = cosine_sim(emb, p)
        if s > best_score:
            best_score = s
            best_name = name

    if best_score >= threshold:
        return best_name, best_score
    return "Unknown", best_score
