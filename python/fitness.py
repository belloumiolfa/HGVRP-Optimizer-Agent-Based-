from typing import Dict, Optional

DEFAULT_WZ = {"Z1":1.0, "Z2":0.0, "Z3":0.0, "Z4":0.0, "Z5":0.0, "Z6":0.0}
DEFAULT_WP = {"P_capacity":10.0, "P_duration":10.0, "P_timewindows":10.0,
              "P_availability":10.0, "P_battery":10.0, "P_coverage":1000.0, "P_supply":1000.0}

def fitness_weighted(objectives: Dict[str,float], penalties: Dict[str,float],
                     wZ: Optional[Dict[str,float]]=None, wP: Optional[Dict[str,float]]=None) -> float:
    wz = {**DEFAULT_WZ, **(wZ or {})}
    wp = {**DEFAULT_WP, **(wP or {})}
    val = 0.0
    for k, v in objectives.items(): val += wz.get(k, 0.0) * v
    for k, v in penalties.items():  val += wp.get(k, 0.0) * v
    return val

def fitness_lexicographic(objectives: Dict[str,float], penalties: Dict[str,float]) -> float:
    # penalties first; then Z1, Z4, Z5, Z6, Z2, Z3 (encode tuple into one scalar)
    order = [
        sum(penalties.values()),
        objectives.get("Z1",0.0), objectives.get("Z4",0.0),
        objectives.get("Z5",0.0), objectives.get("Z6",0.0),
        objectives.get("Z2",0.0), objectives.get("Z3",0.0),
    ]
    base = 1e6
    acc = 0.0
    for i, v in enumerate(reversed(order)):
        acc += v * (base ** i)
    return acc

def compute_fitness(mode: str,
                    objectives: Dict[str,float],
                    penalties: Dict[str,float],
                    wZ: Optional[Dict[str,float]]=None,
                    wP: Optional[Dict[str,float]]=None) -> float:
    if mode == "lexicographic":
        return fitness_lexicographic(objectives, penalties)
    return fitness_weighted(objectives, penalties, wZ, wP)
