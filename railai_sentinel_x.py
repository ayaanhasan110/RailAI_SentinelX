
"""
RAILAI SENTINEL X
=================
Safety-first adaptive railway intelligence prototype for SIH.

This is a research / decision-support prototype, NOT a certified
railway safety system and NOT AGI.

Design goals
------------
1. Multimodal sensor fusion
2. Weather + track + traffic context
3. Speed, delay and ETA prediction
4. Ensemble learning + champion/challenger validation
5. Adaptive online learning from delayed outcomes
6. Replay memory
7. Concept-drift detection
8. Out-of-distribution detection
9. Uncertainty + conformal-style prediction intervals
10. Deterministic safety gate with abstention
11. Computer-vision hazard observations (optional YOLO)
12. Explainable risk factors
13. Digital-twin simulation and fault injection
14. Persistent model memory and audit log
15. No autonomous railway control

The system may recommend or warn. It must NEVER:
- control signals/interlocking
- issue movement authority
- command brakes/throttle
- change certified speed limits
- override railway protection systems
"""

import os
import json
import time
import math
import random
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque

import numpy as np

try:
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        RandomForestRegressor,
        HistGradientBoostingRegressor,
        IsolationForest,)
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
except ImportError:
    raise SystemExit(
        "\nMissing dependencies.\n"
        "Run:\n"
        "python -m pip install numpy scikit-learn requests\n"
    )

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

STATE_DIR = "railai_sentinel_x_state"
os.makedirs(STATE_DIR, exist_ok=True)

MEMORY_FILE = os.path.join(STATE_DIR, "replay_memory.json")
REGISTRY_FILE = os.path.join(STATE_DIR, "model_registry.json")
AUDIT_FILE = os.path.join(STATE_DIR, "audit_log.jsonl")

# Prototype thresholds. These are NOT Indian Railways operating limits.
MIN_TRUST_FOR_NORMAL_ADVISORY = 0.70
ABSTAIN_TRUST_THRESHOLD = 0.55
HIGH_RISK = 70.0
CRITICAL_RISK = 85.0
MAX_SENSOR_DISAGREEMENT = 15.0
MAX_REASONABLE_SPEED_ERROR = 30.0

FEATURES = [
    "speed_limit",
    "track_quality",
    "track_risk",
    "maintenance_risk",
    "traffic_density",
    "gradient",
    "rain",
    "wind",
    "visibility",
    "temperature",
    "humidity",
    "actual_speed",
    "acceleration",
    "jerk",
    "current_delay",
    "remaining_distance_km",
    "remaining_stops",
    "station_congestion",
    "hour_sin",
    "hour_cos",
    "animal_risk",
    "obstacle_risk",
    "sensor_confidence",
    "sensor_disagreement",
    "speed_rolling_mean",
    "speed_rolling_std",
    "speed_trend",
    "weather_stress",
    "track_stress",
    "traffic_stress",
]

ROUTES = {
    "LUCKNOW_BARABANKI": {
        "speed_limit": 110.0,
        "track_quality": 0.94,
        "track_risk": 0.08,
        "maintenance_risk": 0.12,
        "traffic_density": 0.48,
        "gradient": 0.04,
    },
    "BARABANKI_AYODHYA": {
        "speed_limit": 100.0,
        "track_quality": 0.91,
        "track_risk": 0.13,
        "maintenance_risk": 0.18,
        "traffic_density": 0.52,
        "gradient": 0.08,
    },
    "AYODHYA_SULTANPUR": {
        "speed_limit": 95.0,
        "track_quality": 0.88,
        "track_risk": 0.20,
        "maintenance_risk": 0.23,
        "traffic_density": 0.42,
        "gradient": 0.12,
    },
    "LUCKNOW_KANPUR": {
        "speed_limit": 120.0,
        "track_quality": 0.96,
        "track_risk": 0.06,
        "maintenance_risk": 0.08,
        "traffic_density": 0.72,
        "gradient": 0.03,
    },
}


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))


def now_iso():
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def sha256_obj(obj):
    raw = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


# ---------------------------------------------------------------------
# DATA OBJECTS
# ---------------------------------------------------------------------

@dataclass
class TrainState:
    train_id: str
    route_id: str
    latitude: float
    longitude: float
    scheduled_speed: float
    current_delay: float
    remaining_distance_km: float
    remaining_stops: int
    station_congestion: float
    previous_speed: float
    update_seconds: float = 10.0


@dataclass
class SafetyDecision:
    mode: str
    trust_score: float
    risk_score: float
    reasons: list
    allowed_actions: list
    forbidden_actions: list


# ---------------------------------------------------------------------
# WEATHER PROVIDER
# ---------------------------------------------------------------------

class WeatherProvider:
    """Live Open-Meteo when available, otherwise deterministic simulation."""

    def get(self, lat, lon, forced=None):
        if forced is not None:
            return dict(forced)

        try:
            import requests

            url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                "&current=temperature_2m,"
                "relative_humidity_2m,"
                "precipitation,"
                "rain,"
                "wind_speed_10m,"
                "visibility"
            )

            response = requests.get(url, timeout=3)
            response.raise_for_status()
            current = response.json()["current"]

            return {
                "temperature": safe_float(
                    current.get("temperature_2m"), 28
                ),
                "humidity": safe_float(
                    current.get("relative_humidity_2m"), 60
                ),
                "rain": safe_float(
                    current.get("rain"), 0
                ),
                "wind": safe_float(
                    current.get("wind_speed_10m"), 10
                ),
                "visibility": safe_float(
                    current.get("visibility"), 10000
                ),
                "fresh": True,
                "source": "live_open_meteo",
            }
        except Exception:
            return {
                "temperature": random.uniform(22, 35),
                "humidity": random.uniform(45, 90),
                "rain": max(0, np.random.normal(1.5, 2.5)),
                "wind": max(0, np.random.normal(12, 7)),
                "visibility": max(1500, np.random.normal(9000, 1800)),
                "fresh": False,
                "source": "simulation",
            }


# ---------------------------------------------------------------------
# SENSOR FUSION
# ---------------------------------------------------------------------

class SensorFusion:
    """
    Prototype fusion layer.

    A real system needs time synchronization, authenticated telemetry,
    calibration, redundant channels and railway-certified interfaces.
    """

    def fuse_speed(self, gps, track=None, wheel=None, vision=None):
        values = [safe_float(gps)]
        weights = [0.90]

        for value, weight in [
            (track, 0.98),
            (wheel, 0.99),
            (vision, 0.70),
        ]:
            if value is not None:
                values.append(safe_float(value))
                weights.append(weight)

        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)

        fused = float(np.average(values, weights=weights))
        disagreement = float(values.max() - values.min())

        confidence = 100.0 * math.exp(
            -disagreement / 12.0
        )
        confidence = clamp(confidence, 0, 100)

        return fused, confidence, disagreement


# ---------------------------------------------------------------------
# DRIFT DETECTOR
# ---------------------------------------------------------------------

class PageHinkley:
    def __init__(self, delta=0.005, threshold=7.0):
        self.delta = delta
        self.threshold = threshold
        self.n = 0
        self.mean = 0.0
        self.sum = 0.0
        self.minimum = 0.0

    def update(self, error):
        x = abs(float(error))
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.sum += x - self.mean - self.delta
        self.minimum = min(self.minimum, self.sum)

        drift = (self.sum - self.minimum) > self.threshold
        if drift:
            self.sum = 0.0
            self.minimum = 0.0
        return drift


# ---------------------------------------------------------------------
# OOD DETECTOR
# ---------------------------------------------------------------------

class OODDetector:
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=160,
            contamination=0.03,
            random_state=SEED,
        )
        self.fitted = False

    def fit(self, X):
        if len(X) >= 100:
            self.model.fit(X)
            self.fitted = True

    def score(self, X):
        if not self.fitted:
            return False, 0.0

        row = np.asarray(X).reshape(1, -1)
        raw = float(self.model.decision_function(row)[0])
        label = int(self.model.predict(row)[0])
        return label == -1, raw


# ---------------------------------------------------------------------
# ADAPTIVE ENSEMBLE
# ---------------------------------------------------------------------

class AdaptiveEnsemble:
    """
    Three-model ensemble:
      1. ExtraTrees
      2. RandomForest
      3. HistGradientBoosting
      4. SGD online challenger

    Models are selected by validation error and ensemble disagreement.
    The system learns prediction behavior, not safety rules.
    """

    def __init__(self, name):
        self.name = name

        self.X_memory = deque(maxlen=6000)
        self.y_memory = deque(maxlen=6000)

        self.models = [
            ExtraTreesRegressor(
                n_estimators=130,
                max_depth=18,
                min_samples_leaf=2,
                random_state=SEED,
                n_jobs=-1,
            ),
            RandomForestRegressor(
                n_estimators=110,
                max_depth=16,
                min_samples_leaf=2,
                random_state=SEED,
                n_jobs=-1,
            ),
            HistGradientBoostingRegressor(
                max_iter=160,
                learning_rate=0.06,
                max_leaf_nodes=25,
                l2_regularization=0.2,
                random_state=SEED,
            ),
            SGDRegressor(
                loss="huber",
                penalty="elasticnet",
                alpha=0.0005,
                learning_rate="adaptive",
                eta0=0.01,
                random_state=SEED,
            ),
        ]

        self.scaler = StandardScaler()
        self.fitted = [False] * 4
        self.weights = np.array([0.32, 0.28, 0.25, 0.15])
        self.validation_mae = [999.0] * 4

        self.drift = PageHinkley()
        self.drift_events = 0
        self.update_count = 0
        self.last_retrain_size = 0

        # Conformal-style nonconformity residual memory.
        self.residuals = deque(maxlen=1500)

    def add(self, X, y):
        self.X_memory.append(np.asarray(X, dtype=float))
        self.y_memory.append(float(y))

    def train(self, min_samples=120):
        if len(self.y_memory) < min_samples:
            return False

        X = np.asarray(self.X_memory, dtype=float)
        y = np.asarray(self.y_memory, dtype=float)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.20, random_state=SEED
        )

        self.scaler.fit(X_train)
        A = self.scaler.transform(X_train)
        B = self.scaler.transform(X_val)

        errors = []

        # Batch models
        for i in range(3):
            self.models[i].fit(A, y_train)
            pred = self.models[i].predict(B)
            mae = mean_absolute_error(y_val, pred)
            errors.append(float(mae))
            self.fitted[i] = True

        # Online model
        self.models[3].partial_fit(A, y_train)
        pred = self.models[3].predict(B)
        mae = mean_absolute_error(y_val, pred)
        errors.append(float(mae))
        self.fitted[3] = True

        self.validation_mae = errors

        inv = 1.0 / (np.asarray(errors) + 1e-5)
        inv /= inv.sum()

        # Slow adaptation prevents one noisy validation split from
        # completely changing the ensemble.
        self.weights = 0.70 * self.weights + 0.30 * inv
        self.weights /= self.weights.sum()

        # Validation residuals for uncertainty interval.
        ensemble_val = np.zeros(len(B))
        for i in range(4):
            ensemble_val += self.weights[i] * self.models[i].predict(B)

        self.residuals.clear()
        for residual in np.abs(y_val - ensemble_val):
            self.residuals.append(float(residual))

        self.update_count += 1
        self.last_retrain_size = len(self.y_memory)
        return True

    def predict(self, X):
        if not any(self.fitted):
            return None

        row = np.asarray(X, dtype=float).reshape(1, -1)
        Z = self.scaler.transform(row)

        predictions = np.full(4, np.nan)
        for i, model in enumerate(self.models):
            if self.fitted[i]:
                predictions[i] = float(model.predict(Z)[0])

        valid = np.isfinite(predictions)
        weights = self.weights.copy()
        weights[~valid] = 0
        if weights.sum() <= 0:
            return None
        weights /= weights.sum()

        ensemble = float(np.dot(weights, predictions))
        variance = float(
            np.dot(weights, (predictions - ensemble) ** 2)
        )
        disagreement = math.sqrt(max(variance, 0.0))

        # 90% conformal-style interval based on held-out residuals.
        if len(self.residuals) >= 30:
            q = float(np.quantile(self.residuals, 0.90))
        else:
            q = max(5.0, 2.0 * disagreement)

        lower = ensemble - q
        upper = ensemble + q

        # Trust is a conservative score, not a probability of correctness.
        uncertainty_penalty = math.exp(
            -(disagreement + 0.5 * q) / 15.0
        )
        best_error = min(self.validation_mae)
        validation_factor = math.exp(
            -best_error / 20.0
        )

        trust = clamp(
            0.55 * uncertainty_penalty
            + 0.45 * validation_factor,
            0,
            1,
        )

        return {
            "prediction": ensemble,
            "lower": lower,
            "upper": upper,
            "uncertainty": disagreement,
            "interval_half_width": q,
            "trust": trust,
            "individual_predictions": predictions.tolist(),
            "weights": weights.tolist(),
            "validation_mae": self.validation_mae,
        }

    def learn(self, X, actual, predicted):
        error = float(actual - predicted)
        drift = self.drift.update(error)

        if drift:
            self.drift_events += 1

        self.add(X, actual)

        # Every 50 new examples, retrain from replay memory.
        if (
            len(self.y_memory) - self.last_retrain_size >= 50
            and len(self.y_memory) >= 120
        ):
            self.train()

        return {
            "error": error,
            "absolute_error": abs(error),
            "drift": drift,
            "drift_events": self.drift_events,
        }


# ---------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------

class FeatureEngine:
    def __init__(self):
        self.speed_history = deque(maxlen=12)

    def update_history(self, speed):
        self.speed_history.append(float(speed))

    def build(
        self,
        train,
        route,
        weather,
        speed,
        acceleration,
        animal_risk,
        obstacle_risk,
        sensor_confidence,
        disagreement,
    ):
        self.update_history(speed)

        hist = np.asarray(self.speed_history, dtype=float)

        rolling_mean = float(hist.mean()) if len(hist) else speed
        rolling_std = (
            float(hist.std()) if len(hist) > 1 else 0.0
        )

        if len(hist) >= 3:
            trend = float(
                np.polyfit(
                    np.arange(len(hist)),
                    hist,
                    1,
                )[0]
            )
        else:
            trend = 0.0

        hour = datetime.now().hour
        theta = 2 * math.pi * hour / 24

        weather_stress = clamp(
            min(weather["rain"] / 15, 1)
            + max(0, 1 - weather["visibility"] / 5000)
            + max(0, (weather["wind"] - 35) / 40),
            0,
            3,
        )

        track_stress = clamp(
            (1 - route["track_quality"])
            + route["track_risk"]
            + route["maintenance_risk"],
            0,
            1.5,
        )

        traffic_stress = clamp(
            0.55 * route["traffic_density"]
            + 0.45 * train.station_congestion,
            0,
            1,
        )

        return np.array([
            route["speed_limit"],
            route["track_quality"],
            route["track_risk"],
            route["maintenance_risk"],
            route["traffic_density"],
            route["gradient"],
            weather["rain"],
            weather["wind"],
            weather["visibility"],
            weather["temperature"],
            weather["humidity"],
            speed,
            acceleration,
            0.0,  # jerk is injected by caller when available
            train.current_delay,
            train.remaining_distance_km,
            train.remaining_stops,
            train.station_congestion,
            math.sin(theta),
            math.cos(theta),
            animal_risk,
            obstacle_risk,
            sensor_confidence / 100,
            disagreement,
            rolling_mean,
            rolling_std,
            trend,
            weather_stress,
            track_stress,
            traffic_stress,
        ], dtype=float)


# ---------------------------------------------------------------------
# VISION
# ---------------------------------------------------------------------

ANIMAL_CLASSES = {
    "cow": 0.80,
    "horse": 0.70,
    "sheep": 0.60,
    "dog": 0.40,
    "elephant": 1.00,
    "bird": 0.15,
}

OBSTACLE_CLASSES = {
    "person": 0.90,
    "car": 0.90,
    "truck": 1.00,
    "bus": 1.00,
    "motorcycle": 0.70,
    "bicycle": 0.60,
}

class VisionEngine:
    def __init__(self):
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolo11n.pt")
            print("[VISION] YOLO loaded.")
        except Exception:
            print(
                "[VISION] Optional YOLO unavailable. "
                "Simulation still works."
            )

    def detect(self, frame):
        if self.model is None:
            return []

        results = self.model(frame, verbose=False)
        output = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                output.append({
                    "label": result.names[cls],
                    "confidence": conf,
                    "bbox": box.xyxy[0].tolist(),
                })

        return output


def detections_to_risk(detections):
    animal = 0.0
    obstacle = 0.0

    for d in detections:
        label = d["label"]
        conf = safe_float(d["confidence"])

        animal = max(
            animal,
            ANIMAL_CLASSES.get(label, 0) * conf,
        )

        obstacle = max(
            obstacle,
            OBSTACLE_CLASSES.get(label, 0) * conf,
        )

    return animal, obstacle


# ---------------------------------------------------------------------
# RISK ENGINE
# ---------------------------------------------------------------------

class RiskEngine:
    """
    Deterministic risk scoring.

    AI cannot rewrite these rules during runtime.
    """

    def evaluate(
        self,
        route,
        weather,
        animal_risk,
        obstacle_risk,
        sensor_confidence,
        disagreement,
        ood,
        trust,
        speed_error,
        anomaly,
    ):
        score = 0.0
        reasons = []

        if animal_risk >= 0.75:
            score += 70
            reasons.append("high-confidence animal observation")
        elif animal_risk >= 0.40:
            score += 30
            reasons.append("possible animal hazard")

        if obstacle_risk >= 0.75:
            score += 75
            reasons.append("high-confidence obstruction observation")
        elif obstacle_risk >= 0.40:
            score += 35
            reasons.append("possible obstruction")

        if disagreement > MAX_SENSOR_DISAGREEMENT:
            score += 35
            reasons.append("sensor disagreement")

        if sensor_confidence < 60:
            score += 25
            reasons.append("low sensor confidence")

        if ood:
            score += 35
            reasons.append("out-of-distribution telemetry")

        if trust < ABSTAIN_TRUST_THRESHOLD:
            score += 30
            reasons.append("prediction trust below safety threshold")

        if abs(speed_error) > MAX_REASONABLE_SPEED_ERROR:
            score += 20
            reasons.append("large speed prediction deviation")

        if weather["visibility"] < 2000:
            score += 15
            reasons.append("poor visibility")

        if weather["rain"] > 10:
            score += 12
            reasons.append("heavy rain")

        if route["track_risk"] > 0.18:
            score += 12
            reasons.append("elevated track risk")

        if anomaly:
            score += 20
            reasons.append("telemetry anomaly")

        score = clamp(score, 0, 100)

        if score >= CRITICAL_RISK:
            level = "CRITICAL"
        elif score >= HIGH_RISK:
            level = "HIGH"
        elif score >= 40:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "score": score,
            "level": level,
            "reasons": reasons,
        }


# ---------------------------------------------------------------------
# SAFETY GATE
# ---------------------------------------------------------------------

class SafetyGate:
    """
    Independent deterministic gate.

    The model may be adaptive.
    This layer is intentionally NOT adaptive.
    """

    FORBIDDEN = [
        "issue movement authority",
        "change signal state",
        "change interlocking state",
        "command braking",
        "command throttle",
        "change certified speed limits",
        "override railway protection systems",
        "automatically clear a safety alarm",
    ]

    def decide(
        self,
        trust,
        risk,
        ood,
        sensor_confidence,
        disagreement,
        weather_fresh,
    ):
        reasons = []

        if trust < ABSTAIN_TRUST_THRESHOLD:
            reasons.append("low prediction trust")

        if ood:
            reasons.append("input outside learned distribution")

        if sensor_confidence < 60:
            reasons.append("low sensor confidence")

        if disagreement > MAX_SENSOR_DISAGREEMENT:
            reasons.append("excessive sensor disagreement")

        if risk["score"] >= CRITICAL_RISK:
            reasons.append("critical risk score")

        if not weather_fresh:
            reasons.append("weather source not fresh")

        if reasons:
            mode = "ABSTAIN_HUMAN_REVIEW"
            actions = [
                "display warning",
                "preserve evidence",
                "request authorized human assessment",
                "continue monitoring",
            ]
        elif risk["score"] >= HIGH_RISK:
            mode = "HIGH_RISK_ADVISORY"
            actions = [
                "display high-risk advisory",
                "continue monitoring",
                "request authorized human assessment",
            ]
        else:
            mode = "ADVISORY"
            actions = [
                "display prediction",
                "continue monitoring",
            ]

        return SafetyDecision(
            mode=mode,
            trust_score=trust,
            risk_score=risk["score"],
            reasons=reasons,
            allowed_actions=actions,
            forbidden_actions=self.FORBIDDEN,
        )


# ---------------------------------------------------------------------
# EXPLANATION
# ---------------------------------------------------------------------

class ExplanationEngine:
    @staticmethod
    def explain(
        route,
        weather,
        prediction,
        uncertainty,
        delay,
        animal_risk,
        obstacle_risk,
    ):
        factors = []

        if weather["rain"] > 2:
            factors.append("rain affected operating-condition estimate")
        if weather["wind"] > 30:
            factors.append("high wind affected operating-condition estimate")
        if weather["visibility"] < 5000:
            factors.append("reduced visibility increased caution")
        if route["traffic_density"] > 0.60:
            factors.append("traffic density increased expected delay")
        if route["track_risk"] > 0.15:
            factors.append("track-risk features increased caution")
        if uncertainty > 10:
            factors.append("ensemble disagreement increased uncertainty")
        if delay > 10:
            factors.append("delay model predicts elevated delay")
        if animal_risk >= 0.40:
            factors.append("animal observation increased risk")
        if obstacle_risk >= 0.40:
            factors.append("obstruction observation increased risk")

        return factors or ["no dominant adverse factor detected"]


# ---------------------------------------------------------------------
# MODEL REGISTRY
# ---------------------------------------------------------------------

class ModelRegistry:
    """
    Lightweight champion/challenger registry.

    A challenger is promoted only if its validation MAE improves
    over the champion by a configurable margin.
    """

    def __init__(self):
        self.data = {
            "speed_champion_mae": None,
            "delay_champion_mae": None,
            "promotion_count": 0,
            "last_update": None,
        }
        self.load()

    def load(self):
        if not os.path.exists(REGISTRY_FILE):
            return
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                self.data.update(json.load(f))
        except Exception:
            pass

    def save(self):
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def validate_and_record(self, speed_mae, delay_mae):
        changed = False

        old_speed = self.data["speed_champion_mae"]
        old_delay = self.data["delay_champion_mae"]

        if old_speed is None or speed_mae < old_speed * 0.995:
            self.data["speed_champion_mae"] = speed_mae
            changed = True

        if old_delay is None or delay_mae < old_delay * 0.995:
            self.data["delay_champion_mae"] = delay_mae
            changed = True

        if changed:
            self.data["promotion_count"] += 1
            self.data["last_update"] = now_iso()
            self.save()

        return changed


# ---------------------------------------------------------------------
# MAIN ENGINE
# ---------------------------------------------------------------------

class RailAISentinelX:
    def __init__(self):
        self.weather = WeatherProvider()
        self.sensor = SensorFusion()
        self.features = FeatureEngine()
        self.ood = OODDetector()

        self.speed_model = AdaptiveEnsemble("speed")
        self.delay_model = AdaptiveEnsemble("delay")

        self.risk = RiskEngine()
        self.safety = SafetyGate()
        self.explainer = ExplanationEngine()
        self.registry = ModelRegistry()
        self.vision = VisionEngine()

        self.last_inputs = {}
        self.last_result = {}

        self.load_memory()

    # -----------------------------------------------------------------
    # SYNTHETIC DIGITAL-TWIN BOOTSTRAP
    # -----------------------------------------------------------------

    def bootstrap(self, samples=3200):
        """
        Creates a synthetic prior ONLY so the prototype can run offline.

        Replace this with real historical railway data before making
        real-world accuracy claims.
        """
        if len(self.speed_model.y_memory) >= 500:
            print("[AI] Existing replay memory found.")
            return

        print("[AI] Building synthetic digital-twin prior...")

        all_X = []

        for _ in range(samples):
            route_id = random.choice(list(ROUTES))
            route = ROUTES[route_id]

            train = TrainState(
                "BOOT",
                route_id,
                26.85,
                80.95,
                random.uniform(70, route["speed_limit"]),
                max(0, np.random.normal(8, 5)),
                random.uniform(5, 450),
                random.randint(0, 10),
                random.random(),
                random.uniform(45, route["speed_limit"]),
            )

            weather = {
                "temperature": random.uniform(15, 40),
                "humidity": random.uniform(30, 95),
                "rain": max(0, np.random.normal(2, 4)),
                "wind": max(0, np.random.normal(12, 10)),
                "visibility": max(
                    1000, np.random.normal(9000, 2500)
                ),
                "fresh": True,
                "source": "synthetic",
            }

            speed = random.uniform(35, route["speed_limit"])
            acceleration = np.random.normal(0, 0.7)
            animal = random.random() * 0.04
            obstacle = random.random() * 0.04
            sensor_conf = random.uniform(0.90, 1.0)
            disagreement = random.uniform(0, 4)

            X = self.features.build(
                train,
                route,
                weather,
                speed,
                acceleration,
                animal,
                obstacle,
                sensor_conf * 100,
                disagreement,
            )

            # Synthetic "physics + operations" relationship.
            weather_factor = clamp(
                1
                - min(weather["rain"], 20) * 0.006
                - max(weather["wind"] - 30, 0) * 0.001
                - max(5000 - weather["visibility"], 0) / 100000,
                0.60,
                1.0,
            )

            target_speed = (
                route["speed_limit"]
                * weather_factor
                * (0.88 - 0.18 * route["traffic_density"])
                * route["track_quality"]
            )

            target_speed += np.random.normal(0, 2.5)
            target_speed = clamp(
                target_speed,
                20,
                route["speed_limit"],
            )

            target_delay = max(
                0,
                train.current_delay * 0.55
                + 14 * route["traffic_density"]
                + 12 * route["maintenance_risk"]
                + 1.5 * train.remaining_stops
                + max(0, 2 - weather["visibility"] / 5000) * 8
                + np.random.normal(0, 1.8),
            )

            self.speed_model.add(X, target_speed)
            self.delay_model.add(X, target_delay)
            all_X.append(X)

        self.ood.fit(np.asarray(all_X))

        self.speed_model.train()
        self.delay_model.train()

        self.registry.validate_and_record(
            min(self.speed_model.validation_mae),
            min(self.delay_model.validation_mae),
        )

        self.save_memory()

        print("[AI] Bootstrap complete.")

    # -----------------------------------------------------------------
    # PROCESS LIVE OBSERVATION
    # -----------------------------------------------------------------

    def process(
        self,
        train,
        gps_speed,
        track_speed=None,
        wheel_speed=None,
        vision_speed=None,
        detections=None,
        forced_weather=None,
        forced_track=None,
    ):
        route = dict(ROUTES[train.route_id])

        if forced_track:
            route.update(forced_track)

        weather = self.weather.get(
            train.latitude,
            train.longitude,
            forced=forced_weather,
        )

        fused_speed, sensor_conf, disagreement = (
            self.sensor.fuse_speed(
                gps_speed,
                track_speed,
                wheel_speed,
                vision_speed,
            )
        )

        acceleration = (
            fused_speed - train.previous_speed
        ) / max(train.update_seconds, 1)

        animal_risk = 0.0
        obstacle_risk = 0.0

        if detections:
            animal_risk, obstacle_risk = detections_to_risk(
                detections
            )

        X = self.features.build(
            train,
            route,
            weather,
            fused_speed,
            acceleration,
            animal_risk,
            obstacle_risk,
            sensor_conf,
            disagreement,
        )

        # Safety-critical feature: OOD check happens before trusting AI.
        ood, ood_score = self.ood.score(X)

        speed_pred = self.speed_model.predict(X)
        delay_pred = self.delay_model.predict(X)

        if speed_pred is None:
            speed_pred = {
                "prediction": fused_speed,
                "lower": fused_speed - 999,
                "upper": fused_speed + 999,
                "uncertainty": 999,
                "interval_half_width": 999,
                "trust": 0.0,
                "individual_predictions": [],
                "weights": [],
                "validation_mae": [],
            }

        if delay_pred is None:
            delay_pred = {
                "prediction": train.current_delay,
                "lower": train.current_delay - 999,
                "upper": train.current_delay + 999,
                "uncertainty": 999,
                "interval_half_width": 999,
                "trust": 0.0,
                "individual_predictions": [],
                "weights": [],
                "validation_mae": [],
            }

        predicted_speed = clamp(
            speed_pred["prediction"],
            10,
            route["speed_limit"],
        )

        # Weather is used to estimate operating impact, NOT to alter
        # certified railway speed limits.
        weather_factor = clamp(
            1
            - min(weather["rain"], 20) * 0.005
            - max(weather["wind"] - 30, 0) * 0.001
            - max(5000 - weather["visibility"], 0) / 100000,
            0.65,
            1.0,
        )

        advisory_speed = clamp(
            predicted_speed * weather_factor,
            10,
            route["speed_limit"],
        )

        predicted_delay = max(
            0,
            delay_pred["prediction"],
        )

        model_trust = min(
            speed_pred["trust"],
            delay_pred["trust"],
        )

        speed_error = fused_speed - advisory_speed
        anomaly = (
            abs(speed_error) > MAX_REASONABLE_SPEED_ERROR
            or disagreement > MAX_SENSOR_DISAGREEMENT
        )

        risk = self.risk.evaluate(
            route=route,
            weather=weather,
            animal_risk=animal_risk,
            obstacle_risk=obstacle_risk,
            sensor_confidence=sensor_conf,
            disagreement=disagreement,
            ood=ood,
            trust=model_trust,
            speed_error=speed_error,
            anomaly=anomaly,
        )

        safety = self.safety.decide(
            trust=model_trust,
            risk=risk,
            ood=ood,
            sensor_confidence=sensor_conf,
            disagreement=disagreement,
            weather_fresh=weather["fresh"],
        )

        distance = max(train.remaining_distance_km, 0.1)

        travel_minutes = (
            distance / max(advisory_speed, 10) * 60
        )

        eta_minutes = (
            travel_minutes + predicted_delay
        )

        arrival = datetime.now() + timedelta(
            minutes=eta_minutes
        )

        explanation = self.explainer.explain(
            route,
            weather,
            advisory_speed,
            speed_pred["uncertainty"],
            predicted_delay,
            animal_risk,
            obstacle_risk,
        )

        result = {
            "timestamp": now_iso(),
            "system": "RailAI Sentinel X",
            "train_id": train.train_id,
            "route": train.route_id,

            "observed_speed": fused_speed,
            "predicted_speed": predicted_speed,
            "weather_adjusted_advisory_speed": advisory_speed,
            "speed_interval": [
                speed_pred["lower"],
                speed_pred["upper"],
            ],
            "speed_uncertainty": speed_pred["uncertainty"],
            "speed_trust": speed_pred["trust"],

            "predicted_delay_min": predicted_delay,
            "delay_interval": [
                delay_pred["lower"],
                delay_pred["upper"],
            ],
            "delay_uncertainty": delay_pred["uncertainty"],
            "delay_trust": delay_pred["trust"],

            "eta_minutes": eta_minutes,
            "predicted_arrival": arrival.isoformat(
                timespec="seconds"
            ),

            "weather": weather,
            "weather_factor": weather_factor,

            "sensor_confidence": sensor_conf,
            "sensor_disagreement": disagreement,

            "animal_risk": animal_risk,
            "obstacle_risk": obstacle_risk,

            "ood": {
                "is_ood": ood,
                "score": ood_score,
            },

            "risk": risk,
            "safety_decision": asdict(safety),
            "explanation": explanation,

            "ensemble": {
                "speed_models": speed_pred["individual_predictions"],
                "speed_weights": speed_pred["weights"],
                "delay_models": delay_pred["individual_predictions"],
                "delay_weights": delay_pred["weights"],
            },

            "model_metrics": {
                "speed_validation_mae": speed_pred["validation_mae"],
                "delay_validation_mae": delay_pred["validation_mae"],
                "speed_drift_events": self.speed_model.drift_events,
                "delay_drift_events": self.delay_model.drift_events,
            },

            "safety_note": (
                "Prediction-only prototype. "
                "No railway control action is issued."
            ),
        }

        self.last_inputs[train.train_id] = X
        self.last_result[train.train_id] = result

        self.audit("PREDICTION", result)

        return result

    # -----------------------------------------------------------------
    # DELAYED OUTCOME / ADAPTIVE LEARNING
    # -----------------------------------------------------------------

    def learn_from_outcome(
        self,
        train_id,
        actual_speed,
        actual_delay,
    ):
        if train_id not in self.last_inputs:
            return {"status": "NO_MATCHING_PREDICTION"}

        X = self.last_inputs[train_id]

        speed_prediction = self.speed_model.predict(X)
        delay_prediction = self.delay_model.predict(X)

        predicted_speed = (
            speed_prediction["prediction"]
            if speed_prediction
            else actual_speed
        )

        predicted_delay = (
            delay_prediction["prediction"]
            if delay_prediction
            else actual_delay
        )

        speed_update = self.speed_model.learn(
            X,
            actual_speed,
            predicted_speed,
        )

        delay_update = self.delay_model.learn(
            X,
            actual_delay,
            predicted_delay,
        )

        # Re-fit OOD detector periodically on replay memory.
        if (
            len(self.speed_model.X_memory) >= 200
            and (
                self.speed_model.update_count % 2 == 0
            )
        ):
            X_mem = np.asarray(
                list(self.speed_model.X_memory)[-3000:]
            )
            self.ood.fit(X_mem)

        # Registry records validated champion metrics.
        if (
            self.speed_model.validation_mae
            and self.delay_model.validation_mae
        ):
            self.registry.validate_and_record(
                min(self.speed_model.validation_mae),
                min(self.delay_model.validation_mae),
            )

        self.save_memory()

        result = {
            "timestamp": now_iso(),
            "train_id": train_id,
            "speed": speed_update,
            "delay": delay_update,
            "adaptive_memory_size": len(
                self.speed_model.y_memory
            ),
            "message": (
                "Prediction models updated from observed outcome. "
                "Deterministic safety gate unchanged."
            ),
        }

        self.audit("LEARNING_UPDATE", result)
        return result

    # -----------------------------------------------------------------
    # PERSISTENCE
    # -----------------------------------------------------------------

    def save_memory(self):
        payload = {
            "speed_X": [
                x.tolist()
                for x in self.speed_model.X_memory
            ],
            "speed_y": list(self.speed_model.y_memory),
            "delay_X": [
                x.tolist()
                for x in self.delay_model.X_memory
            ],
            "delay_y": list(self.delay_model.y_memory),
            "saved_at": now_iso(),
        }

        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load_memory(self):
        if not os.path.exists(MEMORY_FILE):
            return

        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)

            for X, y in zip(
                payload.get("speed_X", []),
                payload.get("speed_y", []),
            ):
                self.speed_model.add(X, y)

            for X, y in zip(
                payload.get("delay_X", []),
                payload.get("delay_y", []),
            ):
                self.delay_model.add(X, y)

            print(
                f"[MEMORY] Loaded "
                f"{len(self.speed_model.y_memory)} speed samples."
            )

        except Exception as exc:
            print("[MEMORY] Load failed:", exc)

    # -----------------------------------------------------------------
    # AUDIT
    # -----------------------------------------------------------------

    def audit(self, event, payload):
        record = {
            "timestamp": now_iso(),
            "event": event,
            "payload_hash": sha256_obj(payload),
            "payload": payload,
        }

        with open(AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    record,
                    default=str,
                ) + "\n"
            )


# ---------------------------------------------------------------------
# DIGITAL TWIN SCENARIO GENERATOR
# ---------------------------------------------------------------------

class DigitalTwin:
    """
    Generates controlled scenarios for testing.

    This is not a substitute for real railway telemetry.
    """

    def __init__(self):
        self.train = TrainState(
            train_id="SIH-DEMO-001",
            route_id="LUCKNOW_BARABANKI",
            latitude=26.8467,
            longitude=80.9462,
            scheduled_speed=105,
            current_delay=4,
            remaining_distance_km=120,
            remaining_stops=4,
            station_congestion=0.45,
            previous_speed=75,
        )

    def scenario(self, cycle):
        if cycle <= 10:
            weather = {
                "temperature": 28,
                "humidity": 55,
                "rain": 0.0,
                "wind": 12,
                "visibility": 12000,
                "fresh": True,
                "source": "digital_twin",
            }
            gps = random.uniform(82, 98)
            track = gps + random.uniform(-2, 2)
            detections = []

        elif cycle <= 20:
            weather = {
                "temperature": 26,
                "humidity": 78,
                "rain": random.uniform(3, 9),
                "wind": random.uniform(18, 30),
                "visibility": random.uniform(4000, 8000),
                "fresh": True,
                "source": "digital_twin",
            }
            gps = random.uniform(60, 82)
            track = gps + random.uniform(-3, 3)
            self.train.current_delay += random.uniform(0.4, 1.8)
            detections = []

        elif cycle <= 24:
            # Sensor fault injection.
            weather = {
                "temperature": 25,
                "humidity": 85,
                "rain": 12,
                "wind": 38,
                "visibility": 1800,
                "fresh": True,
                "source": "digital_twin",
            }
            gps = random.uniform(55, 70)
            track = gps + random.uniform(28, 45)
            self.train.current_delay += random.uniform(1, 3)
            detections = []

        elif cycle <= 27:
            # Animal / obstruction event.
            weather = {
                "temperature": 24,
                "humidity": 88,
                "rain": 9,
                "wind": 34,
                "visibility": 2500,
                "fresh": True,
                "source": "digital_twin",
            }
            gps = random.uniform(45, 62)
            track = gps + random.uniform(-2, 3)
            detections = [
                {
                    "label": "cow",
                    "confidence": 0.92,
                    "bbox": [320, 180, 500, 400],
                }
            ]

        else:
            # Post-event recovery / regime shift.
            weather = {
                "temperature": 29,
                "humidity": 65,
                "rain": 0.5,
                "wind": 16,
                "visibility": 9000,
                "fresh": True,
                "source": "digital_twin",
            }
            gps = random.uniform(70, 90)
            track = gps + random.uniform(-2, 2)
            detections = []

        return {
            "gps": gps,
            "track": track,
            "weather": weather,
            "detections": detections,
        }

    def apply_outcome(self, observed_speed, predicted_speed):
        # Synthetic ground truth with realistic measurement noise.
        actual_speed = (
            observed_speed
            + random.uniform(-1.5, 1.5)
        )

        actual_delay = max(
            0,
            self.train.current_delay
            + random.uniform(-0.8, 2.0)
        )

        self.train.previous_speed = actual_speed

        self.train.remaining_distance_km = max(
            0,
            self.train.remaining_distance_km
            - actual_speed
            * self.train.update_seconds
            / 3600,
        )

        return actual_speed, actual_delay


# ---------------------------------------------------------------------
# TERMINAL DISPLAY
# ---------------------------------------------------------------------

def print_result(result, cycle=None):
    print("\n" + "=" * 82)
    if cycle:
        print(f"LIVE CYCLE {cycle}")
    else:
        print("LIVE PREDICTION")

    print("=" * 82)

    print(
        f"Observed speed          : "
        f"{result['observed_speed']:.1f} km/h"
    )

    print(
        f"AI predicted speed      : "
        f"{result['predicted_speed']:.1f} km/h"
    )

    print(
        f"Advisory speed estimate : "
        f"{result['weather_adjusted_advisory_speed']:.1f} km/h"
    )

    lo, hi = result["speed_interval"]

    print(
        f"Prediction interval     : "
        f"{lo:.1f} to {hi:.1f} km/h"
    )

    print(
        f"Speed uncertainty       : "
        f"{result['speed_uncertainty']:.2f}"
    )

    print(
        f"Speed trust             : "
        f"{result['speed_trust']:.2f}"
    )

    print(
        f"Predicted delay         : "
        f"{result['predicted_delay_min']:.1f} min"
    )

    print(
        f"ETA                     : "
        f"{result['eta_minutes']:.1f} min"
    )

    print(
        f"Predicted arrival       : "
        f"{result['predicted_arrival']}"
    )

    print(
        f"Sensor confidence       : "
        f"{result['sensor_confidence']:.1f}%"
    )

    print(
        f"Sensor disagreement     : "
        f"{result['sensor_disagreement']:.1f} km/h"
    )

    print(
        f"OOD / unusual input     : "
        f"{result['ood']['is_ood']}"
    )

    print(
        f"Animal risk             : "
        f"{result['animal_risk']:.2f}"
    )

    print(
        f"Obstacle risk           : "
        f"{result['obstacle_risk']:.2f}"
    )

    print(
        f"Risk                    : "
        f"{result['risk']['level']} "
        f"({result['risk']['score']:.0f}/100)"
    )

    print(
        f"SAFETY MODE             : "
        f"{result['safety_decision']['mode']}"
    )

    if result["safety_decision"]["reasons"]:
        print(
            "Safety reasons          : "
            + "; ".join(
                result["safety_decision"]["reasons"]
            )
        )

    print(
        "AI explanation          : "
        + "; ".join(result["explanation"])
    )


# ---------------------------------------------------------------------
# SIMULATION
# ---------------------------------------------------------------------

def run_simulation():
    print(
        """
===============================================================
RAILAI SENTINEL X
Adaptive Multimodal Railway Intelligence
===============================================================

SAFE MODE:
Prediction + warning + human-review only.
No signal/brake/throttle/movement-authority control.

"""
    )

    ai = RailAISentinelX()
    ai.bootstrap()

    twin = DigitalTwin()

    for cycle in range(1, 41):
        inputs = twin.scenario(cycle)

        result = ai.process(
            twin.train,
            gps_speed=inputs["gps"],
            track_speed=inputs["track"],
            detections=inputs["detections"],
            forced_weather=inputs["weather"],
        )

        print_result(result, cycle)

        actual_speed, actual_delay = twin.apply_outcome(
            inputs["gps"],
            result["predicted_speed"],
        )

        learning = ai.learn_from_outcome(
            twin.train.train_id,
            actual_speed,
            actual_delay,
        )

        print(
            "\nADAPTIVE LEARNING"
        )

        print(
            f"Speed error             : "
            f"{learning['speed']['error']:+.2f} km/h"
        )

        print(
            f"Delay error             : "
            f"{learning['delay']['error']:+.2f} min"
        )

        if learning["speed"]["drift"]:
            print(
                "!!! SPEED CONCEPT DRIFT DETECTED"
            )

        if learning["delay"]["drift"]:
            print(
                "!!! DELAY CONCEPT DRIFT DETECTED"
            )

        print(
            f"Replay memory           : "
            f"{learning['adaptive_memory_size']} samples"
        )

        time.sleep(0.15)

    print("\nSIMULATION COMPLETE.")
    print(f"Memory : {MEMORY_FILE}")
    print(f"Audit  : {AUDIT_FILE}")
    print(f"Registry: {REGISTRY_FILE}")


# ---------------------------------------------------------------------
# MANUAL SAFETY TESTS
# ---------------------------------------------------------------------

def run_safety_tests():
    print(
        """
===============================================================
RAILAI SENTINEL X - SAFETY TEST SUITE
===============================================================
These are controlled software tests, not railway certification tests.
"""
    )

    ai = RailAISentinelX()
    ai.bootstrap()

    train = TrainState(
        "SAFETY-TEST",
        "LUCKNOW_BARABANKI",
        26.85,
        80.95,
        105,
        3,
        80,
        2,
        0.4,
        75,
    )

    cases = [
        (
            "NORMAL",
            88,
            89,
            [],
            {
                "temperature": 28,
                "humidity": 55,
                "rain": 0,
                "wind": 10,
                "visibility": 12000,
                "fresh": True,
                "source": "test",
            },
        ),
        (
            "SENSOR DISAGREEMENT",
            85,
            120,
            [],
            {
                "temperature": 28,
                "humidity": 55,
                "rain": 0,
                "wind": 10,
                "visibility": 12000,
                "fresh": True,
                "source": "test",
            },
        ),
        (
            "ANIMAL ON TRACK",
            60,
            61,
            [
                {
                    "label": "cow",
                    "confidence": 0.96,
                    "bbox": [100, 100, 300, 350],
                }
            ],
            {
                "temperature": 25,
                "humidity": 80,
                "rain": 7,
                "wind": 25,
                "visibility": 3500,
                "fresh": True,
                "source": "test",
            },
        ),
        (
            "STALE WEATHER",
            75,
            76,
            [],
            {
                "temperature": 30,
                "humidity": 80,
                "rain": 5,
                "wind": 20,
                "visibility": 5000,
                "fresh": False,
                "source": "stale_test",
            },
        ),
    ]

    for name, gps, track, detections, weather in cases:
        result = ai.process(
            train,
            gps_speed=gps,
            track_speed=track,
            detections=detections,
            forced_weather=weather,
        )

        print(
            f"\n{name}"
        )
        print(
            f"Risk  : {result['risk']['level']} "
            f"({result['risk']['score']:.0f})"
        )
        print(
            f"Mode  : "
            f"{result['safety_decision']['mode']}"
        )
        print(
            f"Trust : "
            f"{min(result['speed_trust'], result['delay_trust']):.2f}"
        )

    print(
        "\nSafety test suite complete."
    )


# ---------------------------------------------------------------------
# CAMERA MODE
# ---------------------------------------------------------------------

def run_camera():
    try:
        import cv2
    except ImportError:
        print(
            "Install OpenCV first:\n"
            "python -m pip install opencv-python"
        )
        return

    ai = RailAISentinelX()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print(
            "Could not open camera."
        )
        return

    print(
        "Camera mode started. Press Q to quit."
    )

    while True:
        ok, frame = camera.read()

        if not ok:
            break

        detections = ai.vision.detect(frame)

        animal, obstacle = detections_to_risk(
            detections
        )

        for d in detections:
            x1, y1, x2, y2 = [
                int(v) for v in d["bbox"]
            ]

            label = (
                f"{d['label']} "
                f"{d['confidence']:.2f}"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if animal >= 0.75:
            cv2.putText(
                frame,
                "ANIMAL HAZARD - HUMAN REVIEW",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                3,
            )

        if obstacle >= 0.75:
            cv2.putText(
                frame,
                "OBSTRUCTION - HUMAN REVIEW",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                3,
            )
        cv2.imshow(
            "RailAI Sentinel X - Advisory Vision",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            print("Stopping camera...")
            break

    camera.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------
# MAIN MENU
# ---------------------------------------------------------------------

def main():
    print(
        """

RAILAI SENTINEL X

1. Digital-twin adaptive simulation
2. Safety test suite
3. Live camera hazard observation
4. Exit

"""
    )

    choice = input("Select: ").strip()

    if choice == "1":
        run_simulation()
    elif choice == "2":
        run_safety_tests()
    elif choice == "3":
        run_camera()
    else:
        print("Exit.")



if __name__ == "__main__":
    main()