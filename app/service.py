# app/service.py
import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from unidecode import unidecode

# -------------------------------
# Paths
# -------------------------------
MODEL_DIR   = os.getenv("MODEL_DIR", "model/prod")
ENCODER_PKL = os.path.join(MODEL_DIR, "prenom_encoder.pkl")
ENCODER_CSV = os.path.join(MODEL_DIR, "data_dpt_encode.csv")
PRIOR_JSON  = os.path.join(MODEL_DIR, "prior_mean.json")

# -------------------------------
# Helpers
# -------------------------------
def normalize_prename(s: str) -> str:
    # EXACTLY match training-time normalization
    return unidecode(str(s)).upper().strip()

# ----- SavedModel (Keras 3 export) wrapper -----
def _load_savedmodel_wrapped(path: str):
    sm  = tf.saved_model.load(path)
    sig = sm.signatures.get("serving_default") or next(iter(sm.signatures.values()))
    in_keys  = list(sig.structured_input_signature[1].keys());  assert in_keys,  "No inputs in SavedModel signature"
    out_keys = list(sig.structured_outputs.keys());             assert out_keys, "No outputs in SavedModel signature"
    in_key, out_key = in_keys[0], out_keys[0]

    class SavedModelWrapper:
        def __init__(self, signature, ik, ok):
            self._sig, self._ik, self._ok = signature, ik, ok
        def predict(self, x, verbose=0):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            out = self._sig(**{self._ik: x})
            return out[self._ok].numpy()
    print(f"[MODEL] Wrapped SavedModel (sig in={in_key}, out={out_key}) from: {path}")
    return SavedModelWrapper(sig, in_key, out_key)

def _find_and_load_model()-> object:
    # Look for SavedModel in exported_model subfolder
    saved_dir = os.path.join(MODEL_DIR, "exported_model")
    if os.path.isdir(saved_dir) and os.path.isfile(os.path.join(saved_dir, "saved_model.pb")):
        return _load_savedmodel_wrapped(saved_dir)

    import glob
    candidates = glob.glob(os.path.join(MODEL_DIR, "**", "saved_model.pb"), recursive=True)
    if candidates:
        discovered = os.path.dirname(candidates[0])
        print(f"[MODEL] Found SavedModel in subfolder: {discovered}")
        return _load_savedmodel_wrapped(discovered)

    keras_path = os.path.join(MODEL_DIR, "gender_model.keras")
    if os.path.isfile(keras_path):
        import keras
        m = keras.saving.load_model(keras_path)
        print(f"[MODEL] Loaded .keras via keras.saving from: {keras_path}")
        return m

    h5_path = os.path.join(MODEL_DIR, "gender_model.h5")
    if os.path.isfile(h5_path):
        m = tf.keras.models.load_model(h5_path)
        print(f"[MODEL] Loaded .h5 via tf.keras from: {h5_path}")
        return m

    raise FileNotFoundError(f"No model artifact found in {MODEL_DIR}")

# ----- Encoder that prefers CSV+JSON, normalizes index -----
class NameEncoder:
    """
    Runtime encoder backed by a (name -> float) mapping and a prior mean.
    Prefers CSV+JSON (portable), falls back to PKL only if needed.
    """
    def __init__(self):
        self._prior = 0.5
        self._map   = None  # dict: NORMALIZED_NAME -> float

        if os.path.isfile(ENCODER_CSV):
            self._map, self._prior = self._load_csv_encoder(ENCODER_CSV, PRIOR_JSON)
            print(f"[ENCODER] CSV mapping loaded: {len(self._map)} names; prior={self._prior:.4f}")
        elif os.path.isfile(ENCODER_PKL):
            self._map, self._prior = self._load_pkl_encoder(ENCODER_PKL)
            print(f"[ENCODER] PKL mapping loaded: {len(self._map)} names; prior={self._prior:.4f}")
        else:
            raise FileNotFoundError(
                f"No encoder artifacts found in {MODEL_DIR}. "
                f"Expected {ENCODER_CSV} (+ {PRIOR_JSON}) or {ENCODER_PKL}."
            )

    @staticmethod
    def _load_csv_encoder(csv_path: str, prior_json: str):
        df = pd.read_csv(csv_path)

        # Heuristic to pick columns robustly across CSV variants
        # If the CSV came from ser.to_frame('te_value').to_csv(...), columns are like: ['Unnamed: 0','te_value']
        # Otherwise first column is name, last numeric column is value.
        # Identify name column:
        if df.shape[1] >= 2 and "te_value" in df.columns:
            # Common case from our export
            name_col = [c for c in df.columns if c != "te_value"][0]
            val_col  = "te_value"
        else:
            name_col = df.columns[0]
            # pick the rightmost numeric column as value
            numeric_candidates = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
            val_col = numeric_candidates[-1] if numeric_candidates else df.columns[-1]

        names = df[name_col].astype(str).map(normalize_prename)
        vals  = pd.to_numeric(df[val_col], errors="coerce")
        ser   = pd.Series(vals.values, index=names).dropna()
        # Collapse duplicates if any by mean to get a single value per name
        ser   = ser.groupby(level=0).mean()

        prior = 0.5
        if os.path.isfile(prior_json):
            try:
                with open(prior_json, "r") as f:
                    prior = float(json.load(f).get("prior_mean", 0.5))
            except Exception:
                pass
        else:
            # fallback: average of mapping
            if len(ser):
                prior = float(ser.mean())

        # Convert to plain dict for fast lookup and to avoid pandas edge cases
        mapping = {k: float(v) for k, v in ser.items()}
        return mapping, prior

    @staticmethod
    def _load_pkl_encoder(pkl_path: str):
        with open(pkl_path, "rb") as f:
            enc = pickle.load(f)
        m = getattr(enc, "mapping", None)
        if m is None:
            raise ValueError("Pickled encoder has no .mapping")

        # Extract pandas Series across category_encoders variants
        ser = None
        if isinstance(m, list):
            for e in m:
                mm = e.get("mapping") if isinstance(e, dict) else None
                if isinstance(mm, pd.Series):
                    ser = mm; break
        elif isinstance(m, dict):
            for v in m.values():
                if isinstance(v, pd.Series):
                    ser = v; break
                if isinstance(v, dict) and isinstance(v.get("mapping"), pd.Series):
                    ser = v["mapping"]; break
        if ser is None:
            raise ValueError("Could not find Series mapping inside pickled encoder")

        ser.index = ser.index.map(normalize_prename)
        ser = pd.to_numeric(ser, errors="coerce").dropna()
        ser = ser.groupby(level=0).mean()
        mapping = {k: float(v) for k, v in ser.items()}

        prior = 0.5
        for attr in ["_mean", "_prior_mean", "prior_mean", "global_mean", "overall_mean"]:
            if hasattr(enc, attr):
                try:
                    prior = float(getattr(enc, attr)); break
                except Exception:
                    pass
        if not np.isfinite(prior):
            prior = float(ser.mean()) if len(ser) else 0.5
        return mapping, prior

    def lookup(self, raw_name: str):
        """Return (value, found_flag)."""
        key = normalize_prename(raw_name)
        v = self._map.get(key)
        if v is None or not np.isfinite(v):
            return self._prior, False
        return float(v), True


# Singletons
name_encoder = NameEncoder()
model = _find_and_load_model()

def _to_feature(name: str):
    te, found = name_encoder.lookup(name)
    x = np.array([[te]], dtype="float32")  # model expects shape (None,1)
    return x, te, found

# -------------------------------
# Public service
# -------------------------------
class Genderservice:
    def __init__(self, threshold: float = 0.5):
        self.model = model
        self.threshold = float(threshold)

    def predict_proba(self, name: str) -> dict:
        if not (name and str(name).strip()):
            return {"prob_f": 0.0, "prob_m": 1.0, "dist_f": None, "dist_m": None, "from_mapping": False}

        x, te_val, found = _to_feature(name)
        y = self.model.predict(x, verbose=0)
        prob_f = float(y[0][0])
        prob_m = 1.0 - prob_f

        # encoder distribution (only if found)
        if found:
            dist_f = float(min(1.0, max(0.0, te_val - 1.0)))
            dist_m = float(1.0 - dist_f)
        else:
            dist_f = None
            dist_m = None

        print(f"[ENCODER] name='{name}' -> key='{normalize_prename(name)}' "
              f"found={found} te={te_val:.4f} prob_f={prob_f:.4f}")

        return {"prob_f": prob_f, "prob_m": prob_m, "dist_f": dist_f, "dist_m": dist_m, "from_mapping": found}

    def predict_label(self, name: str) -> dict:
        res = self.predict_proba(name)
        label = "F" if res["prob_f"] >= self.threshold else "M"
        return {"input": name, "gender": label, **res, "threshold": self.threshold}

# export singleton
service = Genderservice()
