"""Model registry: pickled weights next to a JSON line of metadata, OOS score and the training data's hash."""
import hashlib
import json
import os
import pickle

import pandas as pd


def data_hash(frame):
    """Fingerprint of a training frame: values, index, columns."""
    h = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    h.update(",".join(map(str, frame.columns)).encode())
    return h.hexdigest()[:16]


class ModelRegistry:
    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.index = os.path.join(path, "index.jsonl")

    def save(self, name, model, train_data, train_window, oos_score, meta=None):
        """Pickle `model`; same name, meta and training data gives the same id and overwrites."""
        meta = meta or {}
        dh = data_hash(train_data)
        body = json.dumps({"name": name, "meta": meta, "data_hash": dh}, sort_keys=True, default=str)
        mid = hashlib.sha256(body.encode()).hexdigest()[:12]
        file = os.path.join(self.path, f"{mid}.pkl")
        with open(file, "wb") as fh:
            pickle.dump(model, fh)
        entry = {"id": mid, "name": name, "meta": meta, "data_hash": dh,
                 "train_window": [str(w) for w in train_window], "n_train_rows": int(len(train_data)),
                 "oos_score": None if oos_score is None else float(oos_score), "file": os.path.basename(file),
                 "saved_at": pd.Timestamp.now().isoformat(timespec="seconds")}
        with open(self.index, "a") as fh:
            fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
        return mid

    def entries(self):
        if not os.path.exists(self.index):
            return pd.DataFrame(columns=["id", "name", "oos_score", "data_hash"])
        rows = [json.loads(line) for line in open(self.index) if line.strip()]
        return pd.DataFrame(rows).drop_duplicates("id", keep="last").reset_index(drop=True)

    def load(self, mid):
        e = self.entries().set_index("id").loc[mid]
        with open(os.path.join(self.path, e["file"]), "rb") as fh:
            return pickle.load(fh), e.to_dict()
