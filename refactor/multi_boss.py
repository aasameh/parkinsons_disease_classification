import hashlib
import pickle
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from pyts.transformation import BOSS
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


if not hasattr(spmatrix, "A"):
    spmatrix.A = property(lambda self: self.toarray())  # type: ignore[attr-defined]


class MultiBOSS(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(
        self,
        data_shape=None,
        window_sizes=(40,),
        window_step=2,
        word_size=2,
        n_bins=2,
        cache_dir="./",
        verbose=False,
    ):
        self.classes_ = [0, 1]
        self.data_shape = data_shape
        self.window_sizes = window_sizes
        self.window_step = window_step
        self.word_size = word_size
        self.n_bins = n_bins
        self.cache_dir = cache_dir
        self.verbose = verbose

        if self.data_shape is None:
            raise ValueError("data_shape is required")
        self.n_channels = int(self.data_shape[0])

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.boss_list = []
        for _ in range(self.n_channels):
            for window_size in self.window_sizes:
                self.boss_list.append(
                    BOSS(
                        sparse=False,
                        window_size=window_size,
                        window_step=self.window_step,
                        word_size=self.word_size,
                        n_bins=self.n_bins,
                        norm_std=False,
                        norm_mean=False,
                        anova=False,
                        drop_sum=False,
                    )
                )

    def _hash_path(self, x: np.ndarray):
        payload = x.tobytes()
        digest = hashlib.md5(payload).hexdigest()
        digest += "_" + "_".join([str(w) for w in self.window_sizes])
        digest += f"_{self.word_size}_{self.n_bins}"
        return str(Path(self.cache_dir) / f"{digest}.pkl")

    def fit(self, x, y=None, **kwargs):
        cache_path = self._hash_path(x)

        if Path(cache_path).is_file():
            if self.verbose:
                print(f"[MultiBOSS] Cache hit: {cache_path}", flush=True)
            with open(cache_path, "rb") as f:
                self.boss_list = pickle.load(f)
            return self

        if self.verbose:
            print(f"[MultiBOSS] Cache miss. Fitting BOSS models...", flush=True)

        tasks = []
        i = 0
        for ch in range(self.n_channels):
            for _ in self.window_sizes:
                tasks.append((self.boss_list[i], x[:, ch, :]))
                i += 1

        def _do_fit(boss_obj, data_slice, target):
            return boss_obj.fit(data_slice, target)

        # 👇 THIS is where tqdm goes
        iterator = tasks
        if self.verbose:
            iterator = tqdm(tasks, desc="BOSS fits", total=len(tasks))

        result = Parallel(n_jobs=1, prefer="threads")(
            delayed(_do_fit)(boss_obj, data_slice, y)
            for boss_obj, data_slice in iterator
        )

        self.boss_list = list(result)

        with open(cache_path, "wb") as f:
            pickle.dump(self.boss_list, f)

        if self.verbose:
            print(f"[MultiBOSS] Fit complete and cached to {cache_path}.", flush=True)

        return self

    def transform(self, x):
        t0 = time.perf_counter()
        if self.verbose:
            print("[MultiBOSS] Transform start. Using Parallel backend.", flush=True)

        tasks = []
        i = 0
        for ch in range(self.n_channels):
            for _ in self.window_sizes:
                # We defer the call to be executed in parallel
                tasks.append((self.boss_list[i], x[:, ch, :]))
                i += 1

        def _do_transform(boss_obj, data_slice):
            return boss_obj.transform(data_slice)

        out_list = Parallel(n_jobs=1, prefer = "threads")(
            delayed(_do_transform)(boss_obj, data_slice) for boss_obj, data_slice in tasks
        )

        xt = np.concatenate(list(out_list), axis=1).astype(np.int32) # type: ignore
        if self.verbose:
            dt = time.perf_counter() - t0
            print(f"[MultiBOSS] Transform complete in {dt:.2f}s. Output shape={xt.shape}", flush=True)
        return xt

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, x):
        return self.transform(x)

    def predict_proba(self, x):
        return self.transform(x)
