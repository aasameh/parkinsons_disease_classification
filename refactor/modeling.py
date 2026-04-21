import numpy as np
from time import perf_counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from multi_boss import MultiBOSS


def get_channels(type="movement"):
    channels = []
    if type == "movement":
        for task in [
            "Relaxed1",
            "Relaxed2",
            "RelaxedTask1",
            "RelaxedTask2",
            "StretchHold",
            "HoldWeight",
            "DrinkGlas",
            "CrossArms",
            "TouchNose",
            "Entrainment1",
            "Entrainment2",
        ]:
            for device_location in ["LeftWrist", "RightWrist"]:
                for sensor in ["Acceleration", "Rotation"]:
                    for axis in ["X", "Y", "Z"]:
                        channels.append(f"{task}_{sensor}_{device_location}_{axis}")
    return channels


def get_balanced_weights(y, c=None):
    if c is not None:
        y = np.array([y, c], dtype=str)
        y = np.array(list(map("_".join, zip(*y))))

    weights = np.zeros(len(y), dtype=np.float32)
    unique_labels = np.unique(y)
    for idx, label in enumerate(y):
        weights[idx] = len(y) / (len(unique_labels) * np.count_nonzero(y == label))
    return weights


class SampleWeightPipeline(Pipeline):
    def __init__(self, steps, *, show_progress=False):
        super().__init__(steps)
        self.show_progress = bool(show_progress)

    def fit(self, X, y=None, sample_weight=None, **fit_params):
        if self.show_progress:
            boss = self.named_steps["boss"]
            scaler = self.named_steps["scaler"]
            clf = self.named_steps["clf"]

            t0 = perf_counter()
            print("[Pipeline] Step 1/3: boss.fit_transform", flush=True)
            Xt = boss.fit_transform(X, y)
            print(f"[Pipeline] Step 1/3 done in {perf_counter() - t0:.2f}s | shape={Xt.shape}", flush=True)

            t1 = perf_counter()
            print("[Pipeline] Step 2/3: scaler.fit_transform", flush=True)
            Xt = scaler.fit_transform(Xt, y)
            print(f"[Pipeline] Step 2/3 done in {perf_counter() - t1:.2f}s", flush=True)

            t2 = perf_counter()
            print("[Pipeline] Step 3/3: clf.fit", flush=True)
            if sample_weight is not None:
                clf.fit(Xt, y, sample_weight=sample_weight)
            else:
                clf.fit(Xt, y)
            print(f"[Pipeline] Step 3/3 done in {perf_counter() - t2:.2f}s", flush=True)
            print(f"[Pipeline] Total fit time: {perf_counter() - t0:.2f}s", flush=True)
            return self

        if sample_weight is not None:
            return super().fit(X, y, clf__sample_weight=sample_weight, **fit_params)
        return super().fit(X, y, **fit_params)


def build_estimator(data_shape, svm_params, boss_cache_dir, show_progress=False):
    estimator = SampleWeightPipeline(
        [
            (
                "boss",
                MultiBOSS(
                    data_shape=data_shape,
                    window_sizes=(20, 40, 80),
                    window_step=2,
                    word_size=2,
                    n_bins=3,
                    cache_dir=boss_cache_dir,
                    verbose=show_progress,
                ),
            ),
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=False, verbose=show_progress)),
        ],
        show_progress=show_progress,
    )
    estimator.set_params(**svm_params)
    return estimator
