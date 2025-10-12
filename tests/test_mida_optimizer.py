import importlib.util
import sys
import types
from pathlib import Path


def _install_stub_modules(monkeypatch):
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.inf = float("inf")
    numpy_stub.load = lambda *args, **kwargs: None
    numpy_stub.unique = lambda seq: []
    numpy_stub.asarray = lambda arr: arr
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)

    optuna_stub = types.ModuleType("optuna")
    optuna_pruners_stub = types.ModuleType("optuna.pruners")
    optuna_samplers_stub = types.ModuleType("optuna.samplers")

    class _HyperbandPruner:
        def __init__(self, *args, **kwargs):
            pass

    class _TPESampler:
        def __init__(self, *args, **kwargs):
            pass

    optuna_stub.pruners = optuna_pruners_stub
    optuna_stub.samplers = optuna_samplers_stub
    optuna_stub.create_study = lambda *args, **kwargs: types.SimpleNamespace(optimize=lambda *a, **k: None)
    optuna_pruners_stub.HyperbandPruner = _HyperbandPruner
    optuna_samplers_stub.TPESampler = _TPESampler

    monkeypatch.setitem(sys.modules, "optuna", optuna_stub)
    monkeypatch.setitem(sys.modules, "optuna.pruners", optuna_pruners_stub)
    monkeypatch.setitem(sys.modules, "optuna.samplers", optuna_samplers_stub)

    sklearn_stub = types.ModuleType("sklearn")
    sklearn_preprocessing_stub = types.ModuleType("sklearn.preprocessing")

    class _LabelEncoder:
        def fit_transform(self, values):
            return values

        def fit(self, values):
            return self

        def transform(self, values):
            return values

    class _OneHotEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, values):
            return self

        def transform(self, values):
            return values

    sklearn_preprocessing_stub.LabelEncoder = _LabelEncoder
    sklearn_preprocessing_stub.OneHotEncoder = _OneHotEncoder
    sklearn_stub.preprocessing = sklearn_preprocessing_stub

    monkeypatch.setitem(sys.modules, "sklearn", sklearn_stub)
    monkeypatch.setitem(sys.modules, "sklearn.preprocessing", sklearn_preprocessing_stub)

    tensorflow_stub = types.ModuleType("tensorflow")
    keras_stub = types.ModuleType("tensorflow.keras")
    layers_stub = types.ModuleType("tensorflow.keras.layers")
    models_stub = types.ModuleType("tensorflow.keras.models")
    callbacks_stub = types.ModuleType("tensorflow.keras.callbacks")
    optimizers_stub = types.ModuleType("tensorflow.keras.optimizers")

    class _DummyCallable:
        def __call__(self, *args, **kwargs):
            return None

    for attr in ["Dense", "Input", "BatchNormalization", "LSTM", "Reshape", "Embedding", "concatenate"]:
        setattr(layers_stub, attr, _DummyCallable())

    class _Model:
        def __init__(self, *args, **kwargs):
            pass

        def compile(self, *args, **kwargs):  # pragma: no cover - placeholder only
            pass

        def summary(self):  # pragma: no cover - placeholder only
            pass

        def count_params(self):  # pragma: no cover - placeholder only
            return 0

        def save(self, *args, **kwargs):  # pragma: no cover - placeholder only
            pass

    class _Callback:
        def __init__(self, *args, **kwargs):
            pass

    class _Nadam:
        def __init__(self, *args, **kwargs):
            pass

    models_stub.Model = _Model
    callbacks_stub.EarlyStopping = _Callback
    callbacks_stub.ReduceLROnPlateau = _Callback
    optimizers_stub.Nadam = _Nadam

    tensorflow_stub.keras = types.SimpleNamespace(
        layers=layers_stub,
        models=models_stub,
        callbacks=callbacks_stub,
        optimizers=optimizers_stub,
    )

    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow_stub)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", keras_stub)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.layers", layers_stub)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.models", models_stub)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.callbacks", callbacks_stub)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.optimizers", optimizers_stub)


def _load_mida(monkeypatch):
    _install_stub_modules(monkeypatch)
    project_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("mida_under_test", project_root / "MiDA.py")
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    return module


def test_create_optimizer_uses_supported_kwargs(monkeypatch):
    mida_module = _load_mida(monkeypatch)
    cfg = {"learning_rate_init": 0.001}
    mida = mida_module.MiDA("dummy")

    captured = {}

    class DummyNadam:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr(mida_module, "Nadam", DummyNadam)

    optimizer = mida._create_optimizer(cfg)

    assert isinstance(optimizer, DummyNadam)
    assert captured["kwargs"]["learning_rate"] == cfg["learning_rate_init"]
    assert "schedule_decay" not in captured["kwargs"]

