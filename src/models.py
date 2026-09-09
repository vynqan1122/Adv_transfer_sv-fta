import os
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SURROGATE_ALL = [
    "resnet50", "densenet121", "vit_base_patch16_224", "deit_small_patch16_224",
]
SURROGATE_CNN = ["resnet50", "densenet121"]
SURROGATE_VIT = ["vit_base_patch16_224", "deit_small_patch16_224"]
TARGET_CNN = ["resnet152", "inception_v3"]
TARGET_VIT = ["vit_large_patch16_224", "deit_base_patch16_224", "swin_tiny_patch4_window7_224"]
TARGET_ALL = TARGET_CNN + TARGET_VIT
VIT_LIKE_KEYWORDS = ("vit_", "deit_", "swin_", "ViT", "Swin")


def is_vit_like(model_name):
    return any(k in model_name for k in VIT_LIKE_KEYWORDS)


class Normalize(nn.Module):
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        if len(mean) != len(std) or not mean or any(s <= 0 for s in std):
            raise ValueError("mean/std must have equal, nonzero length and positive std")
        self.register_buffer("mean", torch.tensor(mean).view(1, len(mean), 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, len(std), 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class NormalizedModel(nn.Module):
    def __init__(self, model, mean=IMAGENET_MEAN, std=IMAGENET_STD, input_size=None):
        super().__init__()
        self.normalize = Normalize(mean, std)
        self.model = model
        self.input_size = tuple(input_size[-2:]) if input_size is not None else None

    def forward(self, x):
        # Attacks use a shared crop; adapt differentiably to checkpoint size.
        if self.input_size is not None and tuple(x.shape[-2:]) != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return self.model(self.normalize(x))


def _default_robustbench_dir():
    env = os.environ.get("ROBUSTBENCH_MODEL_DIR", "").strip()
    if env:
        return os.path.abspath(env)
    # src/models.py -> project root -> models
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def _load_robustbench(spec, device, robustbench_model_dir=None):
    parts = spec.split(":")
    if not 2 <= len(parts) <= 4 or not parts[1]:
        raise ValueError("Expected robustbench:<name>[:<dataset>[:<threat_model>]]")
    rb_name = parts[1]
    dataset = parts[2] if len(parts) >= 3 and parts[2] else "imagenet"
    threat_model = parts[3] if len(parts) >= 4 and parts[3] else "Linf"
    model_dir = os.path.abspath(robustbench_model_dir or _default_robustbench_dir())
    expected = os.path.join(model_dir, dataset, threat_model, rb_name + ".pt")

    if not os.path.isfile(expected):
        raise FileNotFoundError(
            "RobustBench checkpoint not found:\n  {}\n"
            "ROBUSTBENCH_MODEL_DIR must be the PARENT directory containing {}/{}/.\n"
            "For this project use: <repo>/models (not <repo>/models/{}/{}).".format(
                expected, dataset, threat_model, dataset, threat_model
            )
        )

    try:
        from robustbench.utils import load_model as rb_load_model
    except Exception as exc:
        raise RuntimeError(
            "The local defense checkpoint exists, but RobustBench could not be imported. "
            "Use the project environment where RobustBench is installed. The checkpoint path itself is correct: {}".format(expected)
        ) from exc

    kwargs = dict(model_name=rb_name, dataset=dataset, threat_model=threat_model)
    try:
        sig = inspect.signature(rb_load_model)
        if "model_dir" in sig.parameters:
            kwargs["model_dir"] = model_dir
        elif "models_dir" in sig.parameters:
            kwargs["models_dir"] = model_dir
    except Exception:
        kwargs["model_dir"] = model_dir

    try:
        model = rb_load_model(**kwargs)
    except TypeError:
        # Older RobustBench versions may only accept model_dir positionally/by a different signature.
        model = rb_load_model(rb_name, model_dir=model_dir, dataset=dataset, threat_model=threat_model)

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def create_model(model_name, device, pretrained=True, robustbench_model_dir=None):
    """Create a timm or RobustBench model.

    Supported specs:
      timm:<name>
      <plain timm name>
      robustbench:<name>:imagenet:Linf
    """
    if model_name.startswith("robustbench:"):
        return _load_robustbench(model_name, device, robustbench_model_dir)

    if model_name.startswith("timm:"):
        model_name = model_name.split(":", 1)[1]

    import timm
    model = timm.create_model(model_name, pretrained=pretrained)
    cfg = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", {})
    model = NormalizedModel(
        model,
        mean=cfg.get("mean", IMAGENET_MEAN),
        std=cfg.get("std", IMAGENET_STD),
        input_size=cfg.get("input_size"),
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_models(model_names, device, pretrained=True, robustbench_model_dir=None):
    model_names = list(model_names)
    if len(model_names) != len(set(model_names)):
        raise ValueError("Model names must be unique; duplicates would silently reduce the source pool")
    models = {}
    for name in model_names:
        models[name] = create_model(
            name, device, pretrained=pretrained, robustbench_model_dir=robustbench_model_dir
        )
    return models
