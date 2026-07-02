"""
Registration-based motion field for the ICCV2019 flow head.

Uses the fine-tuned biomechanics ``Registration_Net`` (Qin et al. MICCAI 2020) as a
FROZEN teacher: given a (source, target) grayscale frame pair it returns a dense 2-D
displacement field in the SAME convention as ``cv2.calcOpticalFlowFarneback`` — a
``(H, W, 2)`` float32 array ``[dx, dy]`` in pixel units — so it is a drop-in replacement
for the optical-flow GT target that ``data_loader.py`` feeds the GAN aux head.

The net is not trained here; we only distill its motion into the ICCV aux head, so the
GAN objective/losses are unchanged.  See the plan / CLAUDE.md for the full rationale.

Convention bridge (the crux):
  * ``net['out']`` is a displacement in NORMALISED grid units (~[-1, 1]) at the net's
    training size (96x96).  A normalised value ``v`` == ``v * (R/2)`` pixels at side R.
  * ``generate_grid`` (network.py) splits ``net['out']`` as (offset_h, offset_w) =
    channels (0, 1) = (dy, dx).  Farneback uses flow[...,0]=dx, flow[...,1]=dy.
  * Direction is handled by the caller: it passes (src, dst) already ordered per the
    ED/ES direction, and we register src->dst via ``model(src, dst, src)``.
"""
import os
import sys

import cv2
import numpy as np

_REPO = os.path.dirname(os.path.abspath(__file__))
# The biomechanics repo is a sibling of this one under the MRes workspace root.
_DEF_REG_REPO = os.path.normpath(os.path.join(_REPO, "..", "Biomechanics-informed-motion-tracking"))
_DEF_REG_CKPT = os.path.join(_DEF_REG_REPO, "checkpoints", "ckpt_best.pth")

# module-level config + lazily-loaded model (torch is imported only on first use, so a
# pure Farneback run never needs torch installed alongside TensorFlow)
_CFG = dict(reg_repo=_DEF_REG_REPO, reg_ckpt=_DEF_REG_CKPT, size=96, device="cpu")
_MODEL = None
_TORCH = None


def configure(reg_repo=None, reg_ckpt=None, size=None, device=None):
    """Override defaults (called from run_model.py before any load_*)."""
    global _MODEL
    if reg_repo is not None:
        _CFG["reg_repo"] = reg_repo
        if reg_ckpt is None:
            _CFG["reg_ckpt"] = os.path.join(reg_repo, "checkpoints", "ckpt_best.pth")
    if reg_ckpt is not None:
        _CFG["reg_ckpt"] = reg_ckpt
    if size is not None:
        _CFG["size"] = int(size)
    if device is not None:
        _CFG["device"] = device
    _MODEL = None  # force reload with new config


def _load_model():
    global _MODEL, _TORCH
    if _MODEL is not None:
        return _MODEL
    try:
        import torch  # deferred: only needed for the registration backend
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "The registration aux backend (--aux_source registration) needs PyTorch, but "
            "it is not installed in this environment. Install a CPU build on the machine "
            "that runs training (on an HPC, do this on the LOGIN node so it persists in the "
            "conda env; compute nodes usually have no internet):\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "CPU-only is enough — the reg-net runs on CPU and won't contend with TensorFlow "
            "on the GPU. Or use --aux_source flow (Farneback), which needs no torch."
        ) from e
    _TORCH = torch
    reg_repo, reg_ckpt = _CFG["reg_repo"], _CFG["reg_ckpt"]
    if reg_repo not in sys.path:
        sys.path.insert(0, reg_repo)
    from network import Registration_Net  # from the biomechanics repo

    if not os.path.exists(reg_ckpt):
        raise FileNotFoundError(
            f"registration checkpoint not found: {reg_ckpt}\n"
            f"Download the fine-tuned ckpt_best.pth into {os.path.dirname(reg_ckpt)} "
            f"or pass --reg_ckpt / --reg_repo.")
    model = Registration_Net().to(_CFG["device"])
    sd = torch.load(reg_ckpt, map_location=_CFG["device"])
    if isinstance(sd, dict) and "model" in sd:   # train_hpc.py checkpoint -> unwrap
        print(f"[registration_flow] loaded fine-tuned ckpt (epoch {sd.get('epoch','?')}, "
              f"best {sd.get('best','?')}): {os.path.basename(reg_ckpt)}")
        sd = sd["model"]
    else:
        print(f"[registration_flow] loaded state_dict: {os.path.basename(reg_ckpt)}")
    model.load_state_dict(sd)
    model.eval()
    _MODEL = model
    return _MODEL


def _to_net_input(gray, size):
    """uint8/float (H,W) grayscale -> (1,1,size,size) float32 tensor in ~[0,1]."""
    g = gray.astype(np.float32)
    if g.max() > 1.5:            # uint8 [0,255] -> [0,1]
        g = g / 255.0
    p99 = np.percentile(g, 99)  # light renorm to match the reg-net's training distribution
    if p99 > 1e-6:
        g = np.clip(g, 0.0, p99) / p99
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_LINEAR)
    t = _TORCH.from_numpy(g[None, None]).float().to(_CFG["device"])
    return t


def registration_flow(src_gray, dst_gray):
    """Dense src->dst displacement field as (H, W, 2) float32 [dx, dy] in pixels.

    Drop-in replacement for cv2.calcOpticalFlowFarneback's return value.
    """
    model = _load_model()
    torch = _TORCH
    H, W = src_gray.shape[:2]
    size = _CFG["size"]

    src_t = _to_net_input(src_gray, size)
    dst_t = _to_net_input(dst_gray, size)
    with torch.no_grad():
        net = model(src_t, dst_t, src_t)          # register src -> dst
    out = net["out"][0].cpu().numpy()             # (2, size, size), normalised grid units

    # Channel->axis mapping verified empirically by myocardium-Dice of the propagated ACDC
    # seg on ICCV-preprocessed 128px frames: out[0] is the x/width (dx) displacement and
    # out[1] the y/height (dy) displacement in the Farneback [dx, dy] convention (this is
    # the swap of generate_grid's misleading offset_h/offset_w naming; getting it backwards
    # drops Dice from 0.76 -> 0.68, i.e. barely above the no-motion baseline).
    # Upsample the normalised field to the frame resolution, then convert to pixels.
    dx = cv2.resize(out[0], (W, H), interpolation=cv2.INTER_LINEAR) * (W / 2.0)
    dy = cv2.resize(out[1], (W, H), interpolation=cv2.INTER_LINEAR) * (H / 2.0)
    flow = np.dstack((dx, dy)).astype(np.float32)  # [dx, dy] to match Farneback
    return flow
