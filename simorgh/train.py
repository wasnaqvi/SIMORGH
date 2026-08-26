"""Training loops.

Two entry points:

  train_npe        toy / small runs: simulate in memory, then train.
  train_npe_grid   cluster runs: train from a precomputed clean grid with
                   noise resampled every epoch, on GPU, with
                   checkpoint/resume.

Resume is not a nicety on a scheduler with walltime limits: a job that
dies at hour 5 of a 6 h allocation must restart from its last epoch, not
from scratch. Every epoch writes an atomic checkpoint containing model,
optimizer, scheduler and RNG state.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .data import NoiseModel, NoisyGridDataset, make_training_set
from .models.npe import AmortizedPosterior
from .priors import BoxPrior


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _TorchGrid(Dataset):
    """Adapts NoisyGridDataset to torch's Dataset protocol."""

    def __init__(self, inner: NoisyGridDataset):
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int):
        theta, tokens, mask = self.inner[idx]
        return (torch.from_numpy(np.asarray(theta)),
                torch.from_numpy(tokens),
                torch.from_numpy(mask))


def train_npe(simulator, prior: BoxPrior, wavelength: np.ndarray,
              n_sims: int = 30_000, epochs: int = 20, batch_size: int = 256,
              lr: float = 1e-3, noise: NoiseModel | None = None,
              val_fraction: float = 0.1, seed: int = 0,
              provenance: dict | None = None,
              verbose: bool = True) -> tuple[AmortizedPosterior, dict]:
    """Simulate once, train in memory. For toy problems and tests."""
    torch.manual_seed(seed)
    theta, tokens, masks = make_training_set(
        simulator, prior, wavelength, n_sims, noise=noise, seed=seed)

    n_val = int(n_sims * val_fraction)
    ds = TensorDataset(torch.as_tensor(theta),
                       torch.as_tensor(tokens),
                       torch.as_tensor(masks))
    val_ds = TensorDataset(*ds[:n_val])
    train_ds = TensorDataset(*ds[n_val:])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1024)

    model = AmortizedPosterior(prior, provenance=provenance or {})
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {"train": [], "val": []}
    best_val, best_state = np.inf, None
    for epoch in range(epochs):
        model.train()
        tr = 0.0
        for th, tok, mk in train_dl:
            opt.zero_grad()
            loss = model.loss(th, tok, mk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr += loss.item() * th.shape[0]
        sched.step()
        tr /= len(train_ds)

        model.eval()
        with torch.no_grad():
            va = sum(model.loss(th, tok, mk).item() * th.shape[0]
                     for th, tok, mk in val_dl) / max(len(val_ds), 1)
        history["train"].append(tr)
        history["val"].append(va)
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if verbose:
            print(f"epoch {epoch + 1:3d}/{epochs}  train {tr:8.3f}  val {va:8.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def _save_checkpoint(path: Path, model, opt, sched, epoch: int,
                     history: dict, best_val: float, best_state) -> None:
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "epoch": epoch,
        "history": history,
        "best_val": best_val,
        "best_state": best_state,
        "torch_rng": torch.get_rng_state(),
    }, tmp)
    tmp.replace(path)


def train_npe_grid(grid_dir: str | Path, out_dir: str | Path,
                   epochs: int = 100, batch_size: int = 512, lr: float = 3e-4,
                   noise: NoiseModel | None = None, val_fraction: float = 0.05,
                   seed: int = 0, device: str = "auto", num_workers: int = 4,
                   context_dim: int = 128, transforms: int = 8,
                   hidden: tuple[int, ...] = (256, 256),
                   resume: bool = True, verbose: bool = True
                   ) -> tuple[AmortizedPosterior, dict]:
    """Train from a precomputed clean grid, with resume. Cluster path."""
    from .simulate.grid import grid_prior, load_grid

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"

    theta, depth, meta = load_grid(grid_dir)
    prior = grid_prior(meta)
    wavelength = np.asarray(meta["wavelength"], dtype=np.float64)
    dev = pick_device(device)
    torch.manual_seed(seed)

    n = theta.shape[0]
    n_val = max(int(n * val_fraction), 1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    noise = noise or NoiseModel()
    train_ds = _TorchGrid(NoisyGridDataset(
        theta[train_idx], depth[train_idx], wavelength, noise, seed=seed))
    val_ds = _TorchGrid(NoisyGridDataset(
        theta[val_idx], depth[val_idx], wavelength, noise, seed=seed + 1,
        fixed=True))          # frozen noise: val loss compares like with like
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=True,
                          persistent_workers=num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=1024, num_workers=0)

    provenance = {
        "grid_dir": str(Path(grid_dir).resolve()),
        "meta_hash": meta["meta_hash"],
        "simulator": meta["simulator"],
        "n_sims": int(n),
        "noise": {"ppm_min": noise.ppm_min, "ppm_max": noise.ppm_max,
                  "red_slope": noise.red_slope, "p_drop": noise.p_drop},
        "epochs_requested": epochs,
        "seed": seed,
    }
    model = AmortizedPosterior(prior, context_dim=context_dim,
                               transforms=transforms, hidden=hidden,
                               provenance=provenance).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    start_epoch, history = 0, {"train": [], "val": []}
    best_val, best_state = np.inf, None
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        history = ck["history"]
        best_val = ck["best_val"]
        best_state = ck["best_state"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if verbose:
            print(f"resumed from {ckpt_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_ds.inner.set_epoch(epoch)       # fresh noise this epoch
        model.train()
        tr, seen = 0.0, 0
        for th, tok, mk in train_dl:
            th, tok, mk = th.to(dev), tok.to(dev), mk.to(dev)
            opt.zero_grad()
            loss = model.loss(th, tok, mk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr += loss.item() * th.shape[0]
            seen += th.shape[0]
        sched.step()
        tr /= max(seen, 1)

        model.eval()
        with torch.no_grad():
            vs, vn = 0.0, 0
            for th, tok, mk in val_dl:
                th, tok, mk = th.to(dev), tok.to(dev), mk.to(dev)
                vs += model.loss(th, tok, mk).item() * th.shape[0]
                vn += th.shape[0]
            va = vs / max(vn, 1)

        history["train"].append(tr)
        history["val"].append(va)
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            model.save(out_dir / "best")
        _save_checkpoint(ckpt_path, model, opt, sched, epoch, history,
                         best_val, best_state)
        if verbose:
            print(f"epoch {epoch + 1:4d}/{epochs}  train {tr:9.3f}  "
                  f"val {va:9.3f}  best {best_val:9.3f}  "
                  f"{time.time() - t0:6.1f}s", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.provenance["epochs_completed"] = epochs
    model.provenance["best_val_loss"] = float(best_val)
    model.save(out_dir / "final")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return model, history
