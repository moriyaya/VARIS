"""Six-response compact U-Net, expert-set matching and QUBIQ preprocessing."""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_2d(path: Path) -> np.ndarray:
    import nibabel as nib
    return np.squeeze(np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32))


def resize(array: np.ndarray, size: int, mode: str) -> torch.Tensor:
    tensor = torch.from_numpy(array).float()[None, None]
    kwargs = {"size": (size, size), "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(tensor, **kwargs)[0, 0]


def preference_indices(masks: torch.Tensor) -> torch.Tensor:
    # masks: A,H,W. Preferences are conservative, expert medoid, sensitive.
    areas = masks.flatten(1).sum(1)
    conservative = int(areas.argmin())
    sensitive = int(areas.argmax())
    flat = masks.flatten(1)
    inter = flat @ flat.T
    denom = areas[:, None] + areas[None, :]
    dice = (2 * inter + 1e-6) / (denom + 1e-6)
    medoid = int(((dice.sum(1) - 1) / (len(masks) - 1)).argmax())
    return torch.tensor([conservative, medoid, sensitive], dtype=torch.long)


class QubiqDataset(Dataset):
    def __init__(self, case_dirs: list[Path], task: int, size: int, augment: bool):
        self.items = []
        self.expert_ids = {}
        self.augment = augment
        for case_dir in case_dirs:
            image = load_2d(case_dir / "image.nii.gz")
            nonzero = image[image > 0]
            if nonzero.size:
                lo, hi = np.percentile(nonzero, [1, 99])
            else:
                lo, hi = float(image.min()), float(image.max())
            image = np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)
            image_t = resize(image, size, "bilinear")[None]
            mask_paths = sorted(case_dir.glob(f"task{task:02d}_seg*.nii.gz"))
            self.expert_ids[case_dir.name] = torch.tensor(
                [int(path.name.split("seg", 1)[1].split(".", 1)[0]) - 1 for path in mask_paths],
                dtype=torch.long,
            )
            masks = torch.stack(
                [resize((load_2d(path) > 0.5).astype(np.float32), size, "nearest") for path in mask_paths]
            )
            self.items.append((case_dir.name, image_t, masks, preference_indices(masks)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        name, image, masks, pref = self.items[index]
        image, masks = image.clone(), masks.clone()
        if self.augment:
            if random.random() < 0.5:
                image, masks = image.flip(-1), masks.flip(-1)
            if random.random() < 0.5:
                image, masks = image.flip(-2), masks.flip(-2)
            k = random.randrange(4)
            if k:
                image, masks = torch.rot90(image, k, (-2, -1)), torch.rot90(masks, k, (-2, -1))
        return name, image, masks, pref


def qubiq_collate(batch):
    """Keep variable-size expert sets while batching images and preferences."""
    names, images, masks, preferences = zip(*batch)
    return list(names), torch.stack(images), list(masks), torch.stack(preferences)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, cout), cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, cout), cout),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class CompactUNet(nn.Module):
    def __init__(self, cin: int, cout: int, width: int = 16):
        super().__init__()
        self.e1 = ConvBlock(cin, width)
        self.e2 = ConvBlock(width, width * 2)
        self.e3 = ConvBlock(width * 2, width * 4)
        self.mid = ConvBlock(width * 4, width * 8)
        self.d3 = ConvBlock(width * 12, width * 4)
        self.d2 = ConvBlock(width * 6, width * 2)
        self.d1 = ConvBlock(width * 3, width)
        self.out = nn.Conv2d(width, cout, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        z = self.mid(F.max_pool2d(e3, 2))
        z = self.d3(torch.cat([F.interpolate(z, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1))
        z = self.d2(torch.cat([F.interpolate(z, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        z = self.d1(torch.cat([F.interpolate(z, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        return self.out(z)


def binary_cost(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits B,K,H,W; targets B,A,H,W -> B,K,A
    logit = logits[:, :, None]
    target = targets[:, None]
    bce = F.binary_cross_entropy_with_logits(
        logit.expand(-1, -1, targets.shape[1], -1, -1),
        target.expand(-1, logits.shape[1], -1, -1, -1),
        reduction="none",
    ).mean((-1, -2))
    prob = torch.sigmoid(logit)
    inter = (prob * target).sum((-1, -2))
    denom = prob.sum((-1, -2)) + target.sum((-1, -2))
    dice_loss = 1 - (2 * inter + 1) / (denom + 1)
    return bce + dice_loss


def matched_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    costs = binary_cost(logits, masks)
    losses = []
    for b in range(len(logits)):
        rows, cols = linear_sum_assignment(costs[b].detach().cpu().numpy())
        pred_indices = torch.as_tensor(rows, device=logits.device)
        target_indices = torch.as_tensor(cols, device=logits.device)
        matched = costs[b, pred_indices, target_indices].mean()
        losses.append(matched)
    return torch.stack(losses).mean()


def iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    af, bf = a.flatten(1), b.flatten(1)
    inter = af @ bf.T
    union = af.sum(1)[:, None] + bf.sum(1)[None, :] - inter
    return (inter + 1e-6) / (union + 1e-6)


def staged_q_dice(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """QUBIQ staged Dice on probability maps at thresholds 0.1,...,0.9."""
    thresholds = torch.arange(1, 10, device=prediction.device) / 10
    pred_binary = prediction[None] >= thresholds[:, None, None]
    target_binary = target[None] >= thresholds[:, None, None]
    inter = (pred_binary & target_binary).flatten(1).sum(1).float()
    denom = pred_binary.flatten(1).sum(1) + target_binary.flatten(1).sum(1)
    dice = torch.where(denom > 0, 2 * inter / denom.float(), torch.ones_like(inter))
    return dice.mean()
