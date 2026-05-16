from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ResizeWithWhitePadding:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.contain(image, (self.size, self.size), method=Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (self.size, self.size), "white")
        canvas.paste(image, ((self.size - image.width) // 2, (self.size - image.height) // 2))
        return canvas


class MeteoriteDataset(Dataset):
    def __init__(self, image_dir: Path, image_size: int) -> None:
        self.paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        self.transform = transforms.Compose(
            [
                ResizeWithWhitePadding(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=6,
                    translate=(0.025, 0.025),
                    scale=(0.94, 1.06),
                    fill=(255, 255, 255),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ColorJitter(brightness=0.06, contrast=0.08, saturation=0.06, hue=0.01),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(Image.open(self.paths[index]).convert("RGB"))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / max(1, half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=time.device) * -emb)
        emb = time[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).transpose(1, 2)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).transpose(1, 2)
        attn = torch.softmax(torch.bmm(q, k) * (c ** -0.5), dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 4),
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        self.init_conv = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        dims = [base_channels * mult for mult in channel_mults]
        in_out = list(zip([base_channels] + dims[:-1], dims))

        self.downs = nn.ModuleList()
        skip_channels: list[int] = []
        for index, (dim_in, dim_out) in enumerate(in_out):
            use_attn = index == len(in_out) - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        ResBlock(dim_in, dim_out, time_dim),
                        ResBlock(dim_out, dim_out, time_dim),
                        AttentionBlock(dim_out) if use_attn else nn.Identity(),
                        Downsample(dim_out) if index < len(in_out) - 1 else nn.Identity(),
                    ]
                )
            )
            skip_channels.extend([dim_out, dim_out])

        mid_dim = dims[-1]
        self.mid1 = ResBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = AttentionBlock(mid_dim)
        self.mid2 = ResBlock(mid_dim, mid_dim, time_dim)

        self.ups = nn.ModuleList()
        current_dim = mid_dim
        for index, (dim_in, dim_out) in enumerate(reversed(in_out)):
            use_attn = index == 0
            skip2 = skip_channels.pop()
            skip1 = skip_channels.pop()
            self.ups.append(
                nn.ModuleList(
                    [
                        ResBlock(current_dim + skip2, dim_out, time_dim),
                        ResBlock(dim_out + skip1, dim_out, time_dim),
                        AttentionBlock(dim_out) if use_attn else nn.Identity(),
                        Upsample(dim_out) if index < len(in_out) - 1 else nn.Identity(),
                    ]
                )
            )
            current_dim = dim_out

        self.final = nn.Sequential(
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, image_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(time)
        x = self.init_conv(x)
        skips = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_emb)
            skips.append(x)
            x = block2(x, time_emb)
            x = attn(x)
            skips.append(x)
            x = downsample(x)

        x = self.mid1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, time_emb)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = block1(x, time_emb)
            x = torch.cat((x, skips.pop()), dim=1)
            x = block2(x, time_emb)
            x = attn(x)
            x = upsample(x)
        return self.final(x)


class GaussianDiffusion(nn.Module):
    def __init__(self, model: UNet, image_size: int, timesteps: int = 1000) -> None:
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.timesteps = timesteps

        betas = self.cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def extract(self, values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = values.gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, x_start: torch.Tensor) -> torch.Tensor:
        batch = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (batch,), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)
        return F.mse_loss(predicted_noise, noise)


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow


@dataclass
class TrainConfig:
    data_dir: str
    checkpoint_dir: str
    image_size: int
    timesteps: int
    base_channels: int
    batch_size: int
    epochs: int
    lr: float
    seed: int
    ema_decay: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "samples").mkdir(parents=True, exist_ok=True)

    dataset = MeteoriteDataset(args.data_dir, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = UNet(base_channels=args.base_channels).to(device)
    diffusion = GaussianDiffusion(model, image_size=args.image_size, timesteps=args.timesteps).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    ema = EMA(model, args.ema_decay)

    start_epoch = 1
    global_step = 0
    if args.resume and args.resume.is_file():
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.shadow = {name: tensor.to(device) for name, tensor in checkpoint["ema"].items()}
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))

    config = TrainConfig(
        data_dir=str(args.data_dir),
        checkpoint_dir=str(args.checkpoint_dir),
        image_size=args.image_size,
        timesteps=args.timesteps,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        ema_decay=args.ema_decay,
    )
    (args.checkpoint_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Training images: {len(dataset)}")
    print(f"Batches per epoch: {len(loader)}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images in loader:
            images = images.to(device, non_blocking=True)
            loss = diffusion(images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            running_loss += float(loss.detach().cpu())
            global_step += 1

        avg_loss = running_loss / max(1, len(loader))
        print(f"Epoch {epoch:03d}/{args.epochs}  loss={avg_loss:.5f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(config),
                },
                args.checkpoint_dir / "latest.pt",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an unconditional DDPM for meteorite image generation.")
    parser.add_argument("--data-dir", type=Path, default=Path("meteorite"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("diffusion_checkpoints"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
