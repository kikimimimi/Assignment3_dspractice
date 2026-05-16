from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import utils

from train_diffusion import GaussianDiffusion, UNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path: Path, device: torch.device, use_ema: bool) -> tuple[UNet, GaussianDiffusion, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    image_size = int(config.get("image_size", 128))
    timesteps = int(config.get("timesteps", 1000))
    base_channels = int(config.get("base_channels", 64))
    model = UNet(base_channels=base_channels).to(device)
    if use_ema and "ema" in checkpoint:
        model_state = checkpoint["model"]
        ema_state = checkpoint["ema"]
        merged = {name: ema_state.get(name, value) for name, value in model_state.items()}
        model.load_state_dict(merged, strict=False)
    else:
        model.load_state_dict(checkpoint["model"])
    model.eval()
    diffusion = GaussianDiffusion(model, image_size=image_size, timesteps=timesteps).to(device)
    return model, diffusion, config


@torch.no_grad()
def ddim_sample(
    diffusion: GaussianDiffusion,
    batch_size: int,
    channels: int,
    image_size: int,
    sample_steps: int,
    eta: float,
    device: torch.device,
) -> torch.Tensor:
    times = torch.linspace(diffusion.timesteps - 1, 0, sample_steps, device=device).long()
    x = torch.randn(batch_size, channels, image_size, image_size, device=device)

    for index, time in enumerate(times):
        t = torch.full((batch_size,), int(time.item()), device=device, dtype=torch.long)
        alpha = diffusion.alphas_cumprod[t][:, None, None, None]
        eps = diffusion.model(x, t)
        pred_x0 = (x - torch.sqrt(1 - alpha) * eps) / torch.sqrt(alpha)
        pred_x0 = pred_x0.clamp(-1, 1)

        if index == len(times) - 1:
            x = pred_x0
            continue

        next_time = int(times[index + 1].item())
        alpha_next = diffusion.alphas_cumprod[
            torch.full((batch_size,), next_time, device=device, dtype=torch.long)
        ][:, None, None, None]
        sigma = (
            eta
            * torch.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).clamp(min=0)
            if eta > 0
            else 0
        )
        noise = torch.randn_like(x) if eta > 0 else 0
        direction = torch.sqrt((1 - alpha_next - (sigma ** 2 if eta > 0 else 0)).clamp(min=0)) * eps
        x = torch.sqrt(alpha_next) * pred_x0 + direction + (sigma * noise if eta > 0 else 0)
    return x.clamp(-1, 1)


def make_grid(image_paths: list[Path], output_path: Path) -> None:
    tile = 256
    gap = 12
    grid = Image.new("RGB", (tile * 3 + gap * 2, tile * 3 + gap * 2), (245, 245, 245))
    for index, path in enumerate(image_paths[:9]):
        image = Image.open(path).convert("RGB")
        image = ImageOps.contain(image, (tile, tile), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tile, tile), "white")
        canvas.paste(image, ((tile - image.width) // 2, (tile - image.height) // 2))
        x = (index % 3) * (tile + gap)
        y = (index // 3) * (tile + gap)
        grid.paste(canvas, (x, y))
    grid.save(output_path)


def generate(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    _, diffusion, config = load_model(args.checkpoint, device, args.use_ema)
    image_size = int(config.get("image_size", 128))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for path in args.output_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                path.unlink()

    generated: list[Path] = []
    written = 0
    while written < args.num_images:
        batch = min(args.batch_size, args.num_images - written)
        images = ddim_sample(
            diffusion=diffusion,
            batch_size=batch,
            channels=3,
            image_size=image_size,
            sample_steps=args.sample_steps,
            eta=args.eta,
            device=device,
        ).cpu()
        for i in range(batch):
            output_path = args.output_dir / f"generated_{written + i + 1:04d}.jpg"
            utils.save_image(images[i], output_path, normalize=True, value_range=(-1, 1))
            generated.append(output_path)
        written += batch
        if written % 100 == 0 or written == args.num_images:
            print(f"Generated {written}/{args.num_images}")

    if args.grid_path:
        make_grid(generated, args.grid_path)

    metadata = {
        "method": "unconditional DDPM U-Net with DDIM sampling",
        "checkpoint": str(args.checkpoint),
        "num_images": args.num_images,
        "sample_steps": args.sample_steps,
        "eta": args.eta,
        "use_ema": args.use_ema,
        "seed": args.seed,
        "config": config,
    }
    if args.metadata_path:
        args.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate meteorite images from a trained DDPM checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("diffusion_checkpoints/latest.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated_pictures"))
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--grid-path", type=Path, default=Path("generated_grid.png"))
    parser.add_argument("--metadata-path", type=Path, default=Path("generation_metadata.json"))
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
