"""Train TF-IDF pooling parameters for multilingual-e5-large-instruct."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from consts import EMBEDDING_MODEL, MAX_VECTOR_LENGTH
from e5 import create_sentence_encoder
from e5_experiment_data import load_jsonl


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


class TripletDataset(Dataset[dict[str, str]]):
    def __init__(self, triplets: list[dict[str, str]]) -> None:
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.triplets[index]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TF-IDF weighted pooling head for multilingual-e5-large-instruct."
    )
    parser.add_argument("--dataset-dir", default="data/e5_pooling")
    parser.add_argument("--output-dir", default="data/e5_pooling/checkpoints")
    parser.add_argument("--idf-path", default="")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--alpha-init", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--train-last-transformer-block",
        action="store_true",
        help="Optional second-stage mode: train the last transformer block together with pooling.",
    )
    return parser.parse_args()


def build_collate_fn(tokenizer: Any):
    def collate_fn(batch: list[dict[str, str]]) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "anchor": tokenizer(
                [item["anchor_text"] for item in batch],
                return_tensors="pt",
                max_length=MAX_VECTOR_LENGTH,
                truncation=True,
                padding=True,
            ),
            "positive": tokenizer(
                [item["positive_text"] for item in batch],
                return_tensors="pt",
                max_length=MAX_VECTOR_LENGTH,
                truncation=True,
                padding=True,
            ),
            "negative": tokenizer(
                [item["negative_text"] for item in batch],
                return_tensors="pt",
                max_length=MAX_VECTOR_LENGTH,
                truncation=True,
                padding=True,
            ),
        }

    return collate_fn


def move_batch_to_device(
    batch: dict[str, dict[str, torch.Tensor]],
    device: str,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        batch_name: {
            tensor_name: tensor.to(device)
            for tensor_name, tensor in tensors.items()
        }
        for batch_name, tensors in batch.items()
    }


@torch.no_grad()
def evaluate_triplets(
    model: nn.Module,
    dataloader: DataLoader[dict[str, dict[str, torch.Tensor]]],
    *,
    device: str,
    margin: float,
    progress_label: str | None = None,
) -> dict[str, float]:
    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    total_loss = 0.0
    total_examples = 0
    correct = 0

    model.eval()
    iterator = dataloader
    if progress_label:
        iterator = tqdm(
            dataloader,
            total=len(dataloader),
            desc=progress_label,
            dynamic_ncols=True,
            leave=False,
        )

    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        anchor_embedding = model(**batch["anchor"]).sentence_embedding
        positive_embedding = model(**batch["positive"]).sentence_embedding
        negative_embedding = model(**batch["negative"]).sentence_embedding

        loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
        batch_size = anchor_embedding.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        positive_similarity = torch.sum(anchor_embedding * positive_embedding, dim=1)
        negative_similarity = torch.sum(anchor_embedding * negative_embedding, dim=1)
        correct += int((positive_similarity > negative_similarity).sum().item())

        if progress_label:
            iterator.set_postfix(
                val_loss=f"{(total_loss / max(total_examples, 1)):.4f}",
                val_acc=f"{(correct / max(total_examples, 1)):.4f}",
            )

    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {
        "loss": total_loss / total_examples,
        "accuracy": correct / total_examples,
    }


def checkpoint_payload(
    model: nn.Module,
    *,
    train_last_transformer_block: bool,
    metrics: dict[str, float],
    idf_path: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_name": EMBEDDING_MODEL,
        "pooling_mode": "tfidf_weightedmean",
        "pooling_state_dict": model.pooling.state_dict(),
        "idf_path": idf_path,
        "train_last_transformer_block": train_last_transformer_block,
        "metrics": metrics,
        "alpha": float(model.pooling.alpha.detach().cpu().item()),
    }

    if train_last_transformer_block:
        payload["last_transformer_block_state_dict"] = (
            model.get_last_transformer_block().state_dict()
        )

    return payload


def save_training_plots(
    history: list[dict[str, float]],
    *,
    output_dir: Path,
) -> None:
    if not history:
        logger.warning("Training history is empty, skipping plot generation")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = [int(item["epoch"]) for item in history]
    train_loss = [item["train_loss"] for item in history]
    validation_loss = [item["validation_loss"] for item in history]
    validation_accuracy = [item["validation_accuracy"] for item in history]
    alpha_values = [item["alpha"] for item in history]

    figure, axes = plt.subplots(3, 1, figsize=(10, 14), constrained_layout=True)

    axes[0].plot(epochs, train_loss, marker="o", linewidth=2, label="train_loss")
    axes[0].plot(
        epochs,
        validation_loss,
        marker="o",
        linewidth=2,
        label="validation_loss",
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        validation_accuracy,
        marker="o",
        linewidth=2,
        color="tab:green",
        label="validation_accuracy",
    )
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        epochs,
        alpha_values,
        marker="o",
        linewidth=2,
        color="tab:orange",
        label="alpha",
    )
    axes[2].set_title("Pooling Alpha")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Alpha")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    png_path = plots_dir / "training_curves.png"
    svg_path = plots_dir / "training_curves.svg"
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)

    best_epoch = max(
        history,
        key=lambda item: (item["validation_accuracy"], -item["validation_loss"]),
    )
    summary = {
        "best_epoch": int(best_epoch["epoch"]),
        "best_validation_accuracy": best_epoch["validation_accuracy"],
        "best_validation_loss": best_epoch["validation_loss"],
        "final_alpha": history[-1]["alpha"],
        "plots": {
            "png": str(png_path),
            "svg": str(svg_path),
        },
    }
    (plots_dir / "plot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    arguments = parse_arguments()
    dataset_dir = Path(arguments.dataset_dir)
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    idf_path = arguments.idf_path or str(dataset_dir / "idf_token_id.json")
    train_triplets = load_jsonl(dataset_dir / "train_triplets.jsonl")
    validation_triplets = load_jsonl(dataset_dir / "validation_triplets.jsonl")

    tokenizer, model = create_sentence_encoder(
        model_name=EMBEDDING_MODEL,
        pooling_mode="tfidf_weightedmean",
        idf_path=idf_path,
        alpha_init=arguments.alpha_init,
        device=arguments.device,
    )
    model.freeze_encoder()
    if arguments.train_last_transformer_block:
        model.unfreeze_last_transformer_block()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    loss_fn = nn.TripletMarginLoss(margin=arguments.margin, p=2)

    train_loader = DataLoader(
        TripletDataset(train_triplets),
        batch_size=arguments.batch_size,
        shuffle=True,
        collate_fn=build_collate_fn(tokenizer),
    )
    validation_loader = DataLoader(
        TripletDataset(validation_triplets),
        batch_size=arguments.batch_size,
        shuffle=False,
        collate_fn=build_collate_fn(tokenizer),
    )

    logger.info(
        "Training started | device=%s epochs=%d batch_size=%d train_triplets=%d validation_triplets=%d train_batches=%d validation_batches=%d",
        arguments.device,
        arguments.epochs,
        arguments.batch_size,
        len(train_triplets),
        len(validation_triplets),
        len(train_loader),
        len(validation_loader),
    )

    best_metrics = {"loss": float("inf"), "accuracy": 0.0}
    history: list[dict[str, float]] = []

    for epoch in range(1, arguments.epochs + 1):
        logger.info("Epoch %d/%d started", epoch, arguments.epochs)
        model.train()
        total_loss = 0.0
        total_examples = 0

        train_iterator = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch}/{arguments.epochs} [train]",
            dynamic_ncols=True,
        )

        for batch in train_iterator:
            batch = move_batch_to_device(batch, arguments.device)
            anchor_embedding = model(**batch["anchor"]).sentence_embedding
            positive_embedding = model(**batch["positive"]).sentence_embedding
            negative_embedding = model(**batch["negative"]).sentence_embedding

            loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = anchor_embedding.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            train_iterator.set_postfix(
                loss=f"{(total_loss / max(total_examples, 1)):.4f}",
                alpha=f"{float(model.pooling.alpha.detach().cpu().item()):.4f}",
            )

        train_loss = total_loss / max(total_examples, 1)
        validation_metrics = evaluate_triplets(
            model,
            validation_loader,
            device=arguments.device,
            margin=arguments.margin,
            progress_label=f"Epoch {epoch}/{arguments.epochs} [valid]",
        )
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "alpha": float(model.pooling.alpha.detach().cpu().item()),
        }
        history.append(epoch_metrics)

        is_better = (
            validation_metrics["accuracy"] > best_metrics["accuracy"]
            or (
                validation_metrics["accuracy"] == best_metrics["accuracy"]
                and validation_metrics["loss"] < best_metrics["loss"]
            )
        )
        if is_better:
            best_metrics = validation_metrics
            checkpoint_path = output_dir / "best_pooling_checkpoint.pt"
            torch.save(
                checkpoint_payload(
                    model,
                    train_last_transformer_block=arguments.train_last_transformer_block,
                    metrics=validation_metrics,
                    idf_path=idf_path,
                ),
                checkpoint_path,
            )
            logger.info("Saved new best checkpoint: %s", checkpoint_path)

        logger.info(
            "Epoch %d/%d finished | train_loss=%.4f validation_loss=%.4f validation_accuracy=%.4f alpha=%.4f%s",
            epoch,
            arguments.epochs,
            train_loss,
            validation_metrics["loss"],
            validation_metrics["accuracy"],
            float(model.pooling.alpha.detach().cpu().item()),
            " | best" if is_better else "",
        )

    (output_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_training_plots(history, output_dir=output_dir)
    logger.info("Training finished | history=%s", output_dir / "training_history.json")
    logger.info("Plots saved in %s", output_dir / "plots")


if __name__ == "__main__":
    main()
