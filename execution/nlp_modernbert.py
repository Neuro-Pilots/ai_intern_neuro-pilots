"""Real ModernBERT fine-tuning for Amazon Reviews 2023 Automotive."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ingestion.dataset_versioning import DatasetVersioner
from utils.models import ExperimentResult


class AmazonAutomotiveModernBERTExecutor:
    """Clean reviews, fine-tune ModernBERT, and return measured classification metrics."""

    dataset_name = "McAuley-Lab/Amazon-Reviews-2023"
    dataset_config = "raw_review_Automotive"

    def run(self, configuration: dict[str, Any], experiment_id: str) -> ExperimentResult:
        try:
            import numpy as np
            from datasets import Dataset, load_dataset
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            from sklearn.model_selection import train_test_split
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
        except ImportError:
            return ExperimentResult(experiment_id=experiment_id, success=False, error="NLP execution requires datasets, transformers, accelerate, torch, and scikit-learn.")

        try:
            frame, provenance = self._load_reviews(configuration, DatasetVersioner())
            max_samples = int(configuration.get("max_samples", 5000))
            frame = frame.head(max_samples).copy()
            if len(frame) < 20 or frame["label"].nunique() < 2:
                raise ValueError("At least 20 cleaned reviews across two labels are required.")
            train_frame, eval_frame = train_test_split(
                frame, test_size=float(configuration.get("validation_fraction", 0.2)),
                random_state=int(configuration.get("seed", 42)), stratify=frame["label"],
            )
            augmentation = configuration.get("nlp_augmentation", {})
            if augmentation.get("enabled"):
                train_frame = train_frame.copy()
                random_state = random.Random(int(configuration.get("seed", 42)))
                probability = float(augmentation.get("probability", 0.0))
                deletion_probability = float(augmentation.get("deletion_probability", 0.1))
                train_frame["text"] = [self._augment_text(text, random_state, probability, deletion_probability)
                                       for text in train_frame["text"]]
            model_id = str(configuration.get("model", "answerdotai/ModernBERT-base"))
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            max_length = int(configuration.get("max_length", 256))

            def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
                return tokenizer(batch["text"], truncation=True, max_length=max_length)

            train_dataset = Dataset.from_pandas(train_frame[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
            eval_dataset = Dataset.from_pandas(eval_frame[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
            output_dir = Path(str(configuration.get("output_dir", "artifacts/nlp"))).expanduser().resolve() / experiment_id
            output_dir.mkdir(parents=True, exist_ok=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=int(frame["label"].nunique()))

            def compute_metrics(prediction: Any) -> dict[str, float]:
                labels = prediction.label_ids
                predictions = np.argmax(prediction.predictions, axis=-1)
                return {
                    "accuracy": float(accuracy_score(labels, predictions)),
                    "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
                    "precision_weighted": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
                    "recall_weighted": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
                }

            epochs = float(configuration.get("epochs", 1))
            if not 0 < epochs <= float(configuration.get("max_epochs", 5)):
                raise ValueError("epochs must be greater than zero and not exceed max_epochs (default 5).")
            args = TrainingArguments(
                output_dir=str(output_dir), num_train_epochs=epochs,
                per_device_train_batch_size=int(configuration.get("batch_size", 8)),
                per_device_eval_batch_size=int(configuration.get("eval_batch_size", 16)),
                learning_rate=float(configuration.get("learning_rate", 2e-5)),
                weight_decay=float(configuration.get("weight_decay", 0.01)),
                eval_strategy="epoch", save_strategy="no", logging_strategy="epoch",
                report_to=[], seed=int(configuration.get("seed", 42)),
            )
            trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                              processing_class=tokenizer, compute_metrics=compute_metrics)
            trainer.train()
            metrics = {str(key).removeprefix("eval_"): float(value) for key, value in trainer.evaluate().items()
                       if isinstance(value, (int, float))}
            output = trainer.predict(eval_dataset)
            predicted = output.predictions.argmax(axis=-1).tolist()
            prediction_path = output_dir / "predictions.json"
            prediction_path.write_text(json.dumps([
                {"text": text, "actual_rating": int(label) + 1, "predicted_rating": int(pred) + 1}
                for text, label, pred in zip(eval_frame["text"].head(100), eval_frame["label"].head(100), predicted[:100])
            ], indent=2))
            explanation_path = output_dir / "token_explanation.json"
            if not eval_frame.empty:
                explanation = self._explain_tokens(model, tokenizer, str(eval_frame.iloc[0]["text"]), max_tokens=32)
                explanation_path.write_text(json.dumps(explanation, indent=2))
            trainer.save_model(str(output_dir / "model"))
            return ExperimentResult(experiment_id=experiment_id, parameters={**configuration, **provenance, "model": model_id},
                                    metrics=metrics, artifacts=[str(prediction_path), str(explanation_path), str(output_dir / "model")], success=True)
        except Exception as exc:
            return ExperimentResult(experiment_id=experiment_id, parameters=configuration, success=False, error=str(exc))

    def _load_reviews(self, configuration: dict[str, Any], versioner: DatasetVersioner):
        import pandas as pd
        local_path = configuration.get("dataset_path")
        if local_path:
            path = Path(str(local_path)).expanduser().resolve()
            if path.suffix == ".csv":
                raw = pd.read_csv(path)
            elif path.suffix in {".json", ".jsonl"}:
                raw = pd.read_json(path, lines=path.suffix == ".jsonl")
            else:
                raw = pd.read_parquet(path)
            provenance = versioner.identify(path, "Amazon Reviews 2023 Automotive").as_dict()
        else:
            from datasets import load_dataset
            raw = load_dataset(self.dataset_name, self.dataset_config, split="full", streaming=False).to_pandas()
            provenance = {"dataset_name": "Amazon Reviews 2023 Automotive", "dataset_source": self.dataset_name,
                          "dataset_config": self.dataset_config, "dvc_revision": "remote-source"}
        text_column = next((name for name in ("text", "reviewText", "title") if name in raw.columns), None)
        rating_column = next((name for name in ("rating", "overall", "stars") if name in raw.columns), None)
        if text_column is None or rating_column is None:
            raise ValueError("Automotive reviews must contain text/reviewText and rating/overall columns.")
        cleaned = raw[[text_column, rating_column]].rename(columns={text_column: "text", rating_column: "rating"}).dropna()
        cleaned["text"] = cleaned["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce")
        cleaned = cleaned[(cleaned["text"].str.len() >= 3) & (cleaned["rating"].between(1, 5))].drop_duplicates("text")
        cleaned["label"] = cleaned["rating"].astype(int) - 1
        return cleaned, provenance

    @staticmethod
    def _explain_tokens(model: Any, tokenizer: Any, text: str, max_tokens: int) -> dict[str, Any]:
        """Measure real token importance by masking one token at a time."""
        import torch

        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        model.eval()
        with torch.no_grad():
            baseline_probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0]
        predicted_class = int(torch.argmax(baseline_probabilities).item())
        mask_id = tokenizer.mask_token_id or tokenizer.unk_token_id
        attributions = []
        for position, token_id in enumerate(encoded["input_ids"][0].tolist()):
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token in tokenizer.all_special_tokens:
                continue
            masked = {key: value.clone() for key, value in encoded.items()}
            masked["input_ids"][0, position] = mask_id
            with torch.no_grad():
                masked_probability = torch.softmax(model(**masked).logits, dim=-1)[0, predicted_class].item()
            attributions.append({"token": token, "importance": round(float(baseline_probabilities[predicted_class].item() - masked_probability), 6)})
        return {
            "text": text,
            "predicted_rating": predicted_class + 1,
            "predicted_probability": round(float(baseline_probabilities[predicted_class].item()), 6),
            "method": "leave-one-token-out occlusion",
            "tokens": sorted(attributions, key=lambda item: abs(item["importance"]), reverse=True),
        }

    @staticmethod
    def _augment_text(text: str, random_state: random.Random, probability: float, deletion_probability: float) -> str:
        """Apply a reproducible word-deletion augmentation to training text only."""
        if not 0.0 <= probability <= 1.0 or not 0.0 <= deletion_probability < 1.0:
            raise ValueError("NLP augmentation probabilities must be in [0, 1), with application probability in [0, 1].")
        words = text.split()
        if len(words) < 3 or random_state.random() >= probability:
            return text
        retained = [word for word in words if random_state.random() >= deletion_probability]
        return " ".join(retained if len(retained) >= 2 else words)
