"""Real Ultralytics YOLO execution for IDD-format object-detection datasets."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from ingestion.dataset_versioning import DatasetVersioner
from utils.models import ExperimentResult


class IDDYOLOExecutor:
    """Train, evaluate, and predict with YOLO using a user-supplied IDD YAML."""

    def run(self, configuration: dict[str, Any], experiment_id: str) -> ExperimentResult:
        try:
            from ultralytics import YOLO
            import yaml
        except ImportError as exc:
            return ExperimentResult(
                experiment_id=experiment_id,
                success=False,
                error="YOLO execution requires 'ultralytics' and PyYAML. Install project requirements first.",
            )

        data_yaml = Path(str(configuration.get("data_yaml", ""))).expanduser().resolve()
        if not data_yaml.is_file():
            return ExperimentResult(experiment_id=experiment_id, success=False, error="CV execution requires an existing YOLO data_yaml path.")

        try:
            data = yaml.safe_load(data_yaml.read_text()) or {}
            root = Path(data.get("path") or data_yaml.parent)
            if not root.is_absolute():
                root = (data_yaml.parent / root).resolve()
            provenance = DatasetVersioner().identify(root, "IDD YOLO").as_dict()
            output_root = Path(str(configuration.get("output_dir", "artifacts/cv"))).expanduser().resolve()
            run_name = str(configuration.get("run_name") or experiment_id)
            epochs = int(configuration.get("epochs", 5))
            if not 1 <= epochs <= int(configuration.get("max_epochs", 50)):
                raise ValueError("epochs must be between 1 and max_epochs (default 50).")
            model_name = str(configuration.get("model", "yolo11n.pt"))
            model = YOLO(model_name)
            started = perf_counter()
            model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=int(configuration.get("imgsz", 640)),
                batch=configuration.get("batch", "auto"),
                device=configuration.get("device", "cpu"),
                project=str(output_root),
                name=run_name,
                exist_ok=True,
                seed=int(configuration.get("seed", 42)),
                workers=int(configuration.get("workers", 2)),
                patience=int(configuration.get("patience", 10)),
                degrees=float(configuration.get("degrees", 0.0)),
                translate=float(configuration.get("translate", 0.0)),
                scale=float(configuration.get("scale", 0.0)),
                fliplr=float(configuration.get("fliplr", 0.0)),
            )
            validation = model.val(data=str(data_yaml), split=str(configuration.get("eval_split", "val")))
            metrics: dict[str, float] = {}
            for key, value in validation.results_dict.items():
                try:
                    metrics[key.replace("/", "_").replace("(", "_").replace(")", "")] = float(value)
                except (TypeError, ValueError):
                    continue
            metrics["training_seconds"] = perf_counter() - started

            run_dir = output_root / run_name
            artifacts = [str(item) for item in (run_dir / "weights").glob("*.pt")]
            artifacts.extend(str(item) for item in (run_dir / "results.csv", run_dir / "results.png") if item.exists())
            prediction_source = configuration.get("prediction_source")
            if prediction_source:
                predictions = model.predict(source=str(prediction_source), save=True, project=str(output_root), name=f"{run_name}-predictions", exist_ok=True)
                summary = []
                explanation_images = []
                for index, result in enumerate(predictions):
                    boxes = result.boxes
                    detections = []
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls.item())
                            detections.append({
                                "class": result.names[class_id],
                                "confidence": float(box.conf.item()),
                                "bbox_xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                            })
                    image_path = run_dir / f"explanation_{index}.jpg"
                    try:
                        import cv2
                        cv2.imwrite(str(image_path), result.plot())
                        explanation_images.append(str(image_path))
                    except ImportError:
                        pass
                    summary.append({"source": str(result.path), "detections": detections,
                                    "explanation": "Each box is an actual YOLO prediction; confidence reflects the model score."})
                prediction_file = run_dir / "prediction_explanations.json"
                prediction_file.parent.mkdir(parents=True, exist_ok=True)
                prediction_file.write_text(json.dumps(summary, indent=2))
                artifacts.append(str(prediction_file))
                artifacts.extend(explanation_images)

            return ExperimentResult(
                experiment_id=experiment_id,
                parameters={**configuration, **provenance, "model": model_name},
                metrics=metrics,
                artifacts=artifacts,
                success=True,
            )
        except Exception as exc:
            return ExperimentResult(experiment_id=experiment_id, parameters=configuration, success=False, error=str(exc))
