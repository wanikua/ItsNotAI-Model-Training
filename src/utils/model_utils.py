from typing import List, Sequence
import torch
from PIL.Image import Image as PILImage

from src.models.model_api import HfModelOutput


def sanitize_label(labels: list[str]) -> list[str]:
    """
    Convert a free-form label that means “real / human” or “fake / AI”
    into the canonical strings ``"real"`` or ``"fake"``.
    """
    _REAL  = {"real", "human", "hum"}
    _FAKE  = {"fake", "ai"}

    def _sanitize(label: str):
        normalized = label.strip().lower()
        if normalized in _REAL:
            return "real"
        if normalized in _FAKE:
            return "fake"
        raise ValueError(f"Unrecognized label: {label!r}")
    
    return [_sanitize(l) for l in labels]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BatchableMixin:
    def predict_batch(
        self,
        images: Sequence[PILImage],
        *,
        batch_size: int = 8,
        with_probs: bool = False,
    ) -> List["HfModelOutput"]: 
        """
        Batched version of `.predict()`.
        Works for any HuggingFace vision model or custom timm model.

        Parameters
        ----------
        images : list[PIL.Image]
            Images in **RGB** mode.
        batch_size : int
            How many images per forward-pass.
        with_probs : bool
            Return probability vectors as well.

        Returns
        -------
        list[HfModelOutput]
            One output per input image, in order.
        """
        device = self.device
        out_list = []

        # mini-batch the input iterable
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]

            # --- HuggingFace models have a Processor/FeatureExtractor ----
            if hasattr(self, "processor"):
                inputs = self.processor(chunk, return_tensors="pt").to(device)
            else:  # Dafilab model (timm + manual transform)
                tensors = [self.transform(im) for im in chunk]
                inputs = torch.stack(tensors).to(device)

            with torch.no_grad():
                logits = self.model(**inputs).logits if isinstance(inputs, dict) else self.model(inputs)
                probs  = torch.softmax(logits, dim=-1)

            top_idx = probs.argmax(-1).tolist()
            probs   = probs.tolist()

            for idx, img_probs in zip(top_idx, probs):
                if hasattr(self, "labels"):
                    label = sanitize_label(self.labels[idx])
                elif hasattr(self, "label_mapping"):
                    label = sanitize_label(self.label_mapping[idx])
                else:  # SigLIP config
                    label = sanitize_label(self.model.config.id2label[idx])

                if with_probs:
                    out_list.append(HfModelOutput(label=label, probs=img_probs))
                else:
                    out_list.append(HfModelOutput(label=label))

        return out_list
