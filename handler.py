from pathlib import Path
import torch

from Eyettention import Eyettention
from Eyettention import EyettentionRawTextInference

class EndpointHandler:
    def __init__(self, path: str = ""):
        self.path = Path(path) if path else Path(__file__).resolve().parent
        self.inference = None
        self.dataset = None
        self.checkpoint_path = None
        self.device = None


    def __call__(self, data):
        inputs = data.get("inputs", data)  
        parameters = data.get("parameters", {})
        dataset = parameters.get("dataset", "BSC") # default = BSC
        max_pred_len = parameters.get("max_pred_len", 60)
        previous_scanpath = parameters.get("previous_scanpath")

        inference = self._get_inference(dataset, parameters)

        if isinstance(inputs, str):
            if dataset == "BSC":
                scanpath, density = inference.generate_from_chinese_text(
                    text = inputs,
                    max_pred_len = max_pred_len,
                    previous_scanpath = previous_scanpath
                )

            elif dataset == "celer":
                scanpath, density = inference.generate_from_english_text(
                    text = inputs,
                    max_pred_len = max_pred_len,
                    previous_scanpath = previous_scanpath
                )
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")

        elif isinstance(inputs, dict):
            with torch.no_grad():
                scanpath, density = inference.model.scanpath_generation(
                    sn_emd=inputs["sn_input_ids"],
                    sn_mask=inputs["sn_mask"],
                    word_ids_sn=inputs.get("word_ids_sn"),
                    sn_word_len=inputs["sn_word_len"],
                    le=inference.label_encoder,
                    max_pred_len=max_pred_len or inference.cf["max_pred_len"],
                    previous_scanpath=previous_scanpath
                )
        else:
            raise ValueError("'inputs' must be raw text or model-ready tensors.")
        
        return {
            "scanpath": scanpath.tolist(),
            "density_steps": len(density),
        }

    def _get_inference(self, dataset, parameters):
        device = parameters.get("device", "cpu")
        checkpoint_path = parameters.get("checkpoint_path") or self._default_checkpoint_path(dataset)

        if self.inference is None or self.dataset != dataset or self.checkpoint_path != checkpoint_path or self.device != device:
            self.inference = EyettentionRawTextInference(checkpoint_path=str(checkpoint_path), dataset=dataset, device=device) 
            self.dataset = dataset
            self.checkpoint_path = checkpoint_path
            self.device = device

        return self.inference

    def _default_checkpoint_path(self, dataset):
        if dataset == "BSC":
            return self.path / "results" / "BSC" / "Eyettention_chinese.pth"
        if dataset == "celer":
            return self.path / "results" / "CELER" / "Eyettention_english.pth"
        raise ValueError(f"Unsupported dataset: {dataset}")
