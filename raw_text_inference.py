import torch
from transformers import BertTokenizer, BertTokenizerFast

from Eyettention.model import Eyettention
from Eyettention.utils import build_bsc_config, build_celer_config, build_label_encoder, text_to_bsc_inputs, text_to_celer_inputs


class EyettentionRawTextInference:
	def __init__(self, checkpoint_path, dataset="BSC", cf=None, device="cpu"):
		if cf is not None:
			self.cf = cf
		elif dataset == "BSC":
			self.cf = build_bsc_config()
		elif dataset == "celer":
			self.cf = build_celer_config()
		else:
			raise ValueError(f"Unsupported dataset: {dataset}")
		
		self.device = device
		if self.cf["dataset"] == "celer":
			self.tokenizer = BertTokenizerFast.from_pretrained(self.cf["model_pretrained"])
		else:
			self.tokenizer = BertTokenizer.from_pretrained(self.cf["model_pretrained"])
		self.label_encoder = build_label_encoder(self.cf)

		self.model = Eyettention(self.cf)
		state_dict = torch.load(checkpoint_path, map_location=device)
		if "encoder.embeddings.position_ids" not in self.model.state_dict(): # Based on transformers version
			state_dict.pop("encoder.embeddings.position_ids", None)
		self.model.load_state_dict(state_dict)
		self.model.to(device)
		self.model.eval()

	def generate_from_chinese_text(self, text, max_pred_len=None, previous_scanpath=None):
		"""Generate from raw Chinese text."""
		if self.cf["dataset"] != "BSC":
			raise ValueError("generate_from_chinese_text requires a BSC config.")
		if max_pred_len is not None and max_pred_len <= 0:
			raise ValueError("max_pred_len must be positive.")
		if not isinstance(text, str):
			raise TypeError("text must be a string.")
		if not text.strip():
			raise ValueError("text must not be empty.")
		
		sn_input_ids, sn_mask, sn_word_len = text_to_bsc_inputs(
			sn_str=text,
			tokenizer=self.tokenizer,
			cf=self.cf,
			device=self.device,
		)

		with torch.no_grad():
			return self.model.scanpath_generation(
				sn_emd=sn_input_ids,
				sn_mask=sn_mask,
				word_ids_sn=None,
				sn_word_len=sn_word_len,
				le=self.label_encoder,
				max_pred_len=max_pred_len or self.cf["max_pred_len"],
				previous_scanpath=previous_scanpath
			)

	def generate_from_english_text(self, text, max_pred_len=None, previous_scanpath=None):
		"""Generate from raw English text."""
		if self.cf["dataset"] != "celer":
			raise ValueError("generate_from_english_text requires a CELER config.")
		if max_pred_len is not None and max_pred_len <= 0:
			raise ValueError("max_pred_len must be positive.")
		if not isinstance(text, str):
			raise TypeError("text must be a string.")
		if not text.strip():
			raise ValueError("text must not be empty.")
		
		sn_input_ids, sn_mask, word_ids_sn, sn_word_len = text_to_celer_inputs(
			sn_str=text,
			tokenizer=self.tokenizer,
			cf=self.cf,
			device=self.device,
		)

		with torch.no_grad():
			return self.model.scanpath_generation(
				sn_emd=sn_input_ids,
				sn_mask=sn_mask,
				word_ids_sn=word_ids_sn,
				sn_word_len=sn_word_len,
				le=self.label_encoder,
				max_pred_len=max_pred_len or self.cf["max_pred_len"],
				previous_scanpath=previous_scanpath
			)
