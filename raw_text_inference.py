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
		self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
		self.model.to(device)
		self.model.eval()

	def generate_from_chinese_text(self, text, max_pred_len=None):
		"""Generate from raw Chinese text."""
		if self.cf["dataset"] != "BSC":
			raise ValueError("generate_from_chinese_text requires a BSC config.")
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
			)

	def generate_from_english_text(self, text, max_pred_len=None):
		"""Generate from raw English text."""
		if self.cf["dataset"] != "celer":
			raise ValueError("generate_from_english_text requires a CELER config.")
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
			)
