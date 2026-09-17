import torch
from transformers import BertTokenizer

from Eyettention.model import Eyettention
from Eyettention.utils import build_bsc_config, build_label_encoder, text_to_bsc_inputs


class EyettentionPredictor:
	def __init__(self, checkpoint_path, cf=None, device="cpu"):
		self.cf = cf or build_bsc_config()
		self.device = device
		self.tokenizer = BertTokenizer.from_pretrained(self.cf["model_pretrained"])
		self.label_encoder = build_label_encoder(self.cf)

		self.model = Eyettention(self.cf)
		self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
		self.model.to(device)
		self.model.eval()

	def generate_from_text(self, text, max_pred_len=None):
		"""Generate from raw text."""
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

	def generate_from_tokenized_text(
		self,
		sn_input_ids,
		sn_mask,
		sn_word_len,
		word_ids_sn=None,
		max_pred_len=None,
	):
		"""Generate from tokenised text."""
		sn_input_ids = sn_input_ids.to(self.device)
		sn_mask = sn_mask.to(self.device)
		sn_word_len = sn_word_len.to(self.device)

		if word_ids_sn is not None:
			word_ids_sn = word_ids_sn.to(self.device)

		with torch.no_grad():
			return self.model.scanpath_generation(
				sn_emd=sn_input_ids,
				sn_mask=sn_mask,
				word_ids_sn=word_ids_sn,
				sn_word_len=sn_word_len,
				le=self.label_encoder,
				max_pred_len=max_pred_len or self.cf["max_pred_len"],
			)
