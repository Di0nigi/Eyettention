import functools
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch


warnings.simplefilter("ignore")

EYETTENTION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EYETTENTION_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

BSC_CHECKPOINT = EYETTENTION_ROOT / "results" / "BSC" / "Eyettention_chinese.pth"
CELER_CHECKPOINT = EYETTENTION_ROOT / "results" / "CELER" / "Eyettention_english.pth"


def _import_utils():
	try:
		from Eyettention.utils import (
			build_bsc_config,
			build_celer_config,
			build_label_encoder,
			text_to_bsc_inputs,
			text_to_celer_inputs,
		)
	except ImportError as exc:
		raise unittest.SkipTest(f"Eyettention utils dependencies are unavailable: {exc}") from exc

	return (
		build_bsc_config,
		build_celer_config,
		build_label_encoder,
		text_to_bsc_inputs,
		text_to_celer_inputs,
	)


def _load_tokenizer(tokenizer_cls, model_name):
	try:
		return tokenizer_cls.from_pretrained(model_name)
	except OSError as exc:
		raise unittest.SkipTest(f"{model_name} is not available in the local Hugging Face cache") from exc


@functools.lru_cache(maxsize=None)
def _raw_text_runner(dataset, checkpoint_path):
	from Eyettention.raw_text_inference import EyettentionRawTextInference

	if not Path(checkpoint_path).exists():
		raise unittest.SkipTest(f"Missing checkpoint: {checkpoint_path}")

	try:
		return EyettentionRawTextInference(
			checkpoint_path=str(checkpoint_path),
			dataset=dataset,
			device="cpu",
		)
	except OSError as exc:
		raise unittest.SkipTest(f"Pretrained model for {dataset} is not available locally") from exc
	except ImportError as exc:
		raise unittest.SkipTest(f"Raw text dependency for {dataset} is unavailable: {exc}") from exc


class ConfigAndInputTests(unittest.TestCase):

	def test_text_to_bsc_inputs_returns_model_ready_tensors(self):
		from transformers import BertTokenizer

		build_bsc_config, _, _, text_to_bsc_inputs, _ = _import_utils()
		cf = build_bsc_config(max_pred_len=5)
		tokenizer = _load_tokenizer(BertTokenizer, cf["model_pretrained"])

		sn_input_ids, sn_mask, sn_word_len = text_to_bsc_inputs(
			"中国选手在男子滑雪比赛中有望蝉联冠军",
			tokenizer,
			cf,
			device="cpu",
		)

		self.assertEqual(tuple(sn_input_ids.shape), (1, cf["max_sn_len"]))
		self.assertEqual(tuple(sn_mask.shape), (1, cf["max_sn_len"]))
		self.assertEqual(tuple(sn_word_len.shape), (1, cf["max_sn_len"]))
		self.assertTrue(torch.is_floating_point(sn_mask))
		self.assertTrue(torch.isfinite(sn_word_len).all())

	def test_text_to_celer_inputs_returns_model_ready_tensors(self):
		from transformers import BertTokenizerFast

		_, build_celer_config, _, _, text_to_celer_inputs = _import_utils()
		cf = build_celer_config(max_pred_len=5)
		tokenizer = _load_tokenizer(BertTokenizerFast, cf["model_pretrained"])

		sn_input_ids, sn_mask, word_ids_sn, sn_word_len = text_to_celer_inputs(
			"He said BankEast's offer appears to be \"attractive to the bank's shareholders.\"",
			tokenizer,
			cf,
			device="cpu",
		)

		self.assertEqual(tuple(sn_input_ids.shape), (1, cf["max_sn_token"]))
		self.assertEqual(tuple(sn_mask.shape), (1, cf["max_sn_token"]))
		self.assertEqual(tuple(word_ids_sn.shape), (1, cf["max_sn_token"]))
		self.assertEqual(tuple(sn_word_len.shape), (1, cf["max_sn_len"]))
		self.assertTrue(torch.is_floating_point(sn_mask))
		self.assertTrue(torch.isfinite(sn_word_len).all())
		self.assertGreaterEqual(np.nanmax(word_ids_sn.numpy()), 1)


class RawTextInferenceSmokeTests(unittest.TestCase):
	def test_chinese_raw_text_generation_smoke(self):
		runner = _raw_text_runner("BSC", BSC_CHECKPOINT)
		torch.manual_seed(0)

		scanpath, density = runner.generate_from_chinese_text(
			"中国经济发展很快。",
			max_pred_len=5,
		)

		self.assertEqual(tuple(scanpath.shape), (1, 5))
		self.assertEqual(len(density), 4)
		self.assertEqual(scanpath[0, 0].item(), 0)

	def test_english_raw_text_generation_smoke(self):
		runner = _raw_text_runner("celer", CELER_CHECKPOINT)
		torch.manual_seed(0)

		scanpath, density = runner.generate_from_english_text(
			"The quick brown fox jumps.",
			max_pred_len=5,
		)

		self.assertEqual(tuple(scanpath.shape), (1, 5))
		self.assertEqual(len(density), 4)
		self.assertEqual(scanpath[0, 0].item(), 0)

	def test_previous_scanpath_is_replayed_before_sampling(self):
		runner = _raw_text_runner("BSC", BSC_CHECKPOINT)
		torch.manual_seed(0)

		scanpath, _ = runner.generate_from_chinese_text(
			"中国选手在男子滑雪比赛中有望蝉联冠军",
			max_pred_len=3,
			previous_scanpath=[0, 1, 2],
		)

		self.assertEqual(scanpath[0, :3].tolist(), [0, 1, 2])
		self.assertEqual(scanpath.shape[1], 6)

	def test_dataset_specific_methods_guard_against_wrong_config(self):
		from Eyettention.raw_text_inference import EyettentionRawTextInference

		bsc_runner = object.__new__(EyettentionRawTextInference)
		bsc_runner.cf = {"dataset": "celer"}
		with self.assertRaises(ValueError):
			bsc_runner.generate_from_chinese_text("中国选手在男子滑雪比赛中有望蝉联冠军")

		celer_runner = object.__new__(EyettentionRawTextInference)
		celer_runner.cf = {"dataset": "BSC"}
		with self.assertRaises(ValueError):
			celer_runner.generate_from_english_text("He said BankEast's offer appears to be \"attractive to the bank's shareholders.\"")

	def test_text_is_none(self):
		from Eyettention.raw_text_inference import EyettentionRawTextInference

		bsc_runner = object.__new__(EyettentionRawTextInference)
		bsc_runner.cf = {"dataset": "celer"}
		with self.assertRaises(ValueError):
			bsc_runner.generate_from_chinese_text(None)

		celer_runner = object.__new__(EyettentionRawTextInference)
		celer_runner.cf = {"dataset": "BSC"}
		with self.assertRaises(ValueError):
			celer_runner.generate_from_english_text(None)

	def test_text_is_empty(self):
		from Eyettention.raw_text_inference import EyettentionRawTextInference

		bsc_runner = object.__new__(EyettentionRawTextInference)
		bsc_runner.cf = {"dataset": "celer"}
		with self.assertRaises(ValueError):
			bsc_runner.generate_from_chinese_text(" ")

		celer_runner = object.__new__(EyettentionRawTextInference)
		celer_runner.cf = {"dataset": "BSC"}
		with self.assertRaises(ValueError):
			celer_runner.generate_from_english_text(" ")


if __name__ == "__main__":
	unittest.main()
