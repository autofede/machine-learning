import torch
from transformers import MarianMTModel, MarianTokenizer
import os
import json
from typing import List, Dict, Optional, Union
import time
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

@dataclass
class TranslationConfig:
	"""Translation configuration class"""
	model_name: str = "Helsinki-NLP/opus-mt-en-fr"
	max_length: int = 512
	num_beams: int = 4
	temperature: float = 1.0
	do_sample: bool = False
	device: str = "auto"
	batch_size: int = 8
	# cache_dir: Optional[str] = "./model_cache"

class MarianTranslator:
	"""Simplified Marian Transformer translator"""

	def __init__(self, config: TranslationConfig = None):
		self.config = config or TranslationConfig()
		self.device = self._setup_device()
		self.tokenizer = None
		self.model = None
		self._load_model()

		# Cache for frequently used translations
		self.translation_cache = {}

	def _setup_device(self) -> torch.device:
		"""Setup computation device"""
		if self.config.device == "auto":
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		else:
			device = torch.device(self.config.device)

		print(f"Using device: {device}")
		if device.type == 'cuda':
			print(f"GPU: {torch.cuda.get_device_name()}")
			total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
			print(f"GPU Memory: {total_memory:.1f} GB")

		return device

	def _load_model(self):
		"""Load pre-trained model"""
		print(f"Loading Marian model: {self.config.model_name}")

		try:
			# Load tokenizer and model
			self.tokenizer = MarianTokenizer.from_pretrained(
				self.config.model_name,
				# cache_dir=self.config.cache_dir
			)
			self.model = MarianMTModel.from_pretrained(
				self.config.model_name,
				# cache_dir=self.config.cache_dir
			)

			# Move to specified device
			self.model.to(self.device)
			self.model.eval()  # Set to evaluation mode

			print("Model loaded successfully!")
			print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

		except Exception as e:
			print(f"Error loading model: {e}")
			raise

	def translate_single(self, text: str, **kwargs) -> str:
		"""Translate a single sentence"""
		if not text or not text.strip():
			return ""

		# Check cache
		# cache_key = text.strip()
		# if cache_key in self.translation_cache:
		# 	return self.translation_cache[cache_key]

		try:
			# Merge configuration parameters
			generation_config = {
				'max_length': kwargs.get('max_length', self.config.max_length),
				'num_beams': kwargs.get('num_beams', self.config.num_beams),
				'temperature': kwargs.get('temperature', self.config.temperature),
				'do_sample': kwargs.get('do_sample', self.config.do_sample),
				'early_stopping': True,
				'pad_token_id': self.tokenizer.pad_token_id,
				'eos_token_id': self.tokenizer.eos_token_id,
				'no_repeat_ngram_size': 3,
				'length_penalty': 1.0
			}

			# Tokenize input
			inputs = self.tokenizer(
				text.strip(),
				return_tensors='pt',
				padding=True,
				truncation=True,
				max_length=self.config.max_length
			)
			inputs = inputs.to(self.device)

			# Generate translation
			with torch.no_grad():
				outputs = self.model.generate(**inputs, **generation_config)

			# Decode output
			translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
			translation = translation.strip()

			# Cache result
			# self.translation_cache[cache_key] = translation

			return translation

		except Exception as e:
			print(f"Translation error for '{text}': {e}")
			return f"[Translation Error: {text}]"

	def translate(self, input_data: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
		"""Unified translation interface"""
		if isinstance(input_data, str):
			return self.translate_single(input_data, **kwargs)
		else:
			raise ValueError("Input must be string or list of strings")

def main():
	"""Main function to translate predefined test sentences"""
	print("English to French Translation")
	print("=" * 50)

	# Initialize translator
	config = TranslationConfig(
		model_name="Helsinki-NLP/opus-mt-en-fr",
		max_length=256,
		num_beams=4,
		batch_size=8
	)

	translator = MarianTranslator(config)

	# Test sentences to translate
	test_sentences = [
		"Hello, how are you?",
		"I love learning languages.",
		"The weather is beautiful today.",
		"Thank you for your help.",
		"What time is it?",
		"I am going to the store.",
		"Can you please repeat that?",
		"This book is very interesting.",
		"We are planning a trip next summer.",
		"She doesn't like spicy food."
	]

	print("\nTranslating test sentences...")
	print("-" * 40)

	# Record total start time
	# total_start_time = time.time()

	# Translate sentences one by one
	for i, sentence in enumerate(test_sentences, 1):
		# Translate single sentence
		translation = translator.translate_single(sentence)

		# Display result
		print(f"{i:2d}. English: {sentence}")
		print(f"    French:  {translation}")
		print()

if __name__ == "__main__":
	main()