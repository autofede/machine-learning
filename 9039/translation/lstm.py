import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import os
import requests
import zipfile
from tqdm import tqdm
import random
import re
import math
from collections import Counter


def set_seed(seed=42):
	"""Set random seed for reproducibility"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


set_seed(42)


def download_better_opus_data():
	"""Download multiple OPUS datasets for better quality"""
	datasets = [
		{
			"name": "UN Corpus (formal, high quality)",
			"url": "https://object.pouta.csc.fi/OPUS-UN/v20090831/moses/en-fr.txt.zip",
			"filename": "un-en-fr.txt.zip",
			"en_file": "UN.en-fr.en",
			"fr_file": "UN.en-fr.fr"
		},
		{
			"name": "MultiUN (UN documents)",
			"url": "https://object.pouta.csc.fi/OPUS-MultiUN/v1/moses/en-fr.txt.zip",
			"filename": "multiun-en-fr.txt.zip",
			"en_file": "MultiUN.en-fr.en",
			"fr_file": "MultiUN.en-fr.fr"
		},
		{
			"name": "News Commentary (news articles)",
			"url": "https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/en-fr.txt.zip",
			"filename": "news-en-fr.txt.zip",
			"en_file": "News-Commentary.en-fr.en",
			"fr_file": "News-Commentary.en-fr.fr"
		}
	]

	downloaded_files = []

	for dataset in datasets:
		filename = dataset["filename"]
		if not os.path.exists(filename):
			print(f"Downloading {dataset['name']}...")
			try:
				response = requests.get(dataset["url"], stream=True, timeout=60)
				response.raise_for_status()
				total_size = int(response.headers.get('content-length', 0))

				with open(filename, 'wb') as f, tqdm(
						desc=filename,
						total=total_size,
						unit='iB',
						unit_scale=True,
						unit_divisor=1024,
				) as pbar:
					for chunk in response.iter_content(chunk_size=8192):
						size = f.write(chunk)
						pbar.update(size)
				print(f"Downloaded {dataset['name']}")
			except Exception as e:
				print(f"Failed to download {dataset['name']}: {e}")
				continue

		# Extract files
		try:
			print(f"Extracting {filename}...")
			with zipfile.ZipFile(filename, 'r') as zip_ref:
				zip_ref.extractall('.')
			downloaded_files.append((dataset["en_file"], dataset["fr_file"]))
		except Exception as e:
			print(f"Extraction failed for {filename}: {e}")
			continue

	return downloaded_files


def advanced_clean_text(text):
	"""More sophisticated text cleaning"""
	# Convert to lowercase
	text = text.lower()

	# Remove HTML tags
	text = re.sub(r'<[^>]+>', '', text)

	# Remove URLs
	text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

	# Remove email addresses
	text = re.sub(r'\S*@\S*\s?', '', text)

	# Normalize punctuation - use explicit Unicode characters
	text = re.sub(r'[""]', '"', text)  # Smart double quotes
	text = re.sub(r'['']', "'", text)  # Smart single quotes
	text = re.sub(r'[—–]', '-', text)  # Em dash and en dash

	# Remove excessive punctuation
	text = re.sub(r'[^\w\s\.\?\!\,\;\:\'\-\"()]', '', text)

	# Remove multiple spaces and normalize
	text = re.sub(r'\s+', ' ', text)
	text = text.strip()

	return text


def debug_load_single_file(filename, max_lines=10):
	"""Debug function to test file loading"""
	print(f"Debug: Testing file {filename}")
	try:
		with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
			for i, line in enumerate(f):
				if i >= max_lines:
					break
				print(f"Line {i + 1}: {repr(line[:100])}")  # Show first 100 chars
		return True
	except Exception as e:
		print(f"Debug: Error reading {filename}: {e}")
		import traceback
		traceback.print_exc()
		return False


def ultra_simple_clean(text):
	"""Ultra simple cleaning without any regex"""
	if not text:
		return ""

	# Convert to lowercase
	text = text.lower().strip()

	# Only keep basic characters - no regex at all
	result = ""
	for char in text:
		if char.isalpha() or char.isdigit() or char in ' .,!?':
			result += char

	# Simple space normalization
	words = result.split()
	return ' '.join(words)


def load_multiple_datasets_debug(file_pairs, max_samples_per_dataset=1000):
	"""Debug version with extensive logging"""
	all_en_sentences = []
	all_fr_sentences = []

	for en_file, fr_file in file_pairs:
		print(f"\n=== Processing {en_file} and {fr_file} ===")

		# Check if files exist
		if not os.path.exists(en_file):
			print(f"File {en_file} does not exist")
			continue
		if not os.path.exists(fr_file):
			print(f"File {fr_file} does not exist")
			continue

		print(f"Both files exist, testing read access...")

		# Test file reading first
		if not debug_load_single_file(en_file, 3):
			continue
		if not debug_load_single_file(fr_file, 3):
			continue

		try:
			print("Starting actual data loading...")
			en_lines = []
			fr_lines = []

			# Load English file
			print(f"Loading English file: {en_file}")
			with open(en_file, 'r', encoding='utf-8', errors='ignore') as f:
				for i, line in enumerate(f):
					if i >= max_samples_per_dataset:
						break
					if i % 10000 == 0:
						print(f"  Loaded {i} English lines...")

					try:
						cleaned = ultra_simple_clean(line)
						if cleaned and len(cleaned) > 5:
							en_lines.append(cleaned)
					except Exception as e:
						print(f"  Error cleaning English line {i}: {e}")
						continue

			print(f"Loaded {len(en_lines)} English sentences")

			# Load French file
			print(f"Loading French file: {fr_file}")
			with open(fr_file, 'r', encoding='utf-8', errors='ignore') as f:
				for i, line in enumerate(f):
					if i >= max_samples_per_dataset:
						break
					if i % 10000 == 0:
						print(f"  Loaded {i} French lines...")

					try:
						cleaned = ultra_simple_clean(line)
						if cleaned and len(cleaned) > 5:
							fr_lines.append(cleaned)
					except Exception as e:
						print(f"  Error cleaning French line {i}: {e}")
						continue

			print(f"Loaded {len(fr_lines)} French sentences")

			# Pair sentences
			min_lines = min(len(en_lines), len(fr_lines))
			print(f"Pairing {min_lines} sentence pairs...")

			paired_count = 0
			for i in range(min_lines):
				en = en_lines[i]
				fr = fr_lines[i]

				en_words = en.split()
				fr_words = fr.split()

				# Very simple filtering
				if (3 <= len(en_words) <= 30 and
						3 <= len(fr_words) <= 30 and
						len(en) >= 10 and len(fr) >= 10):
					all_en_sentences.append(en)
					all_fr_sentences.append(fr)
					paired_count += 1

			print(f"Added {paired_count} valid sentence pairs from {en_file}")

		except Exception as e:
			print(f"Error loading data from {en_file}: {e}")
			import traceback
			traceback.print_exc()
			continue

	print(f"\nTotal sentences loaded: {len(all_en_sentences)}")
	return all_en_sentences, all_fr_sentences


def build_bpe_like_vocab(sentences, max_vocab=25000, min_freq=2):
	"""Build vocabulary with better subword handling - no regex version"""
	print(f"Building vocabulary from {len(sentences)} sentences...")

	if not sentences:
		print("No sentences provided for vocabulary building!")
		return {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}, {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}

	word_freq = Counter()

	# Count word frequencies
	for i, sentence in enumerate(sentences):
		if i % 10000 == 0:
			print(f"Processing sentence {i}/{len(sentences)}")
		try:
			for word in sentence.split():
				if word.strip():  # Only count non-empty words
					word_freq[word] += 1
		except Exception as e:
			print(f"Error processing sentence {i}: {e}")
			continue

	print(f"Found {len(word_freq)} unique words")

	# Filter by minimum frequency and get most common words
	common_words = [(word, freq) for word, freq in word_freq.most_common()
	                if freq >= min_freq]

	print(f"Words with freq >= {min_freq}: {len(common_words)}")

	# Create vocabulary with special tokens
	word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}

	# Add most frequent words up to max_vocab limit
	words_added = 0
	for word, freq in common_words:
		if len(word2idx) >= max_vocab:
			break
		word2idx[word] = len(word2idx)
		words_added += 1

	idx2word = {idx: word for word, idx in word2idx.items()}

	print(f"Built vocabulary with {len(word2idx)} words")
	print(f"Added {words_added} content words")
	if len(word_freq) > 0:
		coverage = len([w for w, f in common_words[:words_added] if w in word2idx]) / len(word_freq) * 100
		print(f"Vocabulary coverage: {coverage:.1f}%")

	return word2idx, idx2word


class ImprovedTranslationDataset(Dataset):
	"""Improved dataset with better data augmentation"""

	def __init__(self, src_sentences, tgt_sentences, src_word2idx, tgt_word2idx, max_len=40):
		self.src_sentences = src_sentences
		self.tgt_sentences = tgt_sentences
		self.src_word2idx = src_word2idx
		self.tgt_word2idx = tgt_word2idx
		self.max_len = max_len

	def __len__(self):
		return len(self.src_sentences)

	def __getitem__(self, idx):
		src = self.src_sentences[idx]
		tgt = self.tgt_sentences[idx]

		# Convert to indices with UNK handling
		src_indices = [self.src_word2idx.get(word, self.src_word2idx['<UNK>'])
		               for word in src.split()][:self.max_len - 1]
		tgt_indices = [self.tgt_word2idx.get(word, self.tgt_word2idx['<UNK>'])
		               for word in tgt.split()][:self.max_len - 1]

		# Add EOS tokens
		src_indices.append(self.src_word2idx['<EOS>'])
		tgt_indices.append(self.tgt_word2idx['<EOS>'])

		# Create decoder input (with SOS) and target (with EOS)
		tgt_input = [self.tgt_word2idx['<SOS>']] + tgt_indices[:-1]
		tgt_output = tgt_indices

		return {
			'src': torch.tensor(src_indices, dtype=torch.long),
			'tgt_input': torch.tensor(tgt_input, dtype=torch.long),
			'tgt_output': torch.tensor(tgt_output, dtype=torch.long),
			'src_len': len(src_indices),
			'tgt_len': len(tgt_indices)
		}


class MultiHeadAttention(nn.Module):
	"""Multi-head attention mechanism"""

	def __init__(self, d_model, num_heads, dropout=0.1):
		super().__init__()
		assert d_model % num_heads == 0

		self.d_model = d_model
		self.num_heads = num_heads
		self.d_k = d_model // num_heads

		self.w_q = nn.Linear(d_model, d_model)
		self.w_k = nn.Linear(d_model, d_model)
		self.w_v = nn.Linear(d_model, d_model)
		self.w_o = nn.Linear(d_model, d_model)

		self.dropout = nn.Dropout(dropout)
		self.scale = math.sqrt(self.d_k)

	def forward(self, query, key, value, mask=None):
		batch_size = query.size(0)

		# Linear transformations and split into heads
		Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
		K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
		V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

		# Attention
		scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

		if mask is not None:
			# Fix mask dimensions - ensure it's the right shape for multi-head attention
			if mask.dim() == 4:  # [batch_size, 1, 1, seq_len]
				mask = mask.expand(-1, self.num_heads, mask.size(2), -1)
			elif mask.dim() == 5:  # Wrong dimensions, fix it
				mask = mask.squeeze()  # Remove extra dimensions
				if mask.dim() == 3:  # [batch_size, 1, seq_len]
					mask = mask.unsqueeze(1).expand(-1, self.num_heads, mask.size(1), -1)
				elif mask.dim() == 4:  # [batch_size, 1, 1, seq_len]
					mask = mask.expand(-1, self.num_heads, mask.size(2), -1)

			scores = scores.masked_fill(mask == 0, -1e9)

		attention_weights = F.softmax(scores, dim=-1)
		attention_weights = self.dropout(attention_weights)

		context = torch.matmul(attention_weights, V)

		# Concatenate heads
		context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
		output = self.w_o(context)

		return output, attention_weights.mean(dim=1)  # Average across heads


class ImprovedEncoder(nn.Module):
	"""Improved encoder with multi-head attention"""

	def __init__(self, vocab_size, embed_size, hidden_size, num_layers=3, num_heads=8, dropout=0.1):
		super().__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
		self.pos_encoding = self.create_pos_encoding(5000, embed_size)
		self.dropout = nn.Dropout(dropout)

		# LSTM layers
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
		                    batch_first=True, bidirectional=True, dropout=dropout)

		# Project bidirectional LSTM output to match attention dimension
		self.output_projection = nn.Linear(hidden_size * 2, hidden_size * 2)

		# Multi-head attention with correct dimensions
		self.self_attention = MultiHeadAttention(hidden_size * 2, num_heads, dropout)
		self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
		self.layer_norm2 = nn.LayerNorm(hidden_size * 2)

		# Feed forward
		self.feed_forward = nn.Sequential(
			nn.Linear(hidden_size * 2, hidden_size * 4),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size * 4, hidden_size * 2)
		)

	def create_pos_encoding(self, max_len, embed_size):
		pe = torch.zeros(max_len, embed_size)
		position = torch.arange(0, max_len).unsqueeze(1).float()

		div_term = torch.exp(torch.arange(0, embed_size, 2).float() *
		                     -(math.log(10000.0) / embed_size))

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		return pe.unsqueeze(0)

	def forward(self, x):
		batch_size, seq_len = x.size()

		# Create mask for padding - fix dimensions
		mask = (x != 0)  # [batch_size, seq_len]
		# For self-attention, we need [batch_size, seq_len, seq_len]
		mask = mask.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, seq_len]
		mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len] for multi-head attention

		# Embedding with positional encoding
		embedded = self.embedding(x)
		if embedded.size(1) <= self.pos_encoding.size(1):
			pos_enc = self.pos_encoding[:, :embedded.size(1), :].to(embedded.device)
			embedded = embedded + pos_enc

		embedded = self.dropout(embedded)

		# LSTM
		lstm_out, (hidden, cell) = self.lstm(embedded)

		# Project LSTM output to ensure correct dimensions
		lstm_out = self.output_projection(lstm_out)

		# Self-attention
		attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out, mask)
		lstm_out = self.layer_norm1(lstm_out + self.dropout(attn_out))

		# Feed forward
		ff_out = self.feed_forward(lstm_out)
		output = self.layer_norm2(lstm_out + self.dropout(ff_out))

		return output, hidden, cell


class ImprovedDecoder(nn.Module):
	"""Improved decoder with cross-attention"""

	def __init__(self, vocab_size, embed_size, hidden_size, num_layers=3, num_heads=8, dropout=0.1):
		super().__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.embed_size = embed_size

		self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
		self.pos_encoding = self.create_pos_encoding(5000, embed_size)
		self.dropout = nn.Dropout(dropout)

		# LSTM with larger input size to accommodate context
		self.lstm = nn.LSTM(embed_size + hidden_size * 2, hidden_size, num_layers,
		                    batch_first=True, dropout=dropout)

		# Multi-head attention layers
		self.cross_attention = MultiHeadAttention(hidden_size, num_heads, dropout)
		self.self_attention = MultiHeadAttention(hidden_size, num_heads, dropout)

		# Layer normalization
		self.layer_norm1 = nn.LayerNorm(hidden_size)
		self.layer_norm2 = nn.LayerNorm(hidden_size)
		self.layer_norm3 = nn.LayerNorm(hidden_size)

		# Feed forward network
		self.feed_forward = nn.Sequential(
			nn.Linear(hidden_size, hidden_size * 4),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size * 4, hidden_size)
		)

		# Output projection with vocabulary prediction
		self.output_projection = nn.Linear(hidden_size, vocab_size)

	def create_pos_encoding(self, max_len, embed_size):
		pe = torch.zeros(max_len, embed_size)
		position = torch.arange(0, max_len).unsqueeze(1).float()

		div_term = torch.exp(torch.arange(0, embed_size, 2).float() *
		                     -(math.log(10000.0) / embed_size))

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		return pe.unsqueeze(0)

	def forward(self, input_token, hidden, cell, encoder_outputs, src_mask=None):
		batch_size = input_token.size(0)

		# Embedding with positional encoding
		embedded = self.embedding(input_token)
		if embedded.size(1) <= self.pos_encoding.size(1):
			pos_enc = self.pos_encoding[:, :embedded.size(1), :].to(embedded.device)
			embedded = embedded + pos_enc

		embedded = self.dropout(embedded)

		# Calculate attention context from encoder
		query = hidden[-1].unsqueeze(1)  # Use last layer hidden state

		# Fix src_mask dimensions for cross-attention
		cross_mask = src_mask
		if cross_mask is not None:
			# src_mask should be [batch_size, 1, 1, src_seq_len] for cross-attention
			if cross_mask.dim() == 4:
				# Expand for decoder sequence length (which is 1 in this case)
				cross_mask = cross_mask.expand(-1, -1, input_token.size(1), -1)

		context, cross_attn_weights = self.cross_attention(query, encoder_outputs, encoder_outputs, cross_mask)

		# Combine embedding with context
		lstm_input = torch.cat([embedded, context.expand(-1, embedded.size(1), -1)], dim=2)

		# LSTM forward pass
		lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

		# Self-attention on decoder output (no mask needed for single token)
		self_attn_out, _ = self.self_attention(lstm_output, lstm_output, lstm_output, None)
		lstm_output = self.layer_norm1(lstm_output + self.dropout(self_attn_out))

		# Cross-attention with fixed mask
		cross_attn_out, cross_attn_weights = self.cross_attention(lstm_output, encoder_outputs, encoder_outputs,
		                                                          cross_mask)
		lstm_output = self.layer_norm2(lstm_output + self.dropout(cross_attn_out))

		# Feed forward
		ff_out = self.feed_forward(lstm_output)
		output = self.layer_norm3(lstm_output + self.dropout(ff_out))

		# Final projection
		output = self.output_projection(output)

		return output, hidden, cell, cross_attn_weights


class ImprovedSeq2Seq(nn.Module):
	"""Improved Seq2Seq with better architecture"""

	def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=512, hidden_size=512,
	             num_layers=3, num_heads=8, dropout=0.1):
		super().__init__()
		self.encoder = ImprovedEncoder(src_vocab_size, embed_size, hidden_size,
		                               num_layers, num_heads, dropout)
		self.decoder = ImprovedDecoder(tgt_vocab_size, embed_size, hidden_size,
		                               num_layers, num_heads, dropout)

		# Bridge layers
		self.bridge_h = nn.Linear(hidden_size * 2, hidden_size)
		self.bridge_c = nn.Linear(hidden_size * 2, hidden_size)

		self.init_weights()

	def init_weights(self):
		for name, param in self.named_parameters():
			if 'weight' in name and param.dim() > 1:
				if 'embedding' in name:
					nn.init.normal_(param, 0, 0.1)
				else:
					nn.init.xavier_uniform_(param)
			elif 'bias' in name:
				nn.init.zeros_(param)

	def create_src_mask(self, src):
		# Create mask for source padding
		# src: [batch_size, seq_len]
		# Return: [batch_size, 1, 1, seq_len] for cross-attention compatibility
		mask = (src != 0).unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
		return mask

	def forward(self, src, tgt, teacher_forcing_ratio=0.5):
		batch_size = src.size(0)
		tgt_len = tgt.size(1)
		tgt_vocab_size = self.decoder.output_projection.out_features

		src_mask = self.create_src_mask(src)

		# Encode
		encoder_outputs, encoder_hidden, encoder_cell = self.encoder(src)

		# Bridge encoder states to decoder
		h_forward = encoder_hidden[-2]
		h_backward = encoder_hidden[-1]
		c_forward = encoder_cell[-2]
		c_backward = encoder_cell[-1]

		h_combined = torch.cat([h_forward, h_backward], dim=1)
		c_combined = torch.cat([c_forward, c_backward], dim=1)

		decoder_hidden = torch.tanh(self.bridge_h(h_combined))
		decoder_cell = torch.tanh(self.bridge_c(c_combined))

		decoder_hidden = decoder_hidden.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
		decoder_cell = decoder_cell.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)

		# Decode
		outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)
		input_token = tgt[:, 0].unsqueeze(1)

		for t in range(1, tgt_len):
			output, decoder_hidden, decoder_cell, _ = self.decoder(
				input_token, decoder_hidden, decoder_cell, encoder_outputs, src_mask)
			outputs[:, t] = output.squeeze(1)

			# Teacher forcing with curriculum learning
			if random.random() < teacher_forcing_ratio:
				input_token = tgt[:, t].unsqueeze(1)
			else:
				input_token = output.argmax(2)

		return outputs


def advanced_beam_search(model, sentence, src_word2idx, tgt_idx2word, device,
                         beam_size=5, max_len=50, length_penalty=0.6):
	"""Advanced beam search with length penalty and coverage"""
	model.eval()

	with torch.no_grad():
		# Prepare input
		src_indices = [src_word2idx.get(word, src_word2idx['<UNK>'])
		               for word in sentence.lower().split()] + [src_word2idx['<EOS>']]
		src_tensor = torch.tensor([src_indices], dtype=torch.long).to(device)
		src_mask = model.create_src_mask(src_tensor)

		# Encode
		encoder_outputs, encoder_hidden, encoder_cell = model.encoder(src_tensor)

		# Initialize decoder states
		h_forward = encoder_hidden[-2]
		h_backward = encoder_hidden[-1]
		c_forward = encoder_cell[-2]
		c_backward = encoder_cell[-1]

		h_combined = torch.cat([h_forward, h_backward], dim=1)
		c_combined = torch.cat([c_forward, c_backward], dim=1)

		decoder_hidden = torch.tanh(model.bridge_h(h_combined))
		decoder_cell = torch.tanh(model.bridge_c(c_combined))

		decoder_hidden = decoder_hidden.unsqueeze(0).repeat(model.decoder.num_layers, 1, 1)
		decoder_cell = decoder_cell.unsqueeze(0).repeat(model.decoder.num_layers, 1, 1)

		# Beam search with improved scoring
		beams = [(torch.tensor([[1]], device=device), decoder_hidden, decoder_cell, 0.0, [])]

		for step in range(max_len):
			candidates = []

			for input_token, hidden, cell, score, sequence in beams:
				if len(sequence) > 0 and sequence[-1] == 2:  # EOS
					candidates.append((input_token, hidden, cell, score, sequence))
					continue

				output, new_hidden, new_cell, _ = model.decoder(
					input_token, hidden, cell, encoder_outputs, src_mask)

				log_probs = F.log_softmax(output.squeeze(1), dim=-1)
				top_probs, top_indices = log_probs.topk(beam_size)

				for i in range(beam_size):
					new_token = top_indices[0, i].item()
					token_score = top_probs[0, i].item()
					new_sequence = sequence + [new_token]

					# Length penalty: encourage longer sequences
					length_pen = ((5 + len(new_sequence)) / 6) ** length_penalty
					new_score = (score + token_score) / length_pen

					candidates.append((
						torch.tensor([[new_token]], device=device),
						new_hidden, new_cell, new_score, new_sequence
					))

			# Keep top beam_size candidates
			beams = sorted(candidates, key=lambda x: x[3], reverse=True)[:beam_size]

			# Early stopping if all beams end
			if all(len(seq) > 0 and seq[-1] == 2 for _, _, _, _, seq in beams):
				break

		# Return best sequence
		best_sequence = max(beams, key=lambda x: x[3])[4]
		return indices_to_sentence(best_sequence, tgt_idx2word)


def sentence_to_indices(sentence, word2idx):
	"""Convert sentence to indices with UNK handling"""
	return [word2idx.get(word, word2idx['<UNK>']) for word in sentence.split()]


def indices_to_sentence(indices, idx2word):
	"""Convert indices to sentence"""
	words = []
	for idx in indices:
		if idx == 2:  # EOS
			break
		if idx not in [0, 1]:  # Skip PAD and SOS
			words.append(idx2word.get(idx, '<UNK>'))
	return ' '.join(words)


def train_improved_model(model, train_loader, val_loader, device, epochs=15, patience=7):
	"""Improved training with better optimization"""

	# Use AdamW with weight decay and learning rate scheduling
	optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01,
	                        betas=(0.9, 0.98), eps=1e-9)

	# Cosine annealing with warm restarts
	scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
		optimizer, T_0=5, T_mult=2, eta_min=1e-6)

	# Label smoothing cross entropy
	criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

	best_val_loss = float('inf')
	patience_counter = 0
	training_history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}

	for epoch in range(epochs):
		print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
		current_lr = optimizer.param_groups[0]['lr']
		print(f"Learning rate: {current_lr:.6f}")

		# Training phase
		model.train()
		train_losses = []
		train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

		for batch_idx, batch in enumerate(train_pbar):
			src = batch['src'].to(device)
			tgt_input = batch['tgt_input'].to(device)
			tgt_output = batch['tgt_output'].to(device)

			optimizer.zero_grad()

			# Curriculum learning: start with more teacher forcing, gradually reduce
			teacher_forcing_ratio = max(0.9 * (0.95 ** epoch), 0.1)

			outputs = model(src, tgt_input, teacher_forcing_ratio=teacher_forcing_ratio)
			loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)),
			                 tgt_output[:, 1:].reshape(-1))

			loss.backward()

			# Gradient clipping
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			scheduler.step()

			batch_loss = loss.item()
			train_losses.append(batch_loss)

			if batch_idx % 50 == 0:
				avg_loss = sum(train_losses[-50:]) / min(50, len(train_losses))
				train_pbar.set_postfix({
					'batch_loss': f'{batch_loss:.4f}',
					'avg_loss': f'{avg_loss:.4f}',
					'tf_ratio': f'{teacher_forcing_ratio:.3f}',
					'lr': f'{current_lr:.6f}'
				})

		avg_train_loss = sum(train_losses) / len(train_losses)

		# Validation phase
		model.eval()
		val_losses = []

		with torch.no_grad():
			val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")
			for batch_idx, batch in enumerate(val_pbar):
				src = batch['src'].to(device)
				tgt_input = batch['tgt_input'].to(device)
				tgt_output = batch['tgt_output'].to(device)

				outputs = model(src, tgt_input, teacher_forcing_ratio=0.0)
				loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)),
				                 tgt_output[:, 1:].reshape(-1))

				batch_loss = loss.item()
				val_losses.append(batch_loss)

				if batch_idx % 20 == 0:
					avg_loss = sum(val_losses[-20:]) / min(20, len(val_losses))
					val_pbar.set_postfix({'batch_loss': f'{batch_loss:.4f}', 'avg_loss': f'{avg_loss:.4f}'})

		avg_val_loss = sum(val_losses) / len(val_losses)

		# Store training history
		training_history['train_loss'].append(avg_train_loss)
		training_history['val_loss'].append(avg_val_loss)
		training_history['learning_rates'].append(current_lr)

		# Print epoch summary
		print(f"Epoch {epoch + 1} Summary:")
		print(f"  Train Loss: {avg_train_loss:.4f}")
		print(f"  Val Loss:   {avg_val_loss:.4f}")
		print(f"  Teacher Forcing Ratio: {teacher_forcing_ratio:.3f}")

		# Save best model
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			patience_counter = 0
			torch.save({
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'scheduler_state_dict': scheduler.state_dict(),
				'epoch': epoch,
				'train_loss': avg_train_loss,
				'val_loss': avg_val_loss,
				'training_history': training_history,
				'src_word2idx': None,  # Will be added in main function
				'tgt_word2idx': None,
				'tgt_idx2word': None
			}, 'best_improved_model.pth')
			print(f"  ✓ Saved best model! (Val Loss: {avg_val_loss:.4f})")
		else:
			patience_counter += 1
			print(f"  No improvement for {patience_counter} epochs")

		# Early stopping
		if patience_counter >= patience:
			print(f"Early stopping triggered after {patience} epochs without improvement")
			break

	return training_history


def collate_fn(batch):
	"""Enhanced collate function for DataLoader"""
	src_batch = [item['src'] for item in batch]
	tgt_input_batch = [item['tgt_input'] for item in batch]
	tgt_output_batch = [item['tgt_output'] for item in batch]

	# Pad sequences
	src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
	tgt_input_batch = pad_sequence(tgt_input_batch, batch_first=True, padding_value=0)
	tgt_output_batch = pad_sequence(tgt_output_batch, batch_first=True, padding_value=0)

	return {
		'src': src_batch,
		'tgt_input': tgt_input_batch,
		'tgt_output': tgt_output_batch
	}


def evaluate_translation_quality(model, test_sentences, src_word2idx, tgt_idx2word, device):
	"""Evaluate translation quality with multiple metrics"""
	from collections import Counter
	import time

	print("=== Translation Quality Evaluation ===")

	translations = []
	translation_times = []

	for i, sentence in enumerate(test_sentences):
		start_time = time.time()
		translation = advanced_beam_search(model, sentence, src_word2idx, tgt_idx2word, device)
		end_time = time.time()

		translations.append(translation)
		translation_times.append(end_time - start_time)

		print(f"{i + 1:2d}. EN: {sentence}")
		print(f"    FR: {translation}")
		print(f"    Time: {translation_times[-1]:.2f}s")
		print("-" * 60)

	avg_time = sum(translation_times) / len(translation_times)
	print(f"Average translation time: {avg_time:.3f}s per sentence")

	return translations


def main():
	"""Main function with improved pipeline"""
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")
	if torch.cuda.is_available():
		print(f"GPU: {torch.cuda.get_device_name(0)}")
		print(f"Memory: {torch.cuda.get_device_properties(0).total_memory // 1024 ** 3}GB")

	# Download better quality datasets
	print("=== Downloading Better Quality Datasets ===")
	file_pairs = download_better_opus_data()

	if not file_pairs:
		print("No datasets downloaded successfully. Using OpenSubtitles as fallback...")
		# Fallback to original OpenSubtitles
		url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-fr.txt.zip"
		filename = "en-fr.txt.zip"

		if not os.path.exists(filename):
			print("Downloading OpenSubtitles dataset...")
			try:
				response = requests.get(url, stream=True, timeout=60)
				response.raise_for_status()
				total_size = int(response.headers.get('content-length', 0))

				with open(filename, 'wb') as f, tqdm(
						desc=filename, total=total_size, unit='iB',
						unit_scale=True, unit_divisor=1024) as pbar:
					for chunk in response.iter_content(chunk_size=8192):
						size = f.write(chunk)
						pbar.update(size)

				with zipfile.ZipFile(filename, 'r') as zip_ref:
					zip_ref.extractall('.')
				file_pairs = [("OpenSubtitles.en-fr.en", "OpenSubtitles.en-fr.fr")]
			except Exception as e:
				print(f"Failed to download fallback dataset: {e}")
				return

	# Load and combine datasets
	print("=== Loading and Processing Data ===")
	en_sentences, fr_sentences = load_multiple_datasets_debug(file_pairs, max_samples_per_dataset=50000)

	if len(en_sentences) == 0:
		print("No data loaded successfully. Exiting...")
		return

	print(f"Total dataset size: {len(en_sentences)} sentence pairs")

	# Build improved vocabularies
	print("=== Building Vocabularies ===")
	src_word2idx, src_idx2word = build_bpe_like_vocab(en_sentences, max_vocab=30000, min_freq=2)
	tgt_word2idx, tgt_idx2word = build_bpe_like_vocab(fr_sentences, max_vocab=30000, min_freq=2)

	print(f"English vocabulary size: {len(src_word2idx)}")
	print(f"French vocabulary size: {len(tgt_word2idx)}")

	# Split data with stratification if possible
	print("=== Splitting Data ===")

	# Limit dataset size for memory efficiency
	max_total_samples = 150000
	if len(en_sentences) > max_total_samples:
		print(f"Limiting dataset to {max_total_samples} samples for memory efficiency")
		indices = random.sample(range(len(en_sentences)), max_total_samples)
		en_sentences = [en_sentences[i] for i in indices]
		fr_sentences = [fr_sentences[i] for i in indices]

	train_en, val_en, train_fr, val_fr = train_test_split(
		en_sentences, fr_sentences, test_size=0.15, random_state=42)

	print(f"Training samples: {len(train_en)}")
	print(f"Validation samples: {len(val_en)}")

	# Create datasets and data loaders
	print("=== Creating Data Loaders ===")
	train_dataset = ImprovedTranslationDataset(train_en, train_fr, src_word2idx, tgt_word2idx, max_len=50)
	val_dataset = ImprovedTranslationDataset(val_en, val_fr, src_word2idx, tgt_word2idx, max_len=50)

	# Adjust batch size based on available memory
	batch_size = 32 if torch.cuda.is_available() else 16

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
	                          collate_fn=collate_fn, num_workers=2, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
	                        collate_fn=collate_fn, num_workers=2, pin_memory=True)

	# Create improved model
	print("=== Creating Model ===")
	model = ImprovedSeq2Seq(
		src_vocab_size=len(src_word2idx),
		tgt_vocab_size=len(tgt_word2idx),
		embed_size=512,
		hidden_size=512,
		num_layers=3,
		num_heads=8,
		dropout=0.1
	).to(device)

	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Total parameters: {total_params:,}")
	print(f"Trainable parameters: {trainable_params:,}")

	# Train model
	print("=== Training Model ===")
	training_history = train_improved_model(model, train_loader, val_loader, device, epochs=15, patience=7)

	# Load best model for evaluation
	print("=== Loading Best Model ===")
	try:
		checkpoint = torch.load('best_improved_model.pth', map_location=device)
		model.load_state_dict(checkpoint['model_state_dict'])
		print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
		print(f"Best validation loss: {checkpoint['val_loss']:.4f}")

		# Save vocabularies in checkpoint for future use
		checkpoint['src_word2idx'] = src_word2idx
		checkpoint['tgt_word2idx'] = tgt_word2idx
		checkpoint['tgt_idx2word'] = tgt_idx2word
		torch.save(checkpoint, 'best_improved_model.pth')

	except Exception as e:
		print(f"Error loading best model: {e}")
		print("Using current model weights")

	# Comprehensive evaluation
	print("=== Comprehensive Evaluation ===")

	# Test sentences covering various topics and difficulties
	test_sentences = [
		# Basic greetings and common phrases
		"Hello, how are you today?",
		"Thank you very much for your help.",
		"What time is it now?",
		"I am learning French language.",

		# Daily life and activities
		"I am going to the supermarket to buy groceries.",
		"She likes to read books in the library.",
		"We are planning our vacation for next summer.",
		"The weather is beautiful and sunny today.",

		# More complex sentences
		"The government announced new policies to combat climate change.",
		"Scientists have discovered a new species of butterfly in the rainforest.",
		"The university offers many courses in computer science and engineering.",
		"Children should learn to respect the environment and nature.",

		# Conversational and idiomatic
		"Can you please help me find the train station?",
		"I don't understand what you are saying.",
		"This restaurant serves delicious Italian food.",
		"My brother works as a doctor in the hospital.",

		# Business and formal
		"The meeting has been scheduled for tomorrow morning.",
		"We need to review the financial reports carefully.",
		"The company is expanding its operations internationally.",
		"Please send me the documents by email."
	]

	translations = evaluate_translation_quality(model, test_sentences, src_word2idx, tgt_idx2word, device)

	print("=== Training History Summary ===")
	print(f"Final training loss: {training_history['train_loss'][-1]:.4f}")
	print(f"Final validation loss: {training_history['val_loss'][-1]:.4f}")
	print(f"Best validation loss: {min(training_history['val_loss']):.4f}")
	print(f"Training completed in {len(training_history['train_loss'])} epochs")

	# Save final results
	results = {
		'model_info': {
			'total_params': total_params,
			'trainable_params': trainable_params,
			'vocab_sizes': {'src': len(src_word2idx), 'tgt': len(tgt_word2idx)},
			'dataset_size': len(en_sentences)
		},
		'training_history': training_history,
		'test_translations': list(zip(test_sentences, translations))
	}

	import json
	with open('training_results.json', 'w', encoding='utf-8') as f:
		json.dump(results, f, indent=2, ensure_ascii=False)

	print("Results saved to 'training_results.json'")
	print("Best model saved to 'best_improved_model.pth'")


if __name__ == "__main__":
	main()