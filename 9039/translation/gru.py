import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.model_selection import train_test_split
import os
import requests
import zipfile
from tqdm import tqdm
import random
from collections import Counter
import pickle
import re
import unicodedata


# Set random seed for reproducibility
def set_seed(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


set_seed(42)


# Improved text preprocessing
def preprocess_sentence(sentence, lang='en'):
	"""Clean and preprocess sentences"""
	# Convert to lowercase
	sentence = sentence.lower().strip()

	# Remove extra whitespace
	sentence = re.sub(r'\s+', ' ', sentence)

	# Add space around punctuation
	sentence = re.sub(r'([.!?,:;])', r' \1 ', sentence)
	sentence = re.sub(r'\s+', ' ', sentence)

	# Remove quotes and other special characters
	sentence = re.sub(r'["""\'\'`]', '', sentence)

	# Keep only basic punctuation and alphanumeric characters
	if lang == 'en':
		sentence = re.sub(r'[^a-zA-Z0-9\s.!?,:;-]', '', sentence)
	else:  # French
		sentence = re.sub(r'[^a-zA-ZàâäçéèêëïîôöùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÖÙÛÜŸÑÆŒ0-9\s.!?,:;-]', '', sentence)

	sentence = sentence.strip()
	return sentence


# Enhanced Vocabulary class
class Vocabulary:
	def __init__(self, min_count=2):
		self.word2idx = {}
		self.idx2word = {}
		self.word_count = Counter()
		self.n_words = 0
		self.min_count = min_count

		# Special tokens
		self.PAD_token = 0
		self.SOS_token = 1  # Start of sentence
		self.EOS_token = 2  # End of sentence
		self.UNK_token = 3  # Unknown word

		# Add special tokens
		self.add_word('<PAD>')
		self.add_word('<SOS>')
		self.add_word('<EOS>')
		self.add_word('<UNK>')

	def add_word(self, word):
		if word not in self.word2idx:
			self.word2idx[word] = self.n_words
			self.idx2word[self.n_words] = word
			self.n_words += 1
		self.word_count[word] += 1

	def add_sentence(self, sentence):
		for word in sentence.split():
			self.word_count[word] += 1

	def build_vocab(self):
		"""Build vocabulary from word counts, filtering by min_count"""
		# Reset indices for non-special tokens
		temp_word2idx = {word: idx for word, idx in self.word2idx.items() if idx < 4}
		temp_idx2word = {idx: word for idx, word in self.idx2word.items() if idx < 4}
		temp_n_words = 4

		# Add words that meet minimum count threshold
		for word, count in self.word_count.items():
			if word not in temp_word2idx and count >= self.min_count:
				temp_word2idx[word] = temp_n_words
				temp_idx2word[temp_n_words] = word
				temp_n_words += 1

		self.word2idx = temp_word2idx
		self.idx2word = temp_idx2word
		self.n_words = temp_n_words

	def sentence_to_indices(self, sentence):
		indices = []
		for word in sentence.split():
			if word in self.word2idx:
				indices.append(self.word2idx[word])
			else:
				indices.append(self.UNK_token)
		return indices

	def indices_to_sentence(self, indices):
		words = []
		for idx in indices:
			if idx == self.EOS_token:
				break
			if idx not in [self.PAD_token, self.SOS_token]:
				words.append(self.idx2word.get(idx, '<UNK>'))
		return ' '.join(words)


# Enhanced data loading and preprocessing
class DatasetDownloader:
	@staticmethod
	def download_opus_data():
		"""Download OPUS English-French parallel corpus"""
		url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-fr.txt.zip"
		filename = "en-fr.txt.zip"

		if not os.path.exists(filename):
			print("Downloading OPUS dataset...")
			response = requests.get(url, stream=True)
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

		# Extract files
		with zipfile.ZipFile(filename, 'r') as zip_ref:
			zip_ref.extractall('.')

		return "OpenSubtitles.en-fr.en", "OpenSubtitles.en-fr.fr"

	@staticmethod
	def load_opus_data(en_file, fr_file, max_samples=50000, max_length=50):
		"""Load and preprocess OPUS data"""
		print(f"Loading data from {en_file} and {fr_file}...")

		en_sentences = []
		fr_sentences = []

		try:
			with open(en_file, 'r', encoding='utf-8') as ef, open(fr_file, 'r', encoding='utf-8') as ff:
				for i, (en_line, fr_line) in enumerate(zip(ef, ff)):
					if i >= max_samples:
						break

					en_clean = preprocess_sentence(en_line.strip(), 'en')
					fr_clean = preprocess_sentence(fr_line.strip(), 'fr')

					# Filter out empty sentences or sentences that are too long/short
					if (len(en_clean.split()) >= 3 and len(fr_clean.split()) >= 3 and
							len(en_clean.split()) <= max_length and len(fr_clean.split()) <= max_length and
							len(en_clean) > 0 and len(fr_clean) > 0):
						en_sentences.append(en_clean)
						fr_sentences.append(fr_clean)

		except Exception as e:
			print(f"Error loading OPUS data: {e}")
			return None, None

		print(f"Loaded {len(en_sentences)} sentence pairs from OPUS data")
		return en_sentences, fr_sentences

	@staticmethod
	def create_sample_data():
		"""Create enhanced sample dataset for testing"""
		en_sentences = [
			"hello how are you",
			"what is your name",
			"i love learning languages",
			"the weather is beautiful today",
			"can you help me with this problem",
			"i am going to the store",
			"this book is very interesting",
			"i would like to order food",
			"where is the nearest hospital",
			"thank you for your help",
			"good morning everyone",
			"i need to catch the train",
			"the movie was fantastic",
			"could you please repeat that",
			"i am learning french",
			"what time is it",
			"i enjoy reading books",
			"the food tastes delicious",
			"how much does this cost",
			"have a great day",
			"see you tomorrow",
			"i am sorry for being late",
			"this is a wonderful place",
			"can you speak english",
			"i do not understand",
			"please speak slowly",
			"where are you from",
			"i live in paris",
			"how old are you",
			"i am twenty years old"
		]

		fr_sentences = [
			"bonjour comment allez vous",
			"comment vous appelez vous",
			"j adore apprendre les langues",
			"le temps est magnifique aujourd hui",
			"pouvez vous m aider avec ce probleme",
			"je vais au magasin",
			"ce livre est tres interessant",
			"je voudrais commander de la nourriture",
			"ou est l hopital le plus proche",
			"merci pour votre aide",
			"bonjour tout le monde",
			"je dois prendre le train",
			"le film etait fantastique",
			"pourriez vous repeter cela",
			"j apprends le francais",
			"quelle heure est il",
			"j aime lire des livres",
			"la nourriture a un gout delicieux",
			"combien cela coute t il",
			"passez une excellente journee",
			"a demain",
			"je suis desole d etre en retard",
			"c est un endroit merveilleux",
			"parlez vous anglais",
			"je ne comprends pas",
			"parlez lentement s il vous plait",
			"d ou venez vous",
			"je vis a paris",
			"quel age avez vous",
			"j ai vingt ans"
		]

		# Expand dataset by repetition
		extended_en = en_sentences * 100
		extended_fr = fr_sentences * 100

		return extended_en, extended_fr


# Enhanced dataset class
class GRUTranslationDataset(Dataset):
	def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_length=50):
		self.src_sentences = src_sentences
		self.tgt_sentences = tgt_sentences
		self.src_vocab = src_vocab
		self.tgt_vocab = tgt_vocab
		self.max_length = max_length

	def __len__(self):
		return len(self.src_sentences)

	def __getitem__(self, idx):
		src_sentence = str(self.src_sentences[idx])
		tgt_sentence = str(self.tgt_sentences[idx])

		# Convert sentences to indices
		src_indices = self.src_vocab.sentence_to_indices(src_sentence)
		tgt_indices = self.tgt_vocab.sentence_to_indices(tgt_sentence)

		# Add EOS token and limit length
		src_indices = src_indices[:self.max_length - 1] + [self.src_vocab.EOS_token]
		tgt_indices = tgt_indices[:self.max_length - 1] + [self.tgt_vocab.EOS_token]

		# Create decoder input (SOS + target) and labels (target + EOS)
		decoder_input = [self.tgt_vocab.SOS_token] + tgt_indices[:-1]
		labels = tgt_indices

		return {
			'src': torch.tensor(src_indices, dtype=torch.long),
			'tgt_input': torch.tensor(decoder_input, dtype=torch.long),
			'tgt_output': torch.tensor(labels, dtype=torch.long)
		}


# Collate function for dynamic padding
def collate_fn(batch):
	src_batch = [item['src'] for item in batch]
	tgt_input_batch = [item['tgt_input'] for item in batch]
	tgt_output_batch = [item['tgt_output'] for item in batch]

	# Pad sequences to the same length within the batch
	src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
	tgt_input_batch = pad_sequence(tgt_input_batch, batch_first=True, padding_value=0)
	tgt_output_batch = pad_sequence(tgt_output_batch, batch_first=True, padding_value=0)

	return {
		'src': src_batch,
		'tgt_input': tgt_input_batch,
		'tgt_output': tgt_output_batch
	}


# Enhanced GRU Encoder
class GRUEncoder(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.2):
		super(GRUEncoder, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
		self.gru = nn.GRU(embed_size, hidden_size, num_layers,
		                  batch_first=True, dropout=dropout if num_layers > 1 else 0,
		                  bidirectional=True)
		self.dropout = nn.Dropout(dropout)

		# Initialize embeddings
		nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

	def forward(self, src, src_lengths=None):
		# src: (batch_size, seq_len)
		embedded = self.dropout(self.embedding(src))  # (batch_size, seq_len, embed_size)

		# Pack sequences if lengths are provided
		if src_lengths is not None:
			embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths,
			                                             batch_first=True, enforce_sorted=False)

		# GRU forward pass
		outputs, hidden = self.gru(embedded)

		# Unpack if packed
		if src_lengths is not None:
			outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

		return outputs, hidden


# Enhanced GRU Decoder with improved attention
class GRUDecoder(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.2):
		super(GRUDecoder, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.vocab_size = vocab_size

		self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
		self.gru = nn.GRU(embed_size + hidden_size * 2, hidden_size, num_layers,
		                  batch_first=True, dropout=dropout if num_layers > 1 else 0)

		# Improved attention mechanism
		self.attention_linear = nn.Linear(hidden_size + hidden_size * 2, hidden_size)
		self.attention_v = nn.Linear(hidden_size, 1, bias=False)

		# Output projection with intermediate layer
		self.out_linear1 = nn.Linear(hidden_size + hidden_size * 2, hidden_size)
		self.out_linear2 = nn.Linear(hidden_size, vocab_size)
		self.dropout = nn.Dropout(dropout)

		# Initialize embeddings and weights
		nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
		nn.init.xavier_uniform_(self.out_linear1.weight)
		nn.init.xavier_uniform_(self.out_linear2.weight)

	def forward(self, input_token, hidden, encoder_outputs, encoder_mask=None):
		# input_token: (batch_size, 1)
		# hidden: (num_layers, batch_size, hidden_size)
		# encoder_outputs: (batch_size, src_len, hidden_size * 2)

		batch_size = input_token.size(0)
		src_len = encoder_outputs.size(1)

		# Embedding
		embedded = self.dropout(self.embedding(input_token))  # (batch_size, 1, embed_size)

		# Attention mechanism
		# Use the last layer's hidden state for attention
		query = hidden[-1].unsqueeze(1)  # (batch_size, 1, hidden_size)

		# Expand query to match encoder outputs
		query_expanded = query.expand(-1, src_len, -1)  # (batch_size, src_len, hidden_size)

		# Concatenate query with encoder outputs
		attention_input = torch.cat([query_expanded, encoder_outputs], dim=2)
		attention_hidden = torch.tanh(self.attention_linear(attention_input))
		attention_scores = self.attention_v(attention_hidden).squeeze(2)  # (batch_size, src_len)

		# Apply mask if provided
		if encoder_mask is not None:
			attention_scores.masked_fill_(encoder_mask == 0, -1e9)

		# Calculate attention weights
		attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(2)  # (batch_size, src_len, 1)

		# Apply attention to encoder outputs
		context = torch.sum(attention_weights * encoder_outputs, dim=1,
		                    keepdim=True)  # (batch_size, 1, hidden_size * 2)

		# Concatenate embedded input with context
		gru_input = torch.cat([embedded, context], dim=2)  # (batch_size, 1, embed_size + hidden_size * 2)

		# GRU forward pass
		output, hidden = self.gru(gru_input, hidden)

		# Output projection with intermediate layer
		output_concat = torch.cat([output, context], dim=2)  # (batch_size, 1, hidden_size + hidden_size * 2)
		output_hidden = torch.tanh(self.out_linear1(output_concat))
		prediction = self.out_linear2(self.dropout(output_hidden))  # (batch_size, 1, vocab_size)

		return prediction, hidden, attention_weights.squeeze(2)


# Enhanced Seq2Seq model
class GRUSeq2Seq(nn.Module):
	def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=256, hidden_size=512, num_layers=2, dropout=0.2):
		super(GRUSeq2Seq, self).__init__()

		self.encoder = GRUEncoder(src_vocab_size, embed_size, hidden_size, num_layers, dropout)
		self.decoder = GRUDecoder(tgt_vocab_size, embed_size, hidden_size, num_layers, dropout)

		# Bridge layer to connect bidirectional encoder to decoder
		self.bridge = nn.Linear(hidden_size * 2, hidden_size)
		nn.init.xavier_uniform_(self.bridge.weight)

	def create_mask(self, src):
		# Create mask for padding tokens
		mask = (src != 0).float()
		return mask

	def forward(self, src, tgt, teacher_forcing_ratio=0.5):
		batch_size = src.size(0)
		tgt_len = tgt.size(1)
		tgt_vocab_size = self.decoder.vocab_size

		# Create source mask
		src_mask = self.create_mask(src)

		# Get source lengths for packing
		src_lengths = src_mask.sum(dim=1).cpu()

		# Encode source sequence
		encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)

		# Initialize decoder hidden state from encoder
		# Combine forward and backward hidden states from each layer
		decoder_hidden_list = []
		for i in range(self.decoder.num_layers):
			# Combine forward and backward states for layer i
			h_forward = encoder_hidden[i * 2]  # Forward hidden state
			h_backward = encoder_hidden[i * 2 + 1]  # Backward hidden state
			h_combined = torch.tanh(self.bridge(torch.cat([h_forward, h_backward], dim=1)))
			decoder_hidden_list.append(h_combined)

		decoder_hidden = torch.stack(decoder_hidden_list, dim=0)

		# Initialize output tensor
		outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)

		# First input to decoder is SOS token
		input_token = tgt[:, 0].unsqueeze(1)  # (batch_size, 1)

		for t in range(1, tgt_len):
			# Decoder forward pass
			output, decoder_hidden, attention_weights = self.decoder(
				input_token, decoder_hidden, encoder_outputs, src_mask
			)

			# Store output
			outputs[:, t] = output.squeeze(1)

			# Teacher forcing: use ground truth as next input with probability
			use_teacher_forcing = random.random() < teacher_forcing_ratio
			if use_teacher_forcing:
				input_token = tgt[:, t].unsqueeze(1)
			else:
				input_token = output.argmax(dim=2)

		return outputs


# Enhanced training class with better loss handling
class GRUTranslationTrainer:
	def __init__(self, model, train_loader, val_loader, src_vocab, tgt_vocab, device, lr=1e-3):
		self.model = model.to(device)
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.src_vocab = src_vocab
		self.tgt_vocab = tgt_vocab
		self.device = device

		# Use Adam optimizer with weight decay
		self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

		# Use cosine annealing scheduler
		self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-6)

		# Use label smoothing for better generalization
		self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

	def train_epoch(self):
		self.model.train()
		total_loss = 0
		num_batches = 0

		for batch in tqdm(self.train_loader, desc="Training"):
			src = batch['src'].to(self.device)
			tgt_input = batch['tgt_input'].to(self.device)
			tgt_output = batch['tgt_output'].to(self.device)

			self.optimizer.zero_grad()

			# Forward pass with teacher forcing
			outputs = self.model(src, tgt_input, teacher_forcing_ratio=0.7)

			# Calculate loss (ignore the first time step as it's SOS)
			loss = self.criterion(outputs[:, 1:].contiguous().view(-1, outputs.size(-1)),
			                      tgt_output[:, 1:].contiguous().view(-1))

			# Backward pass
			loss.backward()

			# Gradient clipping
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

			self.optimizer.step()

			total_loss += loss.item()
			num_batches += 1

		return total_loss / num_batches

	def validate(self):
		self.model.eval()
		total_loss = 0
		num_batches = 0

		with torch.no_grad():
			for batch in tqdm(self.val_loader, desc="Validating"):
				src = batch['src'].to(self.device)
				tgt_input = batch['tgt_input'].to(self.device)
				tgt_output = batch['tgt_output'].to(self.device)

				# Forward pass without teacher forcing for validation
				outputs = self.model(src, tgt_input, teacher_forcing_ratio=0.0)

				loss = self.criterion(outputs[:, 1:].contiguous().view(-1, outputs.size(-1)),
				                      tgt_output[:, 1:].contiguous().view(-1))

				total_loss += loss.item()
				num_batches += 1

		return total_loss / num_batches

	def train(self, epochs):
		best_val_loss = float('inf')
		patience = 3
		patience_counter = 0

		for epoch in range(epochs):
			print(f"\nEpoch {epoch + 1}/{epochs}")

			train_loss = self.train_epoch()
			val_loss = self.validate()

			print(f"Train Loss: {train_loss:.4f}")
			print(f"Validation Loss: {val_loss:.4f}")
			print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")

			# Save best model
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				patience_counter = 0
				torch.save({
					'model_state_dict': self.model.state_dict(),
					'src_vocab': self.src_vocab,
					'tgt_vocab': self.tgt_vocab,
					'model_config': {
						'src_vocab_size': len(self.src_vocab.word2idx),
						'tgt_vocab_size': len(self.tgt_vocab.word2idx),
						'embed_size': 256,
						'hidden_size': 512,
						'num_layers': 2
					}
				}, 'best_gru_translation_model.pth')
				print("Saved best model!")
			else:
				patience_counter += 1

			self.scheduler.step()

			# Early stopping
			if patience_counter >= patience:
				print(f"Early stopping triggered after {epoch + 1} epochs")
				break


# Enhanced translator with beam search
class GRUTranslator:
	def __init__(self, model, src_vocab, tgt_vocab, device, max_length=50):
		self.model = model.to(device)
		self.src_vocab = src_vocab
		self.tgt_vocab = tgt_vocab
		self.device = device
		self.max_length = max_length
		self.model.eval()

	def translate(self, sentence):
		with torch.no_grad():
			# Preprocess input sentence
			sentence = preprocess_sentence(sentence, 'en')
			src_indices = self.src_vocab.sentence_to_indices(sentence)
			src_indices = src_indices + [self.src_vocab.EOS_token]
			src_tensor = torch.tensor([src_indices], dtype=torch.long).to(self.device)

			# Create source mask
			src_mask = (src_tensor != 0).float()
			src_lengths = src_mask.sum(dim=1).cpu()

			# Encode source sentence
			encoder_outputs, encoder_hidden = self.model.encoder(src_tensor, src_lengths)

			# Initialize decoder hidden state
			decoder_hidden_list = []
			for i in range(self.model.decoder.num_layers):
				h_forward = encoder_hidden[i * 2]
				h_backward = encoder_hidden[i * 2 + 1]
				h_combined = torch.tanh(self.model.bridge(torch.cat([h_forward, h_backward], dim=1)))
				decoder_hidden_list.append(h_combined)

			decoder_hidden = torch.stack(decoder_hidden_list, dim=0)

			# Start translation with SOS token
			input_token = torch.tensor([[self.tgt_vocab.SOS_token]], dtype=torch.long).to(self.device)
			translated_indices = []

			for _ in range(self.max_length):
				output, decoder_hidden, attention_weights = self.model.decoder(
					input_token, decoder_hidden, encoder_outputs, src_mask
				)

				# Get the most likely next token
				next_token = output.argmax(dim=2)
				next_token_idx = next_token.item()

				# Stop if EOS token is generated
				if next_token_idx == self.tgt_vocab.EOS_token:
					break

				translated_indices.append(next_token_idx)
				input_token = next_token

			# Convert indices back to sentence
			translated_sentence = self.tgt_vocab.indices_to_sentence(translated_indices)
			return translated_sentence


# Main function with improved data handling
def main():
	# Set device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	# Try to get real OPUS data first
	use_opus_data = False
	try:
		print("Attempting to download OPUS data...")
		downloader = DatasetDownloader()
		en_file, fr_file = downloader.download_opus_data()

		# Load OPUS data
		en_sentences, fr_sentences = downloader.load_opus_data(en_file, fr_file, max_samples=30000)

		if en_sentences is not None and len(en_sentences) > 1000:
			use_opus_data = True
			print(f"Successfully loaded {len(en_sentences)} sentence pairs from OPUS data")
		else:
			print("OPUS data loading failed or insufficient data")

	except Exception as e:
		print(f"OPUS data download/loading failed: {e}")

	# Fall back to sample data if OPUS failed
	if not use_opus_data:
		print("Using enhanced sample dataset...")
		downloader = DatasetDownloader()
		en_sentences, fr_sentences = downloader.create_sample_data()

	print(f"Final dataset size: {len(en_sentences)} sentence pairs")

	# Build vocabularies with improved filtering
	print("Building vocabularies...")
	src_vocab = Vocabulary(min_count=2 if use_opus_data else 1)
	tgt_vocab = Vocabulary(min_count=2 if use_opus_data else 1)

	# Add words to vocabulary
	for sentence in en_sentences:
		src_vocab.add_sentence(sentence)

	for sentence in fr_sentences:
		tgt_vocab.add_sentence(sentence)

	# Build final vocabularies
	src_vocab.build_vocab()
	tgt_vocab.build_vocab()

	print(f"Source vocabulary size: {src_vocab.n_words}")
	print(f"Target vocabulary size: {tgt_vocab.n_words}")

	# Split data
	train_en, val_en, train_fr, val_fr = train_test_split(
		en_sentences, fr_sentences, test_size=0.2, random_state=42
	)

	print(f"Training samples: {len(train_en)}")
	print(f"Validation samples: {len(val_en)}")

	# Create datasets
	train_dataset = GRUTranslationDataset(train_en, train_fr, src_vocab, tgt_vocab)
	val_dataset = GRUTranslationDataset(val_en, val_fr, src_vocab, tgt_vocab)

	# Create data loaders
	batch_size = 32 if use_opus_data else 16
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

	# Create enhanced model
	model = GRUSeq2Seq(
		src_vocab_size=src_vocab.n_words,
		tgt_vocab_size=tgt_vocab.n_words,
		embed_size=256,
		hidden_size=512,
		num_layers=2,
		dropout=0.2
	)

	print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

	# Train model with improved trainer
	lr = 1e-3 if use_opus_data else 5e-4
	trainer = GRUTranslationTrainer(model, train_loader, val_loader, src_vocab, tgt_vocab, device, lr=lr)

	# Use more epochs for real data, fewer for sample data
	epochs = 10 if use_opus_data else 20
	trainer.train(epochs=epochs)

	# Load best model for testing
	try:
		checkpoint = torch.load('best_gru_translation_model.pth', map_location=device, weights_only=False)
		model.load_state_dict(checkpoint['model_state_dict'])
		print("Loaded best model for translation")
	except:
		print("Using current model for translation")

	# Create translator
	translator = GRUTranslator(model, src_vocab, tgt_vocab, device)

	# Test translations with diverse examples
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
		"She doesn't like spicy food.",
		"Good morning everyone.",
		"Where are you from?",
		"I need to catch the train.",
		"How much does this cost?",
		"Please speak slowly."
	]

	print("\n=== Translation Results ===")
	for sentence in test_sentences:
		translation = translator.translate(sentence)
		print(f"English: {sentence}")
		print(f"French: {translation}")
		print("-" * 50)

	# Save vocabularies for future use
	with open('vocabularies.pkl', 'wb') as f:
		pickle.dump({'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}, f)
	print("\nVocabularies saved to 'vocabularies.pkl'")


if __name__ == "__main__":
	main()