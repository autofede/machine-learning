import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import random
from tqdm import tqdm
import pickle
import os
import requests
import zipfile


# Set random seed for reproducibility
def set_seed(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


set_seed(42)


# Dataset downloader for OPUS corpus
class OPUSDataDownloader:
	@staticmethod
	def download_opus_data():
		"""Download OPUS English-French parallel corpus"""
		url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-fr.txt.zip"
		filename = "en-fr.txt.zip"

		if not os.path.exists(filename):
			print("Downloading OPUS dataset...")
			try:
				response = requests.get(url, stream=True)
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

				print("Download completed successfully!")

			except Exception as e:
				print(f"Failed to download: {e}")
				return None, None

		# Extract files
		try:
			print("Extracting files...")
			with zipfile.ZipFile(filename, 'r') as zip_ref:
				zip_ref.extractall('.')
			print("Extraction completed!")
			return "OpenSubtitles.en-fr.en", "OpenSubtitles.en-fr.fr"
		except Exception as e:
			print(f"Failed to extract: {e}")
			return None, None

	@staticmethod
	def create_sample_data():
		"""Create sample dataset as fallback"""
		en_sentences = [
			"Hello, how are you?", "What is your name?", "I love learning languages.",
			"The weather is beautiful today.", "Can you help me with this problem?",
			"I am going to the store.", "This book is very interesting.",
			"I would like to order food.", "Where is the nearest hospital?",
			"Thank you for your help.", "Good morning, everyone.",
			"I need to catch the train.", "The movie was fantastic.",
			"Could you please repeat that?", "I am learning French.",
			"What time is it?", "I enjoy reading books.",
			"The food tastes delicious.", "How much does this cost?",
			"Have a great day!", "I like to travel.",
			"This is my favorite restaurant.", "Can you speak English?",
			"I need some help.", "Where do you live?",
			"What do you do for work?", "I am from America.",
			"Nice to meet you.", "See you tomorrow.", "Have a good night.",
			"How old are you?", "What is your favorite color?",
			"I like to read books.", "The sun is shining brightly.",
			"Can I have some water?", "Where is the bathroom?",
			"I am very tired today.", "What did you do yesterday?",
			"I want to go home.", "This is delicious food.",
			"How was your day?", "I need to buy groceries.",
			"The movie starts at eight.", "Can you drive me there?",
			"I forgot my keys.", "What is the weather like?",
			"I have a meeting tomorrow.", "This is my phone number.",
			"Can you call me later?", "I am running late.",
			"The traffic is very heavy.", "I love this song."
		]

		fr_sentences = [
			"Bonjour, comment allez-vous?", "Comment vous appelez-vous?", "J'adore apprendre les langues.",
			"Le temps est magnifique aujourd'hui.", "Pouvez-vous m'aider avec ce problème?",
			"Je vais au magasin.", "Ce livre est très intéressant.",
			"Je voudrais commander de la nourriture.", "Où est l'hôpital le plus proche?",
			"Merci pour votre aide.", "Bonjour tout le monde.",
			"Je dois prendre le train.", "Le film était fantastique.",
			"Pourriez-vous répéter cela?", "J'apprends le français.",
			"Quelle heure est-il?", "J'aime lire des livres.",
			"La nourriture a un goût délicieux.", "Combien cela coûte-t-il?",
			"Passez une excellente journée!", "J'aime voyager.",
			"C'est mon restaurant préféré.", "Parlez-vous anglais?",
			"J'ai besoin d'aide.", "Où habitez-vous?",
			"Que faites-vous comme travail?", "Je viens d'Amérique.",
			"Ravi de vous rencontrer.", "À demain.", "Bonne nuit.",
			"Quel âge avez-vous?", "Quelle est votre couleur préférée?",
			"J'aime lire des livres.", "Le soleil brille intensément.",
			"Puis-je avoir de l'eau?", "Où sont les toilettes?",
			"Je suis très fatigué aujourd'hui.", "Qu'avez-vous fait hier?",
			"Je veux rentrer à la maison.", "C'est de la nourriture délicieuse.",
			"Comment s'est passée votre journée?", "Je dois acheter des courses.",
			"Le film commence à huit heures.", "Pouvez-vous me conduire là-bas?",
			"J'ai oublié mes clés.", "Quel temps fait-il?",
			"J'ai une réunion demain.", "Voici mon numéro de téléphone.",
			"Pouvez-vous m'appeler plus tard?", "Je suis en retard.",
			"La circulation est très dense.", "J'adore cette chanson."
		]

		# Expand dataset by duplication with variations
		extended_en = en_sentences * 20
		extended_fr = fr_sentences * 20

		return extended_en, extended_fr


# Vocabulary builder class
class Vocabulary:
	def __init__(self):
		self.word2idx = {}
		self.idx2word = {}
		self.word_count = Counter()
		self.PAD_TOKEN = '<PAD>'
		self.SOS_TOKEN = '<SOS>'
		self.EOS_TOKEN = '<EOS>'
		self.UNK_TOKEN = '<UNK>'

		# Add special tokens
		self.add_word(self.PAD_TOKEN)  # 0
		self.add_word(self.SOS_TOKEN)  # 1
		self.add_word(self.EOS_TOKEN)  # 2
		self.add_word(self.UNK_TOKEN)  # 3

	def add_word(self, word):
		"""Add a word to vocabulary"""
		if word not in self.word2idx:
			idx = len(self.word2idx)
			self.word2idx[word] = idx
			self.idx2word[idx] = word
		self.word_count[word] += 1

	def add_sentence(self, sentence):
		"""Add all words from sentence to vocabulary"""
		words = self.tokenize(sentence)
		for word in words:
			self.add_word(word)

	def tokenize(self, sentence):
		"""Simple tokenizer - splits on whitespace and removes punctuation"""
		sentence = sentence.lower()
		sentence = re.sub(r'[^\w\s]', ' ', sentence)  # Remove punctuation
		return sentence.split()

	def sentence_to_indices(self, sentence, max_length=None):
		"""Convert sentence to list of word indices"""
		words = self.tokenize(sentence)
		indices = [self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]) for word in words]

		if max_length:
			if len(indices) < max_length:
				indices.extend([self.word2idx[self.PAD_TOKEN]] * (max_length - len(indices)))
			else:
				indices = indices[:max_length]

		return indices

	def indices_to_sentence(self, indices):
		"""Convert list of indices back to sentence"""
		words = []
		for idx in indices:
			if idx == self.word2idx[self.EOS_TOKEN]:
				break
			if idx != self.word2idx[self.PAD_TOKEN]:
				words.append(self.idx2word.get(idx, self.UNK_TOKEN))
		return ' '.join(words)

	def filter_rare_words(self, min_count=2):
		"""Remove words that appear less than min_count times"""
		frequent_words = {word for word, count in self.word_count.items() if count >= min_count}

		# Keep special tokens
		frequent_words.update([self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN])

		# Rebuild vocabulary with frequent words only
		old_word2idx = self.word2idx.copy()
		self.word2idx = {}
		self.idx2word = {}

		# Re-add special tokens first
		for token in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
			self.add_word(token)

		# Add frequent words
		for word in frequent_words:
			if word not in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
				if word not in self.word2idx:
					idx = len(self.word2idx)
					self.word2idx[word] = idx
					self.idx2word[idx] = word

	def __len__(self):
		return len(self.word2idx)


# Translation dataset class
class TranslationDataset(Dataset):
	def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_length=50):
		self.src_sentences = src_sentences
		self.tgt_sentences = tgt_sentences
		self.src_vocab = src_vocab
		self.tgt_vocab = tgt_vocab
		self.max_length = max_length

	def __len__(self):
		return len(self.src_sentences)

	def __getitem__(self, idx):
		src_sentence = self.src_sentences[idx]
		tgt_sentence = self.tgt_sentences[idx]

		# Convert sentences to indices
		src_indices = self.src_vocab.sentence_to_indices(src_sentence, self.max_length)

		# Add SOS and EOS tokens to target sentence
		tgt_indices = [self.tgt_vocab.word2idx[self.tgt_vocab.SOS_TOKEN]]
		tgt_indices.extend(self.tgt_vocab.sentence_to_indices(tgt_sentence, self.max_length - 2))
		tgt_indices.append(self.tgt_vocab.word2idx[self.tgt_vocab.EOS_TOKEN])

		# Pad to same length
		if len(tgt_indices) < self.max_length:
			tgt_indices.extend(
				[self.tgt_vocab.word2idx[self.tgt_vocab.PAD_TOKEN]] * (self.max_length - len(tgt_indices)))
		else:
			tgt_indices = tgt_indices[:self.max_length]

		return {
			'src': torch.tensor(src_indices, dtype=torch.long),
			'tgt_input': torch.tensor(tgt_indices[:-1], dtype=torch.long),  # Decoder input
			'tgt_output': torch.tensor(tgt_indices[1:], dtype=torch.long)  # Decoder target
		}


# RNN Encoder
class RNNEncoder(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
		super(RNNEncoder, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
		self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
		self.dropout = nn.Dropout(0.1)

	def forward(self, x):
		# x: (batch_size, seq_len)
		embedded = self.dropout(self.embedding(x))  # (batch_size, seq_len, embed_size)

		# RNN forward pass
		outputs, hidden = self.rnn(embedded)
		# outputs: (batch_size, seq_len, hidden_size)
		# hidden: (num_layers, batch_size, hidden_size)

		return outputs, hidden


# RNN Decoder
class RNNDecoder(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
		super(RNNDecoder, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.vocab_size = vocab_size

		self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
		self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
		self.attention = nn.Linear(hidden_size * 2, hidden_size)  # Simple attention mechanism
		self.out = nn.Linear(hidden_size, vocab_size)
		self.dropout = nn.Dropout(0.1)

	def forward(self, x, hidden, encoder_outputs=None):
		# x: (batch_size, seq_len)
		# hidden: (num_layers, batch_size, hidden_size)

		embedded = self.dropout(self.embedding(x))  # (batch_size, seq_len, embed_size)

		# RNN forward pass
		outputs, hidden = self.rnn(embedded, hidden)
		# outputs: (batch_size, seq_len, hidden_size)

		# Simple attention mechanism (optional)
		if encoder_outputs is not None:
			# Calculate attention weights (simplified)
			attention_weights = torch.softmax(
				torch.bmm(outputs, encoder_outputs.transpose(1, 2)), dim=2
			)
			context = torch.bmm(attention_weights, encoder_outputs)

			# Combine context with decoder output
			combined = torch.cat([outputs, context], dim=2)
			outputs = torch.tanh(self.attention(combined))

		# Output layer
		predictions = self.out(outputs)  # (batch_size, seq_len, vocab_size)

		return predictions, hidden


# Seq2Seq RNN Model
class Seq2SeqRNN(nn.Module):
	def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=256, hidden_size=512, num_layers=2,
	             use_attention=True):
		super(Seq2SeqRNN, self).__init__()

		self.encoder = RNNEncoder(src_vocab_size, embed_size, hidden_size, num_layers)
		self.decoder = RNNDecoder(tgt_vocab_size, embed_size, hidden_size, num_layers)
		self.use_attention = use_attention

	def forward(self, src, tgt):
		# Encoder
		encoder_outputs, encoder_hidden = self.encoder(src)

		# Decoder
		if self.use_attention:
			decoder_outputs, _ = self.decoder(tgt, encoder_hidden, encoder_outputs)
		else:
			decoder_outputs, _ = self.decoder(tgt, encoder_hidden)

		return decoder_outputs


# Training class
class RNNTranslationTrainer:
	def __init__(self, model, train_loader, val_loader, device, lr=0.001):
		self.model = model.to(device)
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.device = device
		self.optimizer = optim.Adam(model.parameters(), lr=lr)
		self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.8)
		self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token

	def train_epoch(self):
		"""Train for one epoch"""
		self.model.train()
		total_loss = 0

		for batch in tqdm(self.train_loader, desc="Training"):
			self.optimizer.zero_grad()

			src = batch['src'].to(self.device)
			tgt_input = batch['tgt_input'].to(self.device)
			tgt_output = batch['tgt_output'].to(self.device)

			# Forward pass
			outputs = self.model(src, tgt_input)

			# Calculate loss
			loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))

			# Backward pass
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
			self.optimizer.step()

			total_loss += loss.item()

		return total_loss / len(self.train_loader)

	def validate(self):
		"""Validate the model"""
		self.model.eval()
		total_loss = 0

		with torch.no_grad():
			for batch in tqdm(self.val_loader, desc="Validating"):
				src = batch['src'].to(self.device)
				tgt_input = batch['tgt_input'].to(self.device)
				tgt_output = batch['tgt_output'].to(self.device)

				outputs = self.model(src, tgt_input)
				loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))

				total_loss += loss.item()

		return total_loss / len(self.val_loader)

	def train(self, epochs):
		"""Train the model for specified epochs"""
		best_val_loss = float('inf')

		for epoch in range(epochs):
			print(f"\nEpoch {epoch + 1}/{epochs}")

			train_loss = self.train_epoch()
			val_loss = self.validate()

			print(f"Train Loss: {train_loss:.4f}")
			print(f"Val Loss: {val_loss:.4f}")

			# Save best model
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				torch.save({
					'model_state_dict': self.model.state_dict(),
					'optimizer_state_dict': self.optimizer.state_dict(),
					'val_loss': val_loss,
				}, 'best_rnn_translation_model.pth')
				print("Saved best model!")

			self.scheduler.step()


# RNN Translator for inference
class RNNTranslator:
	def __init__(self, model, src_vocab, tgt_vocab, device, max_length=50):
		self.model = model.to(device)
		self.src_vocab = src_vocab
		self.tgt_vocab = tgt_vocab
		self.device = device
		self.max_length = max_length
		self.model.eval()

	def translate(self, sentence):
		"""Translate a single sentence"""
		with torch.no_grad():
			# Encode input sentence
			src_indices = self.src_vocab.sentence_to_indices(sentence, self.max_length)
			src_tensor = torch.tensor([src_indices], dtype=torch.long).to(self.device)

			# Encoder
			encoder_outputs, encoder_hidden = self.model.encoder(src_tensor)

			# Decoder initialization
			decoder_hidden = encoder_hidden
			decoder_input = torch.tensor([[self.tgt_vocab.word2idx[self.tgt_vocab.SOS_TOKEN]]],
			                             dtype=torch.long).to(self.device)

			decoded_words = []

			# Generate translation word by word
			for _ in range(self.max_length):
				if self.model.use_attention:
					decoder_output, decoder_hidden = self.model.decoder(decoder_input, decoder_hidden, encoder_outputs)
				else:
					decoder_output, decoder_hidden = self.model.decoder(decoder_input, decoder_hidden)

				# Get predicted next word
				topv, topi = decoder_output.topk(1)
				decoder_input = topi.squeeze().detach().unsqueeze(0).unsqueeze(0)

				predicted_idx = topi.item()

				# Stop if EOS token is generated
				if predicted_idx == self.tgt_vocab.word2idx[self.tgt_vocab.EOS_TOKEN]:
					break

				# Add word to output if not padding
				if predicted_idx != self.tgt_vocab.word2idx[self.tgt_vocab.PAD_TOKEN]:
					decoded_words.append(self.tgt_vocab.idx2word[predicted_idx])

			return ' '.join(decoded_words)


# Main function
def main():
	# Set device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	# Download and load data
	downloader = OPUSDataDownloader()
	en_file, fr_file = downloader.download_opus_data()

	if en_file and fr_file:
		print("Loading OPUS dataset...")
		try:
			# Load real OPUS data
			with open(en_file, 'r', encoding='utf-8') as f:
				en_sentences = [line.strip() for line in f.readlines()[:50000]]  # Limit dataset size

			with open(fr_file, 'r', encoding='utf-8') as f:
				fr_sentences = [line.strip() for line in f.readlines()[:50000]]

			# Filter out empty sentences and ensure equal length
			valid_pairs = [(en, fr) for en, fr in zip(en_sentences, fr_sentences)
			               if len(en.strip()) > 0 and len(fr.strip()) > 0 and len(en.split()) <= 30 and len(
					fr.split()) <= 30]

			en_sentences, fr_sentences = zip(*valid_pairs)
			en_sentences, fr_sentences = list(en_sentences), list(fr_sentences)

			print(f"Loaded {len(en_sentences)} sentence pairs from OPUS")

		except Exception as e:
			print(f"Error loading OPUS data: {e}")
			print("Using sample data instead...")
			en_sentences, fr_sentences = downloader.create_sample_data()
	else:
		print("Using sample data...")
		en_sentences, fr_sentences = downloader.create_sample_data()

	print(f"Dataset size: {len(en_sentences)} sentences")

	# Build vocabularies
	print("Building vocabularies...")
	src_vocab = Vocabulary()
	tgt_vocab = Vocabulary()

	for sentence in en_sentences:
		src_vocab.add_sentence(sentence)

	for sentence in fr_sentences:
		tgt_vocab.add_sentence(sentence)

	# Filter rare words to reduce vocabulary size
	src_vocab.filter_rare_words(min_count=2)
	tgt_vocab.filter_rare_words(min_count=2)

	print(f"English vocabulary size: {len(src_vocab)}")
	print(f"French vocabulary size: {len(tgt_vocab)}")

	# Split data
	train_en, val_en, train_fr, val_fr = train_test_split(
		en_sentences, fr_sentences, test_size=0.1, random_state=42
	)

	# Create datasets
	train_dataset = TranslationDataset(train_en, train_fr, src_vocab, tgt_vocab, max_length=30)
	val_dataset = TranslationDataset(val_en, val_fr, src_vocab, tgt_vocab, max_length=30)

	# Create data loaders
	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

	# Create model
	model = Seq2SeqRNN(
		src_vocab_size=len(src_vocab),
		tgt_vocab_size=len(tgt_vocab),
		embed_size=256,
		hidden_size=512,
		num_layers=2,
		use_attention=True
	)

	print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

	# Train model
	trainer = RNNTranslationTrainer(model, train_loader, val_loader, device, lr=0.001)
	trainer.train(epochs=10)

	# Load best model for testing
	checkpoint = torch.load('best_rnn_translation_model.pth')
	model.load_state_dict(checkpoint['model_state_dict'])

	translator = RNNTranslator(model, src_vocab, tgt_vocab, device)

	# Test translation
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

	print("\n=== RNN Translation Test ===")
	for sentence in test_sentences:
		translation = translator.translate(sentence)
		print(f"English: {sentence}")
		print(f"French: {translation}")
		print("-" * 60)

	# Save vocabularies
	with open('vocabularies.pkl', 'wb') as f:
		pickle.dump({'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}, f)
	print("Vocabularies saved to vocabularies.pkl")


if __name__ == "__main__":
	main()