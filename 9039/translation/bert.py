import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np
from sklearn.model_selection import train_test_split
import os
import requests
import zipfile
from tqdm import tqdm
import random

# Set random seed
def set_seed(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

set_seed(42)

# Data download and preprocessing
class DatasetDownloader:
	@staticmethod
	def download_opus_data():
		"""Download OPUS English-French parallel corpus"""
		url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-fr.txt.zip"
		filename = "en-fr.txt.zip"

		if not os.path.exists(filename):
			print("Downloading dataset...")
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
	def create_sample_data():
		"""Create sample dataset (if unable to download real data)"""
		en_sentences = [
			"Hello, how are you?",
			"What is your name?",
			"I love learning languages.",
			"The weather is beautiful today.",
			"Can you help me with this problem?",
			"I am going to the store.",
			"This book is very interesting.",
			"I would like to order food.",
			"Where is the nearest hospital?",
			"Thank you for your help.",
			"Good morning, everyone.",
			"I need to catch the train.",
			"The movie was fantastic.",
			"Could you please repeat that?",
			"I am learning French.",
			"What time is it?",
			"I enjoy reading books.",
			"The food tastes delicious.",
			"How much does this cost?",
			"Have a great day!"
		]

		fr_sentences = [
			"Bonjour, comment allez-vous?",
			"Comment vous appelez-vous?",
			"J'adore apprendre les langues.",
			"Le temps est magnifique aujourd'hui.",
			"Pouvez-vous m'aider avec ce problème?",
			"Je vais au magasin.",
			"Ce livre est très intéressant.",
			"Je voudrais commander de la nourriture.",
			"Où est l'hôpital le plus proche?",
			"Merci pour votre aide.",
			"Bonjour tout le monde.",
			"Je dois prendre le train.",
			"Le film était fantastique.",
			"Pourriez-vous répéter cela?",
			"J'apprends le français.",
			"Quelle heure est-il?",
			"J'aime lire des livres.",
			"La nourriture a un goût délicieux.",
			"Combien cela coûte-t-il?",
			"Passez une excellente journée!"
		]

		# Expand dataset
		extended_en = en_sentences * 50  # Expand to 1000 samples
		extended_fr = fr_sentences * 50

		# Add some random variations
		for i in range(len(extended_en)):
			if random.random() < 0.1:  # 10% probability of adding variation
				extended_en[i] = extended_en[i].replace(".", "!")
				extended_fr[i] = extended_fr[i].replace(".", "!")

		return extended_en, extended_fr


# Translation dataset class
class TranslationDataset(Dataset):
	def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_length=128):
		self.src_texts = src_texts
		self.tgt_texts = tgt_texts
		self.src_tokenizer = src_tokenizer
		self.tgt_tokenizer = tgt_tokenizer
		self.max_length = max_length

	def __len__(self):
		return len(self.src_texts)

	def __getitem__(self, idx):
		src_text = str(self.src_texts[idx])
		tgt_text = str(self.tgt_texts[idx])

		# Encode source language text
		src_encoding = self.src_tokenizer(
			src_text,
			truncation=True,
			padding='max_length',
			max_length=self.max_length,
			return_tensors='pt'
		)

		# Encode target language text
		tgt_encoding = self.tgt_tokenizer(
			tgt_text,
			truncation=True,
			padding='max_length',
			max_length=self.max_length,
			return_tensors='pt'
		)

		# Create decoder input and labels
		decoder_input_ids = tgt_encoding['input_ids'].clone()
		labels = tgt_encoding['input_ids'].clone()

		# Shift decoder input to the right by one position
		decoder_input_ids[:, 1:] = labels[:, :-1]
		decoder_input_ids[:, 0] = self.tgt_tokenizer.cls_token_id

		return {
			'src_input_ids': src_encoding['input_ids'].flatten(),
			'src_attention_mask': src_encoding['attention_mask'].flatten(),
			'decoder_input_ids': decoder_input_ids.flatten(),
			'labels': labels.flatten()
		}


# BERT encoder-decoder model
class BertEncoderDecoder(nn.Module):
	def __init__(self, src_vocab_size, tgt_vocab_size, d_model=768, nhead=12, num_layers=6):
		super(BertEncoderDecoder, self).__init__()

		# Use pre-trained BERT as encoder
		self.encoder = BertModel.from_pretrained('bert-base-uncased')

		# Decoder
		self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
		self.pos_embedding = nn.Embedding(512, d_model)

		decoder_layer = nn.TransformerDecoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=2048,
			dropout=0.1,
			batch_first=True
		)
		self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

		# Output projection layer
		self.output_projection = nn.Linear(d_model, tgt_vocab_size)
		self.dropout = nn.Dropout(0.1)

	def forward(self, src_input_ids, src_attention_mask, decoder_input_ids):
		# Encoder
		encoder_outputs = self.encoder(
			input_ids=src_input_ids,
			attention_mask=src_attention_mask
		)
		encoder_hidden_states = encoder_outputs.last_hidden_state

		# Decoder
		batch_size, seq_len = decoder_input_ids.size()

		# Target embeddings
		tgt_embeddings = self.tgt_embedding(decoder_input_ids)

		# Position embeddings
		positions = torch.arange(seq_len, device=decoder_input_ids.device).unsqueeze(0).expand(batch_size, -1)
		pos_embeddings = self.pos_embedding(positions)

		# Combine embeddings
		decoder_inputs = self.dropout(tgt_embeddings + pos_embeddings)

		# Create causal mask
		tgt_mask = self.generate_square_subsequent_mask(seq_len).to(decoder_input_ids.device)

		# Decoder forward pass
		decoder_outputs = self.decoder(
			tgt=decoder_inputs,
			memory=encoder_hidden_states,
			tgt_mask=tgt_mask,
			memory_key_padding_mask=~src_attention_mask.bool()
		)

		# Output projection
		logits = self.output_projection(decoder_outputs)

		return logits

	def generate_square_subsequent_mask(self, sz):
		mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
		return mask


# Trainer class
class TranslationTrainer:
	def __init__(self, model, train_loader, val_loader, device, lr=1e-4):
		self.model = model.to(device)
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.device = device
		self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
		self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.7)
		self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token

	def train_epoch(self):
		self.model.train()
		total_loss = 0

		for batch in tqdm(self.train_loader, desc="Training"):
			self.optimizer.zero_grad()

			src_input_ids = batch['src_input_ids'].to(self.device)
			src_attention_mask = batch['src_attention_mask'].to(self.device)
			decoder_input_ids = batch['decoder_input_ids'].to(self.device)
			labels = batch['labels'].to(self.device)

			# Forward pass
			logits = self.model(src_input_ids, src_attention_mask, decoder_input_ids)

			# Calculate loss
			loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

			# Backward pass
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
			self.optimizer.step()

			total_loss += loss.item()

		return total_loss / len(self.train_loader)

	def validate(self):
		self.model.eval()
		total_loss = 0

		with torch.no_grad():
			for batch in tqdm(self.val_loader, desc="Validating"):
				src_input_ids = batch['src_input_ids'].to(self.device)
				src_attention_mask = batch['src_attention_mask'].to(self.device)
				decoder_input_ids = batch['decoder_input_ids'].to(self.device)
				labels = batch['labels'].to(self.device)

				logits = self.model(src_input_ids, src_attention_mask, decoder_input_ids)
				loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

				total_loss += loss.item()

		return total_loss / len(self.val_loader)

	def train(self, epochs):
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
				torch.save(self.model.state_dict(), 'best_translation_model.pth')
				print("Saved best model!")

			self.scheduler.step()


# Translation inference class
class Translator:
	def __init__(self, model, src_tokenizer, tgt_tokenizer, device):
		self.model = model.to(device)
		self.src_tokenizer = src_tokenizer
		self.tgt_tokenizer = tgt_tokenizer
		self.device = device
		self.model.eval()

	def translate(self, text, max_length=128):
		# Encode input text
		src_encoding = self.src_tokenizer(
			text,
			truncation=True,
			padding='max_length',
			max_length=max_length,
			return_tensors='pt'
		)

		src_input_ids = src_encoding['input_ids'].to(self.device)
		src_attention_mask = src_encoding['attention_mask'].to(self.device)

		# Generate translation
		with torch.no_grad():
			# Start token
			decoder_input_ids = torch.tensor([[self.tgt_tokenizer.cls_token_id]], device=self.device)

			for _ in range(max_length):
				logits = self.model(src_input_ids, src_attention_mask, decoder_input_ids)

				# Get next token
				next_token_logits = logits[0, -1, :]
				next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)

				# If end token is generated, stop generation
				if next_token.item() == self.tgt_tokenizer.sep_token_id:
					break

				# Add to sequence
				decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)

		# Decode generated sequence
		generated_ids = decoder_input_ids[0].cpu().numpy()
		translation = self.tgt_tokenizer.decode(generated_ids, skip_special_tokens=True)

		return translation


# Main function
def main():
	# Set device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Hardware type: {device}")

	# Initialize tokenizer
	src_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

	# Get data
	try:
		print("Trying to download real dataset...")
		downloader = DatasetDownloader()
		en_file, fr_file = downloader.download_opus_data()

		# Read data
		with open(en_file, 'r', encoding='utf-8') as f:
			en_sentences = [line.strip() for line in f.readlines()[:10000]]  # Limit data size

		with open(fr_file, 'r', encoding='utf-8') as f:
			fr_sentences = [line.strip() for line in f.readlines()[:10000]]

	except Exception as e:
		print(f"Download failed: {e}")
		print("Using sample dataset...")
		downloader = DatasetDownloader()
		en_sentences, fr_sentences = downloader.create_sample_data()

	print(f"Dataset size: {len(en_sentences)} sentences")

	# Data split
	train_en, val_en, train_fr, val_fr = train_test_split(
		en_sentences, fr_sentences, test_size=0.2, random_state=42
	)

	# Create datasets
	train_dataset = TranslationDataset(train_en, train_fr, src_tokenizer, tgt_tokenizer)
	val_dataset = TranslationDataset(val_en, val_fr, src_tokenizer, tgt_tokenizer)

	# Create data loaders
	train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

	# Create model
	model = BertEncoderDecoder(
		src_vocab_size=src_tokenizer.vocab_size,
		tgt_vocab_size=tgt_tokenizer.vocab_size
	)

	print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")

	# Train model
	trainer = TranslationTrainer(model, train_loader, val_loader, device)
	trainer.train(epochs=10)

	# Load best model for testing
	model.load_state_dict(torch.load('best_translation_model.pth'))
	translator = Translator(model, src_tokenizer, tgt_tokenizer, device)

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

	print("\n=== Translation Test ===")
	for sentence in test_sentences:
		translation = translator.translate(sentence)
		print(f"English: {sentence}")
		print(f"French: {translation}")
		print("-" * 50)


if __name__ == "__main__":
	main()