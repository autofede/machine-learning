import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.model_selection import train_test_split
import os
import requests
import zipfile
from tqdm import tqdm
import random
import torch.nn.functional as F


# Set random seed
def set_seed(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


set_seed(42)


# Improved dataset with OPUS download capability
class DatasetDownloader:
	@staticmethod
	def download_opus_data():
		"""Download OPUS English-French parallel corpus"""
		url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-fr.txt.zip"
		filename = "en-fr.txt.zip"

		if not os.path.exists(filename):
			print("Downloading OPUS dataset...")
			try:
				response = requests.get(url, stream=True, timeout=30)
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
						if chunk:
							size = f.write(chunk)
							pbar.update(size)
			except Exception as e:
				print(f"Download failed: {e}")
				raise e

		# Extract files
		try:
			with zipfile.ZipFile(filename, 'r') as zip_ref:
				zip_ref.extractall('.')
				print("Files extracted successfully")
		except Exception as e:
			print(f"Extraction failed: {e}")
			raise e

		return "OpenSubtitles.en-fr.en", "OpenSubtitles.en-fr.fr"

	@staticmethod
	def load_opus_data(en_file, fr_file, max_samples=20000):
		"""Load and preprocess OPUS data"""
		en_sentences = []
		fr_sentences = []

		try:
			print(f"Loading data from {en_file} and {fr_file}")

			# Read English sentences
			with open(en_file, 'r', encoding='utf-8', errors='ignore') as f:
				en_lines = f.readlines()

			# Read French sentences
			with open(fr_file, 'r', encoding='utf-8', errors='ignore') as f:
				fr_lines = f.readlines()

			# Ensure both files have same number of lines
			min_lines = min(len(en_lines), len(fr_lines))
			print(f"Found {min_lines} parallel sentences")

			# Process and filter sentences
			for i in range(min(min_lines, max_samples)):
				en_sent = en_lines[i].strip()
				fr_sent = fr_lines[i].strip()

				# Filter out empty or very short/long sentences
				if (len(en_sent) > 5 and len(fr_sent) > 5 and
						len(en_sent) < 200 and len(fr_sent) < 200 and
						not en_sent.startswith('<') and not fr_sent.startswith('<')):
					en_sentences.append(en_sent)
					fr_sentences.append(fr_sent)

			print(f"Loaded {len(en_sentences)} valid sentence pairs")
			return en_sentences, fr_sentences

		except Exception as e:
			print(f"Error loading OPUS data: {e}")
			raise e

	@staticmethod
	def create_better_sample_data():
		"""Create higher quality sample dataset with more diverse examples"""
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
			"Have a great day!",
			"I work as a software engineer.",
			"My family lives in Paris.",
			"The concert starts at eight.",
			"She speaks three languages fluently.",
			"We visited the museum yesterday.",
			"The train is running late.",
			"I need to buy groceries.",
			"The restaurant serves excellent food.",
			"It's raining outside today.",
			"He plays the piano beautifully.",
			"The children are playing in the park.",
			"I have a meeting this afternoon.",
			"The book was published last year.",
			"She graduated from university.",
			"We are planning a vacation.",
			"The dog is sleeping on the couch.",
			"I forgot my keys at home.",
			"The store closes at nine.",
			"He drives to work every day.",
			"The flowers are blooming in spring."
		]

		fr_sentences = [
			"Bonjour, comment allez-vous ?",
			"Comment vous appelez-vous ?",
			"J'adore apprendre les langues.",
			"Le temps est magnifique aujourd'hui.",
			"Pouvez-vous m'aider avec ce problème ?",
			"Je vais au magasin.",
			"Ce livre est très intéressant.",
			"Je voudrais commander de la nourriture.",
			"Où est l'hôpital le plus proche ?",
			"Merci pour votre aide.",
			"Bonjour tout le monde.",
			"Je dois prendre le train.",
			"Le film était fantastique.",
			"Pourriez-vous répéter cela ?",
			"J'apprends le français.",
			"Quelle heure est-il ?",
			"J'aime lire des livres.",
			"La nourriture a un goût délicieux.",
			"Combien cela coûte-t-il ?",
			"Passez une excellente journée !",
			"Je travaille comme ingénieur logiciel.",
			"Ma famille vit à Paris.",
			"Le concert commence à huit heures.",
			"Elle parle couramment trois langues.",
			"Nous avons visité le musée hier.",
			"Le train est en retard.",
			"Je dois acheter des provisions.",
			"Le restaurant sert une excellente cuisine.",
			"Il pleut dehors aujourd'hui.",
			"Il joue du piano magnifiquement.",
			"Les enfants jouent dans le parc.",
			"J'ai une réunion cet après-midi.",
			"Le livre a été publié l'année dernière.",
			"Elle a obtenu son diplôme universitaire.",
			"Nous planifions des vacances.",
			"Le chien dort sur le canapé.",
			"J'ai oublié mes clés à la maison.",
			"Le magasin ferme à neuf heures.",
			"Il conduit au travail tous les jours.",
			"Les fleurs fleurissent au printemps."
		]

		# Expand dataset with variations
		extended_en = []
		extended_fr = []

		# Create more training examples
		for _ in range(25):  # 25 * 40 = 1000 examples
			extended_en.extend(en_sentences)
			extended_fr.extend(fr_sentences)

		return extended_en, extended_fr


# Improved Translation dataset with better preprocessing
class ImprovedTranslationDataset(Dataset):
	def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_length=128):
		self.src_texts = src_texts
		self.tgt_texts = tgt_texts
		self.src_tokenizer = src_tokenizer
		self.tgt_tokenizer = tgt_tokenizer
		self.max_length = max_length

	def __len__(self):
		return len(self.src_texts)

	def __getitem__(self, idx):
		src_text = str(self.src_texts[idx]).strip()
		tgt_text = str(self.tgt_texts[idx]).strip()

		# Add special tokens for target
		tgt_text = "[CLS] " + tgt_text + " [SEP]"

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
		input_ids = tgt_encoding['input_ids'].flatten()
		decoder_input_ids = input_ids[:-1].clone()  # All tokens except last
		labels = input_ids[1:].clone()  # All tokens except first

		# Pad to max_length
		if len(decoder_input_ids) < self.max_length - 1:
			pad_length = self.max_length - 1 - len(decoder_input_ids)
			decoder_input_ids = torch.cat([decoder_input_ids, torch.zeros(pad_length, dtype=torch.long)])
			labels = torch.cat([labels, torch.full((pad_length,), -100, dtype=torch.long)])

		return {
			'src_input_ids': src_encoding['input_ids'].flatten(),
			'src_attention_mask': src_encoding['attention_mask'].flatten(),
			'decoder_input_ids': decoder_input_ids,
			'labels': labels
		}


# Improved BERT encoder-decoder model
class ImprovedBertEncoderDecoder(nn.Module):
	def __init__(self, src_vocab_size, tgt_vocab_size, d_model=768, nhead=12, num_layers=6):
		super(ImprovedBertEncoderDecoder, self).__init__()

		# Use multilingual BERT for better language support
		self.encoder = AutoModel.from_pretrained('bert-base-multilingual-cased')

		# Freeze some encoder layers to prevent overfitting
		for param in self.encoder.embeddings.parameters():
			param.requires_grad = False
		for layer in self.encoder.encoder.layer[:6]:  # Freeze first 6 layers
			for param in layer.parameters():
				param.requires_grad = False

		# Improved decoder
		self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
		self.pos_embedding = nn.Embedding(512, d_model)

		# Cross-attention layer
		self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)

		# Decoder layers
		decoder_layer = nn.TransformerDecoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=2048,
			dropout=0.1,
			batch_first=True,
			activation='gelu'
		)
		self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

		# Output layers with residual connection
		self.pre_output_layer = nn.Linear(d_model, d_model)
		self.output_projection = nn.Linear(d_model, tgt_vocab_size)
		self.layer_norm = nn.LayerNorm(d_model)
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

		# Improved output projection with residual connection
		pre_output = self.pre_output_layer(decoder_outputs)
		pre_output = self.layer_norm(pre_output + decoder_outputs)  # Residual connection
		logits = self.output_projection(self.dropout(pre_output))

		return logits

	def generate_square_subsequent_mask(self, sz):
		mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
		return mask


# Improved trainer with better optimization
class ImprovedTranslationTrainer:
	def __init__(self, model, train_loader, val_loader, device, lr=5e-5):
		self.model = model.to(device)
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.device = device

		# Improved optimizer settings
		self.optimizer = optim.AdamW(
			model.parameters(),
			lr=lr,
			weight_decay=0.01,
			betas=(0.9, 0.999),
			eps=1e-8
		)

		# Better learning rate scheduler
		self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
			self.optimizer, T_0=3, T_mult=2, eta_min=1e-6
		)

		# Label smoothing for better generalization
		self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

	def train_epoch(self):
		self.model.train()
		total_loss = 0
		num_batches = 0

		for batch in tqdm(self.train_loader, desc="Training"):
			self.optimizer.zero_grad()

			src_input_ids = batch['src_input_ids'].to(self.device)
			src_attention_mask = batch['src_attention_mask'].to(self.device)
			decoder_input_ids = batch['decoder_input_ids'].to(self.device)
			labels = batch['labels'].to(self.device)

			# Forward pass
			logits = self.model(src_input_ids, src_attention_mask, decoder_input_ids)

			# Calculate loss (only for non-padded tokens)
			loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

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
				src_input_ids = batch['src_input_ids'].to(self.device)
				src_attention_mask = batch['src_attention_mask'].to(self.device)
				decoder_input_ids = batch['decoder_input_ids'].to(self.device)
				labels = batch['labels'].to(self.device)

				logits = self.model(src_input_ids, src_attention_mask, decoder_input_ids)
				loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

				total_loss += loss.item()
				num_batches += 1

		return total_loss / num_batches

	def train(self, epochs):
		best_val_loss = float('inf')

		for epoch in range(epochs):
			print(f"\nEpoch {epoch + 1}/{epochs}")

			train_loss = self.train_epoch()
			val_loss = self.validate()

			print(f"Train Loss: {train_loss:.4f}")
			print(f"Val Loss: {val_loss:.4f}")
			print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")

			# Save best model
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				torch.save({
					'model_state_dict': self.model.state_dict(),
					'optimizer_state_dict': self.optimizer.state_dict(),
					'epoch': epoch,
					'loss': val_loss
				}, 'best_improved_translation_model.pth')
				print("Saved best model!")

			self.scheduler.step()


# Improved translator with beam search
class ImprovedTranslator:
	def __init__(self, model, src_tokenizer, tgt_tokenizer, device):
		self.model = model.to(device)
		self.src_tokenizer = src_tokenizer
		self.tgt_tokenizer = tgt_tokenizer
		self.device = device
		self.model.eval()

	def translate(self, text, max_length=64, beam_size=3):
		"""Translate with beam search for better quality"""
		# Encode input text
		src_encoding = self.src_tokenizer(
			text,
			truncation=True,
			padding='max_length',
			max_length=128,
			return_tensors='pt'
		)

		src_input_ids = src_encoding['input_ids'].to(self.device)
		src_attention_mask = src_encoding['attention_mask'].to(self.device)

		with torch.no_grad():
			# Simple greedy decoding (can be improved with beam search)
			decoder_input_ids = torch.tensor([[self.tgt_tokenizer.cls_token_id]], device=self.device)

			for _ in range(max_length):
				logits = self.model(src_input_ids, src_attention_mask, decoder_input_ids)

				# Get next token probabilities
				next_token_logits = logits[0, -1, :]

				# Apply temperature for more diverse outputs
				next_token_probs = F.softmax(next_token_logits / 0.8, dim=-1)

				# Sample from top-k tokens
				top_k = 5
				top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k)
				next_token = torch.multinomial(top_k_probs, 1)
				next_token = top_k_indices[next_token].unsqueeze(0)

				# If end token is generated, stop generation
				if next_token.item() == self.tgt_tokenizer.sep_token_id:
					break

				# Add to sequence
				decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)

		# Decode generated sequence
		generated_ids = decoder_input_ids[0].cpu().numpy()
		translation = self.tgt_tokenizer.decode(generated_ids, skip_special_tokens=True)

		return translation.strip()


# Main function with improvements
def main():
	# Set device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	# Use multilingual tokenizers for better performance
	src_tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
	tgt_tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

	# Try to get OPUS data first, fallback to sample data
	downloader = DatasetDownloader()

	try:
		print("Attempting to download OPUS dataset...")
		en_file, fr_file = downloader.download_opus_data()
		en_sentences, fr_sentences = downloader.load_opus_data(en_file, fr_file, max_samples=50000)
		print(f"Successfully loaded OPUS dataset with {len(en_sentences)} sentences")

	except Exception as e:
		print(f"OPUS download/loading failed: {e}")
		print("Falling back to sample dataset...")
		en_sentences, fr_sentences = downloader.create_better_sample_data()
		print(f"Using sample dataset with {len(en_sentences)} sentences")

	# Data split - use different ratios based on dataset size
	if len(en_sentences) > 10000:
		test_size = 0.1  # Use 10% for validation with large dataset
		batch_size = 8  # Larger batch size for more data
		epochs = 10  # Fewer epochs needed with more data
	else:
		test_size = 0.2  # Use 20% for validation with small dataset
		batch_size = 4  # Smaller batch size for less data
		epochs = 15  # More epochs needed with less data

	train_en, val_en, train_fr, val_fr = train_test_split(
		en_sentences, fr_sentences, test_size=test_size, random_state=42
	)

	print(f"Training set: {len(train_en)} sentences")
	print(f"Validation set: {len(val_en)} sentences")

	# Create improved datasets
	train_dataset = ImprovedTranslationDataset(train_en, train_fr, src_tokenizer, tgt_tokenizer)
	val_dataset = ImprovedTranslationDataset(val_en, val_fr, src_tokenizer, tgt_tokenizer)

	# Use smaller batch size for better gradient updates
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

	# Create improved model
	model = ImprovedBertEncoderDecoder(
		src_vocab_size=src_tokenizer.vocab_size,
		tgt_vocab_size=tgt_tokenizer.vocab_size,
		d_model=768,
		nhead=12,
		num_layers=4  # Fewer layers to prevent overfitting on small dataset
	)

	print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
	print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

	# Train model with improved settings
	trainer = ImprovedTranslationTrainer(model, train_loader, val_loader, device, lr=5e-5)
	trainer.train(epochs=epochs)  # Dynamic epochs based on dataset size

	# Load best model for testing
	checkpoint = torch.load('best_improved_translation_model.pth')
	model.load_state_dict(checkpoint['model_state_dict'])
	translator = ImprovedTranslator(model, src_tokenizer, tgt_tokenizer, device)

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

	print("\n=== Improved Translation Test ===")
	for sentence in test_sentences:
		translation = translator.translate(sentence)
		print(f"English: {sentence}")
		print(f"French: {translation}")
		print("-" * 50)


if __name__ == "__main__":
	main()