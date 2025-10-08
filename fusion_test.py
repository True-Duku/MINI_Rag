import PyPDF2
import re
from langchain.text_splitter import TokenTextSplitter
from transformers import AutoModel, AutoTokenizer
import torch 
import faiss
#import pdfplumber
#import os
import torch
import torch.nn as nn
import numpy as np
from transformers import FlaubertTokenizer
from tqdm import tqdm
import sys


import numpy as np

# Ouvrir le fichier PDF en mode lecture binaire
with open("C:\\Users\\thier\\Desktop\Metabase_Dashboard\\Rag_Install_Diverse\\PDF_Stockage\\Fin_Du_Data_Analyste_.pdf", "rb") as fichier:
    # Créer un lecteur PDF
    lecteur = PyPDF2.PdfReader(fichier)
    
    # Initialiser une variable pour stocker le texte extrait
    texte_extrait = ""
    
    # Boucler à travers chaque page et extraire le texte
    for page_num in range(len(lecteur.pages)):
        page = lecteur.pages[page_num]
        texte_extrait += page.extract_text() + "\n"

    
# Afficher le texte extrait
#print(texte_extrait)
# nettoyage du texte
texte_brut = texte_extrait

def nettoyer_texte(texte):
   # Supprimer les espaces supplémentaires et les sauts de ligne
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte

# Appliquer la fonction de nettoyage
texte_nettoye = nettoyer_texte(texte_brut)
#print(texte_nettoye)

# tokenization

# Texte à tokeniser
#texte_nettoye = "Voici un exemple de texte à découper en tokens."

# Créer une instance de TokenTextSplitter
splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)

# Découper le texte en morceaux (tokens)
tokens = splitter.split_text(texte_nettoye)

# Afficher les tokens
#print(tokens)

# Charger le modèle et le tokenizer
model_name = "sentence-transformers/labse"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedder = AutoModel.from_pretrained(model_name)

# Fonction pour vectoriser le texte
def vectorize_text(text):
   inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
   with torch.no_grad():
       embeddings = embedder(**inputs).last_hidden_state.mean(dim=1)
   return embeddings

# Vectoriser le gros texte
vector = vectorize_text(tokens)

# Convertir le vecteur en un tableau NumPy (doit être de type float32)
vector_np = np.array(vector, dtype=np.float32).reshape(1, -1)  # Reshape pour ê
# Étape 1: Créer un index FAISS
dimension = vector_np.shape[1]  # Dimension de vos vecteurs
index = faiss.IndexFlatL2(dimension)  # Index basé sur la distance L2
# Étape 2: Ajouter le vecteur à l'index
index.add(vector_np)
# Étape 3: Sauvegarder l'index (optionnel)
#faiss.write_index(index, 'mon_index_faiss.index')
print("Vecteur ajouté à l'index avec succès.")

# === SETUP ===
tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")
vocab_size = tokenizer.vocab_size

# === Détection automatique du GPU ou CPU
def safe_device():
    try:
        torch.cuda.empty_cache()
        torch.tensor([0.0]).cuda()
        print("Entraînement sur GPU")
        return torch.device("cuda")
    except:
        print("GPU indisponible ou mémoire insuffisante — bascule sur CPU")
        return torch.device("cpu")

device = safe_device()

# === MiniGPT ===
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, max_len=50, num_heads=16, ff_dim=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.positional_enc = self.create_positional_encoding(max_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, ff_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                div_term = 10000 ** (i / d_model)
                pe[pos, i] = np.sin(pos / div_term)
                if i + 1 < d_model:
                    pe[pos, i + 1] = np.cos(pos / div_term)
        return pe.unsqueeze(0)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        x = self.token_emb(input_ids) + self.positional_enc[:, :seq_len, :].to(input_ids.device)
        x = x.transpose(0, 1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_ids.device)
        memory = torch.zeros(1, input_ids.size(0), x.size(-1), device=input_ids.device)
        decoded = self.decoder(x, memory, tgt_mask=mask)
        return self.out_proj(decoded.transpose(0, 1))

# === Chargement du corpus en blocs ===
def stream_corpus_in_blocks(file_path, block_size):
    with open(file_path, "r", encoding="utf-8") as f:
        block = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            block.append(line)
            if len(block) >= block_size:
                yield block
                block = []
        if block:
            yield block

# === Fonction d'entraînement ===
def train(model, inputs, targets, optimizer, criterion, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for input_ids, target_ids in zip(inputs, targets):
            optimizer.zero_grad()
            output = model(input_ids)
            loss = criterion(output.view(-1, vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Époque {epoch + 1} terminée — Perte totale : {total_loss:.4f}")

# === Génération de texte ===
def generate(model, start_text, max_len=20):
    model.eval()
    input_ids = tokenizer.encode(start_text, return_tensors="pt").to(device)
    generated = input_ids.tolist()[0]
    for _ in range(max_len):
        input_tensor = torch.tensor([generated], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        probs = torch.softmax(output[0, -1], dim=0)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated)

# === Initialisation du modèle et optimisation ===
llm_model = MiniGPT(vocab_size).to(device)
optimizer = torch.optim.Adam(llm_model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

# === Paramètres de corpus ===
corpus_path = "Corpus.txt"  # à adapter
block_size = 5000

# === Boucle principale d'entraînement avec sauvegarde et bascule CPU/GPU ===
for block_num, lines in enumerate(stream_corpus_in_blocks(corpus_path, block_size)):
    print(f"\n Bloc {block_num + 1} — {len(lines)} phrases")
    inputs, targets = [], []

    for line in tqdm(lines, desc=f"Préparation du bloc {block_num + 1}"):
        tokenized = tokenizer(line, return_tensors="pt", truncation=True, max_length=50)["input_ids"]
        target = tokenized.clone()
        target[:, :-1] = tokenized[:, 1:]
        inputs.append(tokenized.to(device))
        targets.append(target.to(device))

    try:
        train(llm_model, inputs, targets, optimizer, criterion)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("Mémoire GPU saturée — bascule sur CPU pour ce bloc")
            device = torch.device("cpu")
            llm_model.to(device)
            inputs = [i.cpu() for i in inputs]
            targets = [t.cpu() for t in targets]
            train(llm_model, inputs, targets, optimizer, criterion)
        else:
            raise e

    # Sauvegarde du modèle après chaque bloc
    #checkpoint_name = f"minigpt_bloc_{block_num + 1:02d}.pth"
    #torch.save(model.state_dict(), checkpoint_name)
    #print(f" Modèle sauvegardé : {checkpoint_name}")

# === SAUVEGARDE FINALE UNIQUE DU MODÈLE ===
# Ce bloc de code est en dehors de la boucle,
# donc il est exécuté une seule fois à la fin.
print("\nSauvegarde finale du modèle...")
model_path = "mini_gpt_model_final.pth"
torch.save(llm_model.state_dict(), model_path)
print(f"Modèle final sauvegardé dans {model_path}")

# === Génération interactive ANCIEN PROMPT ===
#user_input = input("Pose-moi une question : ")
#response = generate(model, user_input)
#print("Texte généré :", response)

# === Génération interactive avec Orchestration RAG ===

# 1. Démarrer l'interaction
user_input = input("Pose-moi une question : ")

# --- Début de l'Orchestration RAG (Retrieval-Augmented Generation) ---

# 2. Préparation pour la Recherche (Retrieval)
# La question doit être vectorisée. La fonction vectorize_text retourne un tenseur PyTorch.
# On utilise le même tokenizer et modèle que pour le PDF.
try:
    # Vectoriser la question (on s'assure que le tenseur est sur le CPU pour la conversion NumPy)
    question_vector_pt = vectorize_text(user_input).cpu() 
except Exception as e:
    print(f"Erreur lors de la vectorisation de la question : {e}")
    # Gérer l'erreur ou sortir si la vectorisation échoue
    sys.exit()

# Convertir le tenseur PyTorch en tableau NumPy (float32) pour FAISS
# Note: On utilise .numpy() et .reshape(1, -1) est crucial pour la recherche FAISS
question_vector_np = question_vector_pt.numpy().astype('float32').reshape(1, -1)

# 3. Recherche dans la Base de Connaissances (FAISS)
K = 3  # Nombre de morceaux de texte les plus pertinents à récupérer (ajustable)

try:
    # index.search(vecteur_de_recherche, nombre_de_résultats)
    # D: Distances (non utilisées ici), I: Indices (les positions dans la liste 'tokens')
    D, I = index.search(question_vector_np, K)
except Exception as e:
    print(f"Erreur lors de la recherche FAISS : {e}")
    # Si la recherche échoue, on continue avec la question brute
    I = np.array([[]])


# 4. Récupérer les Morceaux de Texte Correspondants
contexte_pertinent = ""

# Assurez-vous que des indices valides ont été trouvés
if I.size > 0 and I[0].size > 0:
    for index_resultat in I[0]:
        # On vérifie que l'index est valide (pour éviter les erreurs d'indice si K est trop grand)
        if 0 <= index_resultat < len(tokens):
            morceau_de_texte = tokens[index_resultat]
            # Ajouter le contexte avec une séparation claire pour le LLM
            contexte_pertinent += f"\n\n--- MORCEAU {index_resultat + 1} ---\n{morceau_de_texte}"
        # else: Un index invalide a été retourné, on l'ignore

# 5. Augmentation du Prompt
if contexte_pertinent:
    # Créer le prompt structuré avec le contexte
    prompt_augmente = (
        "Utilise le CONTEXTE fourni ci-dessous pour répondre précisément à la QUESTION.\n"
        "Si l'information n'est pas présente dans le CONTEXTE, réponds que tu ne peux pas répondre.\n"
        "CONTEXTE:\n"
        f"{contexte_pertinent}\n\n"
        f"QUESTION: {user_input}"
    )
else:
    # Fallback si FAISS n'a rien trouvé (par exemple, si le document est vide ou la recherche a échoué)
    print("ATTENTION : Aucun contexte pertinent trouvé. Le modèle répondra sans RAG.")
    prompt_augmente = user_input # Utilise la question brute comme fallback

# --- Fin de l'Orchestration RAG ---

# 6. Génération de la Réponse
# On passe le prompt AUGMENTÉ au modèle MiniGPT
response = generate(llm_model, prompt_augmente)

# 7. Affichage du résultat
print("\n--- PROMPT ENVOYÉ AU LLM ---")
print(prompt_augmente)
print("\n--- RÉPONSE DU MODÈLE ---")
print("Texte généré :", response)
 

