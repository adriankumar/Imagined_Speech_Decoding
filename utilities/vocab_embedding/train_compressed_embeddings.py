import torch, numpy as np
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def get_target_similarity(embeddings):
    #normaliuse true embeddings to unit length so matmul = cosine similarity
    norm_emb = F.normalize(torch.tensor(embeddings), p=2, dim=1) #vocab_size x embedding_dim
    return torch.mm(norm_emb, norm_emb.t()) #vocab_size x vocab_size

#objective function, minimise loss for compressed embeddings retaining
#semantic structure of original/true embeddings
#call it stress to 'minimise stress' as it's essentially a de-noising model
#where compressed embeddings is not an output but the parameters itself
def calculate_stress(compressed_embeddings, target_similarities):
    compressed_norm = F.normalize(compressed_embeddings, p=2, dim=1)
    current_similarities = torch.mm(compressed_norm, compressed_norm.t())
    return F.mse_loss(current_similarities, target_similarities)

def plot_compressed_similarity(compressed_embeddings, labels, sample_size=30):
    # 1. Calculate the full similarity matrix for math/stats
    full_sim = cosine_similarity(compressed_embeddings)
    
    # 2. Slice a small portion for the VISUAL heatmap
    # Plotting 800+ words with labels is impossible to read anyway
    sample_sim = full_sim[:sample_size, :sample_size]
    sample_labels = labels[:sample_size]
    
    plt.figure(figsize=(12, 10))
    # We remove annot=True if the sample_size > 20 for readability
    sns.heatmap(sample_sim, 
                annot=sample_size <= 25, 
                xticklabels=sample_labels, 
                yticklabels=sample_labels, 
                cmap="YlGnBu")
    
    plt.title(f"Semantic Relationships (First {sample_size} words)")
    plt.show()

#for policy decoding, you have to normalise the predictions
#which is in shape embedding dim x thought steps, so that matmul 
#yields the correct cosine similarity
def load_vocab_embedding(path, device, normalise=True):
    vocab_matrix = torch.load(path, map_location=device)

    #normalise embeddings so all vectors have length of 1
    #then simple dot product with it is cosine similarity score, ready for use
    #for decoding, but you must ensure the same vocab list used for this vocab embedding
    #must be the same for the model otherwise the indexing will produce different words
    if normalise:
        vocab_matrix_normed = F.normalize(vocab_matrix, p=2, dim=1)
    else:
        vocab_matrix_normed = None

    print(f"Loaded vocab embedding: {vocab_matrix.shape} on {device}")

    return vocab_matrix, vocab_matrix_normed

#MDS training (multidimensional scale)
def train_vocab_compression(compressed_dim, vocab_list, device, lr, epochs, save_path):
    encoder = SentenceTransformer('all-mpnet-base-v2') #load encoder for embeddings

    vocab_size = len(vocab_list)
    #create vocab embedding initialised with random noise
    vocab_embedding = torch.nn.Parameter(torch.randn(vocab_size, compressed_dim, device=device))
    optim = torch.optim.Adam([vocab_embedding], lr=lr)

    true_embeddings = encoder.encode(vocab_list) #<PAD> key should already be in vocab list
    #get target similarities from true embeddings
    target_sim = get_target_similarity(true_embeddings).to(device)

    epoch_bar = tqdm(range(epochs), desc=f'training iterations')

    for step in epoch_bar:
        optim.zero_grad() #zero gradients 
        loss = calculate_stress(vocab_embedding, target_sim)
        loss.backward() #calculate gradients to minimise cosine difference
        optim.step() #update embedding parameters

        if step % 1000 == 0:
            #quick sanity check
            with torch.no_grad():
                compressed_sim = torch.mm(F.normalize(vocab_embedding, p=2, dim=1),
                                          F.normalize(vocab_embedding, p=2, dim=1).t())
                
                score = np.corrcoef(target_sim.cpu().flatten(), compressed_sim.cpu().flatten())[0, 1]
                # print(f"Step {step} | Stress: {loss.item():.6f} | Preservation: {score:.4f}")
        
        epoch_bar.set_postfix({'loss': f'{loss.item():.4f}', 'Preservation': f'{score:.4f}'})
    
    #save after training
    torch.save(vocab_embedding.detach().cpu(), save_path)
    print(f"Training complete! embedding saved to: {save_path} | vocab embedding shape: {vocab_embedding.detach().cpu().shape}")