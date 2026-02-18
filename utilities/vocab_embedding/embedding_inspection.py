import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


#-----------------------------------------------
#Embedding space visualisation
#-----------------------------------------------
def plot_vocab_embeddings_2d(vocab_embeddings, vocab_list, method='pca', figsize=(12, 10), 
                              annotate=True, max_annotations=50):
    #visualizes vocab embeddings in 2d using pca or tsne
    #checks if semantically similar words cluster together
    
    #handle tensor input
    if torch.is_tensor(vocab_embeddings):
        embeddings = vocab_embeddings.detach().cpu().numpy()
    else:
        embeddings = vocab_embeddings
    
    #dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        explained_var = reducer.explained_variance_ratio_
        title = f'vocab embeddings (PCA) - explained var: {sum(explained_var):.2%}'
    elif method.lower() == 'tsne':
        #tsne needs perplexity < n_samples
        perplexity = min(30, len(vocab_list) - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        reduced = reducer.fit_transform(embeddings)
        title = 'vocab embeddings (t-SNE)'
    else:
        raise ValueError(f"method must be 'pca' or 'tsne', got {method}")
    
    plt.figure(figsize=figsize)
    
    #plot points
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=50)
    
    #annotate points with word labels
    if annotate:
        n_annotate = min(max_annotations, len(vocab_list))
        #prioritize non-PAD words
        indices = list(range(len(vocab_list)))
        if '<PAD>' in vocab_list:
            pad_idx = vocab_list.index('<PAD>')
            indices.remove(pad_idx)
            indices = [pad_idx] + indices  #put PAD first so it gets labeled
        
        for i in indices[:n_annotate]:
            plt.annotate(vocab_list[i], (reduced[i, 0], reduced[i, 1]), 
                        fontsize=8, alpha=0.7,
                        xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('dimension 1')
    plt.ylabel('dimension 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return reduced


def compute_vocab_similarity_matrix(vocab_embeddings, vocab_list, figsize=(14, 12)):
    #computes and plots pairwise cosine similarity heatmap
    
    #handle tensor input
    if torch.is_tensor(vocab_embeddings):
        embeddings = vocab_embeddings.detach().cpu().numpy()
    else:
        embeddings = vocab_embeddings
    
    #compute cosine similarity matrix
    sim_matrix = sklearn_cosine(embeddings)
    
    #plot heatmap
    plt.figure(figsize=figsize)
    
    #if vocab is small enough, show labels
    if len(vocab_list) <= 30:
        plt.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.xticks(range(len(vocab_list)), vocab_list, rotation=90, fontsize=8)
        plt.yticks(range(len(vocab_list)), vocab_list, fontsize=8)
    else:
        plt.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.xlabel('word index')
        plt.ylabel('word index')
    
    plt.colorbar(label='cosine similarity')
    plt.title(f'vocab embedding similarity matrix ({len(vocab_list)} words)')
    plt.tight_layout()
    plt.show()
    
    #print statistics
    print("=" * 50)
    print("SIMILARITY MATRIX STATISTICS")
    print("=" * 50)
    
    #exclude diagonal for stats
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    off_diag = sim_matrix[mask]
    
    print(f"  mean similarity: {off_diag.mean():.4f}")
    print(f"  std similarity: {off_diag.std():.4f}")
    print(f"  min similarity: {off_diag.min():.4f}")
    print(f"  max similarity: {off_diag.max():.4f}")
    
    #check for highly similar pairs (potential issue)
    high_sim_threshold = 0.9
    high_sim_count = np.sum(off_diag > high_sim_threshold)
    print(f"  pairs with sim > {high_sim_threshold}: {high_sim_count}")
    
    print("=" * 50)
    
    return sim_matrix


def check_semantic_clustering(vocab_embeddings, vocab_list, n_neighbors=5):
    #prints nearest neighbors for each word to sanity check semantic structure
    
    #handle tensor input
    if torch.is_tensor(vocab_embeddings):
        embeddings = vocab_embeddings.detach().cpu().numpy()
    else:
        embeddings = vocab_embeddings
    
    #compute similarity matrix
    sim_matrix = sklearn_cosine(embeddings)
    
    print("=" * 60)
    print(f"SEMANTIC CLUSTERING CHECK (top {n_neighbors} neighbors)")
    print("=" * 60)
    
    for i, word in enumerate(vocab_list):
        #get similarities for this word (exclude self)
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  #exclude self
        
        #get top k neighbors
        top_indices = np.argsort(sims)[-n_neighbors:][::-1]
        
        neighbors = []
        for idx in top_indices:
            neighbors.append(f"{vocab_list[idx]} ({sims[idx]:.3f})")
        
        print(f"\n  '{word}' →")
        print(f"    {', '.join(neighbors)}")
    
    print("\n" + "=" * 60)


def check_semantic_clustering_sample(vocab_embeddings, vocab_list, sample_words=None, n_neighbors=5):
    #lighter version - only checks specified sample words
    
    #handle tensor input
    if torch.is_tensor(vocab_embeddings):
        embeddings = vocab_embeddings.detach().cpu().numpy()
    else:
        embeddings = vocab_embeddings
    
    #default sample words if not provided
    if sample_words is None:
        #pick evenly spaced words from vocab
        n_samples = min(10, len(vocab_list))
        indices = np.linspace(0, len(vocab_list)-1, n_samples, dtype=int)
        sample_words = [vocab_list[i] for i in indices]
    
    #filter to words that exist in vocab
    sample_words = [w for w in sample_words if w in vocab_list]
    
    if not sample_words:
        print("no valid sample words found in vocab")
        return
    
    #compute similarity matrix
    sim_matrix = sklearn_cosine(embeddings)
    
    print("=" * 60)
    print(f"SEMANTIC CLUSTERING SAMPLE (top {n_neighbors} neighbors)")
    print("=" * 60)
    
    for word in sample_words:
        i = vocab_list.index(word)
        
        #get similarities (exclude self)
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf
        
        #get top k
        top_indices = np.argsort(sims)[-n_neighbors:][::-1]
        
        neighbors = []
        for idx in top_indices:
            neighbors.append(f"{vocab_list[idx]} ({sims[idx]:.3f})")
        
        print(f"\n  '{word}' →")
        print(f"    {', '.join(neighbors)}")
    
    print("\n" + "=" * 60)


#-----------------------------------------------
#Embedding space quality metrics
#-----------------------------------------------
def compute_embedding_quality_metrics(vocab_embeddings, vocab_list):
    #computes various metrics to assess embedding space quality
    
    #handle tensor input
    if torch.is_tensor(vocab_embeddings):
        embeddings = vocab_embeddings.detach().cpu().numpy()
    else:
        embeddings = vocab_embeddings
    
    metrics = {}
    
    #1. basic stats
    metrics['embedding_dim'] = embeddings.shape[1]
    metrics['vocab_size'] = embeddings.shape[0]
    metrics['mean_norm'] = np.linalg.norm(embeddings, axis=1).mean()
    metrics['std_norm'] = np.linalg.norm(embeddings, axis=1).std()
    
    #2. similarity distribution
    sim_matrix = sklearn_cosine(embeddings)
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    off_diag = sim_matrix[mask]
    
    metrics['mean_similarity'] = off_diag.mean()
    metrics['std_similarity'] = off_diag.std()
    metrics['min_similarity'] = off_diag.min()
    metrics['max_similarity'] = off_diag.max()
    
    #3. isotropy score (how uniformly distributed in space)
    #perfect isotropy = embeddings uniformly distributed on hypersphere
    #approximated by: mean similarity close to 0, std similarity moderate
    metrics['isotropy_score'] = 1.0 - abs(metrics['mean_similarity'])
    
    #4. effective dimensionality via pca
    pca = PCA(random_state=42)
    pca.fit(embeddings)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    metrics['dims_for_90_var'] = int(np.argmax(cumvar >= 0.9) + 1)
    metrics['dims_for_95_var'] = int(np.argmax(cumvar >= 0.95) + 1)
    
    #5. collapse detection - are embeddings clustering into few points?
    #use ratio of unique rounded embeddings
    rounded = np.round(embeddings, decimals=2)
    unique_rows = len(np.unique(rounded, axis=0))
    metrics['uniqueness_ratio'] = unique_rows / len(embeddings)
    
    #print results
    print("=" * 60)
    print("EMBEDDING QUALITY METRICS")
    print("=" * 60)
    print(f"\n[BASIC STATS]")
    print(f"  vocab size: {metrics['vocab_size']}")
    print(f"  embedding dim: {metrics['embedding_dim']}")
    print(f"  mean norm: {metrics['mean_norm']:.4f} ± {metrics['std_norm']:.4f}")
    
    print(f"\n[SIMILARITY DISTRIBUTION]")
    print(f"  mean: {metrics['mean_similarity']:.4f}")
    print(f"  std: {metrics['std_similarity']:.4f}")
    print(f"  range: [{metrics['min_similarity']:.4f}, {metrics['max_similarity']:.4f}]")
    
    print(f"\n[SPACE QUALITY]")
    print(f"  isotropy score: {metrics['isotropy_score']:.4f} (1.0 = perfect)")
    print(f"  dims for 90% var: {metrics['dims_for_90_var']} / {metrics['embedding_dim']}")
    print(f"  dims for 95% var: {metrics['dims_for_95_var']} / {metrics['embedding_dim']}")
    print(f"  uniqueness ratio: {metrics['uniqueness_ratio']:.4f} (1.0 = all unique)")
    
    #warnings
    print(f"\n[DIAGNOSTICS]")
    if metrics['mean_similarity'] > 0.5:
        print("  ⚠️  HIGH mean similarity - embeddings may be clustered/collapsed")
    if metrics['isotropy_score'] < 0.5:
        print("  ⚠️  LOW isotropy - embeddings not well distributed in space")
    if metrics['dims_for_90_var'] < metrics['embedding_dim'] * 0.3:
        print("  ⚠️  LOW effective dimensionality - not using full embedding space")
    if metrics['uniqueness_ratio'] < 0.9:
        print("  ⚠️  LOW uniqueness - some embeddings may be near-duplicates")
    if not any([metrics['mean_similarity'] > 0.5, 
                metrics['isotropy_score'] < 0.5,
                metrics['dims_for_90_var'] < metrics['embedding_dim'] * 0.3,
                metrics['uniqueness_ratio'] < 0.9]):
        print("  ✓ Embedding space appears healthy")
    
    print("=" * 60)
    
    return metrics


def plot_embedding_norms(vocab_embeddings, vocab_list, figsize=(12, 5)):
    #plots distribution of embedding norms - checks for outliers
    
    #handle tensor input
    if torch.is_tensor(vocab_embeddings):
        embeddings = vocab_embeddings.detach().cpu().numpy()
    else:
        embeddings = vocab_embeddings
    
    norms = np.linalg.norm(embeddings, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    n_bins = min(20, max(5, len(vocab_list) // 2))
    
    #histogram
    axes[0].hist(norms, bins=n_bins, edgecolor='black', alpha=0.7)
    axes[0].axvline(norms.mean(), color='red', linestyle='--', label=f'mean={norms.mean():.3f}')
    axes[0].set_xlabel('L2 norm')
    axes[0].set_ylabel('count')
    axes[0].set_title('embedding norm distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    #bar plot per word
    if len(vocab_list) <= 50:
        axes[1].bar(range(len(vocab_list)), norms, alpha=0.7)
        axes[1].set_xticks(range(len(vocab_list)))
        axes[1].set_xticklabels(vocab_list, rotation=90, fontsize=7)
    else:
        axes[1].bar(range(len(vocab_list)), norms, alpha=0.7)
        axes[1].set_xlabel('word index')
    axes[1].set_ylabel('L2 norm')
    axes[1].set_title('norm per word')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



def run_vocab_diagnostics(vocab_embeddings, vocab_list):
    #runs all phase 2 embedding space diagnostics
    
    print("\n" + "="*70)
    print("PHASE 2 DIAGNOSTICS - EMBEDDING SPACE ANALYSIS")
    print("="*70)
    
    #1. quality metrics
    print("\n[1/5] EMBEDDING QUALITY METRICS")
    metrics = compute_embedding_quality_metrics(vocab_embeddings, vocab_list)
    
    #2. norm distribution
    print("\n[2/5] EMBEDDING NORM DISTRIBUTION")
    plot_embedding_norms(vocab_embeddings, vocab_list)
    
    #3. similarity matrix
    print("\n[3/5] SIMILARITY MATRIX")
    sim_matrix = compute_vocab_similarity_matrix(vocab_embeddings, vocab_list)
    
    #4. 2d visualization
    print("\n[4/5] 2D VISUALIZATION (PCA)")
    reduced_pca = plot_vocab_embeddings_2d(vocab_embeddings, vocab_list, method='pca')
    
    #5. semantic clustering check
    print("\n[5/5] SEMANTIC CLUSTERING CHECK")
    if len(vocab_list) <= 20:
        check_semantic_clustering(vocab_embeddings, vocab_list)
    else:
        check_semantic_clustering_sample(vocab_embeddings, vocab_list)
    
    return {
        'metrics': metrics,
        'similarity_matrix': sim_matrix,
        'pca_reduced': reduced_pca
    }