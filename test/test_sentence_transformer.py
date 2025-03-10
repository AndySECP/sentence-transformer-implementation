import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from core.sentence_transformer import SentenceTransformer

test_sentences = {
    # Cluster 1: Technology and AI (should have high similarity within cluster)
    "tech_ai": [
        "Machine learning is transforming various industries.",
        "Artificial intelligence continues to advance rapidly.",
        "Deep learning models are becoming more sophisticated.",
        "Neural networks process vast amounts of data.",
        "AI algorithms are improving decision-making processes."
    ],

    # Cluster 2: Nature and Animals (should have high similarity within cluster)
    "nature": [
        "The quick brown fox jumps over the lazy dog.",
        "A sleek leopard stalks through tall grass.",
        "Wild wolves hunt in coordinated packs.",
        "Birds soar gracefully in the morning sky.",
        "Beautiful creatures roam the vast wilderness."
    ],

    "space_concepts": [
        "Galaxies contain billions of stars orbiting a central point.",
        "Stars are massive spheres of hot plasma held by gravity.",
        "Cosmic objects move through the vast expanse of space.",
        "Celestial bodies rotate around their galactic centers.",
        "Astronomical objects traverse the infinite cosmos.",
        "Stellar masses circulate through interstellar space."
    ],
}

# Test cases to evaluate:
evaluation_pairs = [
    # Within-cluster similarity (should be high)
    ("tech_ai", 0, "tech_ai", 1),
    ("nature", 0, "nature", 2),
    ("space_concepts", 1, "space_concepts", 3),

    # Cross-cluster similarity (should be low)
    ("tech_ai", 0, "nature", 0),
    ("space_concepts", 0, "nature", 0),
]

# Helper function to get sentence pairs for testing
def get_test_pairs():
    pairs = []
    for cat1, idx1, cat2, idx2 in evaluation_pairs:
        sent1 = test_sentences[cat1][idx1]
        sent2 = test_sentences[cat2][idx2]
        pairs.append((sent1, sent2))
    return pairs

# Get all sentences as a flat list
def get_all_sentences():
    return [sent for category in test_sentences.values() for sent in category]

def encode_sentences(model, tokenizer, sentences, device="cpu"):
    """
    Generate embeddings for a list of sentences using the custom SentenceTransformer model.
    """
    # Tokenize sentences
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    # Generate embeddings
    with torch.no_grad():
        embeddings = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"]
        )

    return embeddings

def compute_similarity_matrix(embeddings):
    """
    Compute cosine similarity matrix between all sentence pairs.
    """
    return torch.nn.functional.cosine_similarity(
        embeddings.unsqueeze(1),
        embeddings.unsqueeze(0),
        dim=-1
    )

def plot_similarity_heatmap(similarity_matrix, sentences, test_sentences):
    """
    Create a heatmap for high similarity categories.
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Define which categories should have high internal similarity
    high_similarity_categories = ['tech_ai', 'nature', 'space_concepts']

    # Create indices for high similarity sentences
    high_sim_indices = []
    high_sim_labels = []
    current_idx = 0
    for category, sents in test_sentences.items():
        if category in high_similarity_categories:
            indices = list(range(current_idx, current_idx + len(sents)))
            high_sim_indices.extend(indices)
            high_sim_labels.extend([f"{category}_{i+1}" for i in range(len(sents))])
        current_idx += len(sents)

    # Create the high similarity sub-matrix
    high_sim_matrix = similarity_matrix[np.ix_(high_sim_indices, high_sim_indices)]

    # Create figure
    plt.figure(figsize=(12, 10))

    # Plot high similarity heatmap
    sns.heatmap(
        high_sim_matrix,
        xticklabels=high_sim_labels,
        yticklabels=high_sim_labels,
        cmap='YlOrRd',
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title('High Similarity Categories')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Adjust layout
    plt.tight_layout()
    plt.savefig("output_similarity_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistics for the matrix
    print("\nHigh Similarity Matrix Statistics:")
    print(f"Mean similarity: {np.mean(high_sim_matrix):.3f}")
    print(f"Std deviation: {np.std(high_sim_matrix):.3f}")
    print(f"Min similarity: {np.min(high_sim_matrix):.3f}")
    print(f"Max similarity: {np.max(high_sim_matrix):.3f}")

def plot_pca_visualization(embeddings, sentences, clusters):
    """
    Create a PCA visualization for the embeddings with colors by cluster.
    """
    # Convert embeddings to numpy
    embeddings_np = embeddings.cpu().numpy()
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings_np)
    
    # Get variance explained
    variance_explained = pca.explained_variance_ratio_
    
    # Create color map
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot each cluster
    for idx, (cluster_name, indices) in enumerate(clusters.items()):
        cluster_points = embeddings_pca[indices]
        color = colors[idx % len(colors)]

        # Plot points
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[color],
            label=cluster_name,
            alpha=0.7
        )

        # Add labels for each point
        for i, point_idx in enumerate(indices):
            plt.annotate(
                f'S{point_idx + 1}',
                (embeddings_pca[point_idx, 0], embeddings_pca[point_idx, 1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )

    # Add title and labels
    plt.title(f"PCA Visualization\nVariance explained: {variance_explained[0]:.3f}, {variance_explained[1]:.3f}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    # Add legend
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=10,
        title="Clusters"
    )

    # Adjust layout
    plt.tight_layout()
    plt.savefig("output_pca_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Print additional PCA insights
    print("\nPCA Analysis:")
    print(f"Variance explained by first two components: {sum(variance_explained):.3f}")
    print(f"Individual component variance ratio: PC1 = {variance_explained[0]:.3f}, PC2 = {variance_explained[1]:.3f}")

    # Calculate and print cluster separations in PCA space
    print("\nCluster Separation Analysis (PCA space):")
    cluster_centers = {}
    for cluster_name, indices in clusters.items():
        cluster_points = embeddings_pca[indices]
        cluster_centers[cluster_name] = np.mean(cluster_points, axis=0)

    # Print distances between cluster centers
    print("\nCenter-to-center distances between clusters:")
    for name1 in cluster_centers:
        for name2 in cluster_centers:
            if name1 < name2:  # Print each pair only once
                dist = np.linalg.norm(cluster_centers[name1] - cluster_centers[name2])
                print(f"{name1} vs {name2}: {dist:.3f}")
                
    return embeddings_pca

def analyze_cluster_similarities(similarity_matrix, sentences, clusters):
    """
    Analyze and print similarities within and between clusters.
    """
    n_clusters = len(clusters)
    cluster_stats = {}

    # Calculate within-cluster similarities
    for cluster_name, indices in clusters.items():
        if len(indices) > 1:
            cluster_similarities = []
            for i in indices:
                for j in indices:
                    if i != j:
                        cluster_similarities.append(similarity_matrix[i, j].item())

            cluster_stats[cluster_name] = {
                'mean': np.mean(cluster_similarities),
                'std': np.std(cluster_similarities),
                'min': np.min(cluster_similarities),
                'max': np.max(cluster_similarities)
            }

    return pd.DataFrame(cluster_stats).transpose()

def test_main():
    # Initialize model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        model_name="bert-base-uncased",
        embedding_dim=768,
        pooling_strategy="mean",
        normalize=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    sentences = get_all_sentences()

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = encode_sentences(model, tokenizer, sentences, device)
    print(f"Generated embeddings shape: {embeddings.shape}")

    # Compute similarity matrix
    print("\nComputing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(embeddings)

    # Plot heatmap
    print("\nPlotting similarity heatmap...")
    plot_similarity_heatmap(
        similarity_matrix.cpu().numpy(),
        sentences,
        test_sentences
    )

    # Create clusters dictionary with indices
    clusters = {}
    current_idx = 0
    for category, sents in test_sentences.items():
        clusters[category] = list(range(current_idx, current_idx + len(sents)))
        current_idx += len(sents)

    # Create PCA visualization
    print("\nCreating PCA visualization...")
    plot_pca_visualization(embeddings, sentences, clusters)

    # Analyze cluster similarities
    print("\nAnalyzing cluster similarities...")
    cluster_stats = analyze_cluster_similarities(similarity_matrix, sentences, clusters)
    print("\nCluster Statistics:")
    print(cluster_stats)

    # Print example similarities
    print("\nExample sentence similarities:")
    for i in range(min(3, len(sentences))):
        for j in range(i + 1, min(i + 3, len(sentences))):
            print(f"\nSentence {i+1}: {sentences[i][:50]}...")
            print(f"Sentence {j+1}: {sentences[j][:50]}...")
            print(f"Similarity: {similarity_matrix[i, j].item():.3f}")

if __name__ == "__main__":
    test_main()
