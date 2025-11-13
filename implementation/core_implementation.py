
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import List, Tuple, Dict, Set
import networkx as nx
from scipy.spatial.distance import cosine


class SocialNetwork:
    """
    Represents a social network with topic-aware user profiles and propagation models.
    
    This class implements the topic-aware extensions to IC and LT models as described 
    in Section 3.1-3.3 of the paper.
    """
    
    def __init__(self, graph: nx.DiGraph, user_profiles: Dict[int, np.ndarray], 
                 alpha1: float = 0.8, alpha2: float = 0.8, alpha3: float = 1.0):
        """
        Initialize the social network with topic-aware capabilities.
        
        Args:
            graph: NetworkX directed graph representing social network
            user_profiles: Dictionary mapping user_id to topic interest vector
            alpha1, alpha2, alpha3: Balance parameters for contact frequency, 
                                   similarity, and benefit respectively
        """
        self.graph = graph
        self.user_profiles = user_profiles
        self.alpha1, self.alpha2, self.alpha3 = alpha1, alpha2, alpha3
        
        # Normalize user profiles to sum to 1
        for user_id in user_profiles:
            profile_sum = np.sum(user_profiles[user_id])
            if profile_sum > 0:
                user_profiles[user_id] = user_profiles[user_id] / profile_sum
                
        # Precompute user similarities for efficiency
        self.user_similarities = self._compute_user_similarities()
        
    def _compute_user_similarities(self) -> Dict[Tuple[int, int], float]:
        """
        Precompute cosine similarities between all user pairs.
        
        Returns:
            Dictionary mapping (user1, user2) to their cosine similarity
        """
        similarities = {}
        users = list(self.user_profiles.keys())
        
        for i, u in enumerate(users):
            for j, v in enumerate(users):
                if i <= j:  # Compute only upper triangle (symmetric)
                    profile_u = self.user_profiles[u]
                    profile_v = self.user_profiles[v]
                    
                    # Cosine similarity calculation
                    dot_product = np.dot(profile_u, profile_v)
                    norm_u = np.linalg.norm(profile_u)
                    norm_v = np.linalg.norm(profile_v)
                    
                    if norm_u > 0 and norm_v > 0:
                        sim = dot_product / (norm_u * norm_v)
                    else:
                        sim = 0.0
                        
                    similarities[(u, v)] = sim
                    similarities[(v, u)] = sim  # Symmetric
                    
        return similarities
    
    def get_user_similarity(self, u: int, v: int) -> float:
        """Get precomputed similarity between users u and v."""
        return self.user_similarities.get((u, v), 0.0)
    
    def get_user_benefit(self, user: int, topics: List[int]) -> float:
        """
        Calculate user benefit for given topics.
        
        Args:
            user: User ID
            topics: List of topic indices
            
        Returns:
            Sum of user's interest levels for the given topics
        """
        if user not in self.user_profiles:
            return 0.0
        
        profile = self.user_profiles[user]
        benefit = sum(profile[t] if t < len(profile) else 0.0 for t in topics)
        return min(benefit, 1.0)  # Cap at 1.0
    
    def get_topic_aware_ic_probability(self, u: int, v: int, topics: List[int]) -> float:
        """
        Calculate topic-aware activation probability for IC model.
        
        Implements Equation (1) from the paper:
        p_uv = (α₁w_uv + α₂sim(u,v) + α₃B_v^τ) / 3
        
        Args:
            u, v: Source and target users
            topics: Query topic indices
            
        Returns:
            Activation probability for edge (u,v) under given topics
        """
        if not self.graph.has_edge(u, v):
            return 0.0
            
        # Contact frequency (initial edge weight)
        w_uv = self.graph[u][v].get('weight', 0.5)  # Default weight if not specified
        
        # User similarity
        sim_uv = self.get_user_similarity(u, v)
        
        # User benefit for topics
        benefit_v = self.get_user_benefit(v, topics)
        
        # Topic-aware probability calculation
        prob = (self.alpha1 * w_uv + self.alpha2 * sim_uv + self.alpha3 * benefit_v) / 3.0
        
        return min(max(prob, 0.0), 1.0)  # Clamp to [0,1]
    
    def get_topic_aware_lt_threshold(self, v: int, topics: List[int]) -> float:
        """
        Calculate topic-aware threshold for LT model.
        
        Implements θ_v = 1 - α₃B_v^τ from Section 3.3
        
        Args:
            v: Target user
            topics: Query topic indices
            
        Returns:
            Activation threshold for user v under given topics
        """
        benefit_v = self.get_user_benefit(v, topics)
        threshold = 1.0 - self.alpha3 * benefit_v
        return min(max(threshold, 0.0), 1.0)
    
    def get_topic_aware_lt_weight(self, u: int, v: int, topics: List[int]) -> float:
        """
        Calculate topic-aware edge weight for LT model.
        
        Implements b_uv = α₁w_uv + α₂sim(u,v) from Section 3.3
        
        Args:
            u, v: Source and target users
            topics: Query topic indices
            
        Returns:
            Edge weight for (u,v) in LT model
        """
        if not self.graph.has_edge(u, v):
            return 0.0
            
        w_uv = self.graph[u][v].get('weight', 0.5)
        sim_uv = self.get_user_similarity(u, v)
        
        weight = self.alpha1 * w_uv + self.alpha2 * sim_uv
        return min(max(weight, 0.0), 1.0)


class Diffusion2Vec(nn.Module):
    """
    Graph embedding network for TIM as described in Section 3.4.
    
    This implements the iterative embedding update mechanism that captures
    T-hop neighborhood information including user profiles and topic-aware
    propagation probabilities.
    """
    
    def __init__(self, num_topics: int, embedding_dim: int = 64, num_iterations: int = 4):
        """
        Initialize Diffusion2Vec network.
        
        Args:
            num_topics: Number of topics in the system
            embedding_dim: Dimension of node embeddings (p in paper)
            num_iterations: Number of embedding iterations (T in paper)
        """
        super(Diffusion2Vec, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_iterations = num_iterations
        self.num_topics = num_topics
        
        # Feature dimension: selection_indicator (1) + user_profile (num_topics)
        feature_dim = 1 + num_topics
        
        # Network parameters as defined in Equation (3)
        self.theta1 = nn.Linear(feature_dim, embedding_dim)      # Θ₁: node features
        self.theta2 = nn.Linear(embedding_dim, embedding_dim)    # Θ₂: neighbor embeddings  
        self.theta3 = nn.Linear(embedding_dim, embedding_dim)    # Θ₃: weighted aggregation
        self.theta4 = nn.Linear(1, embedding_dim)               # Θ₄: edge weights
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize network parameters using Xavier initialization."""
        for layer in [self.theta1, self.theta2, self.theta3, self.theta4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor, edge_weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Diffusion2Vec.
        
        Args:
            node_features: [N, feature_dim] tensor of node features
            adjacency_matrix: [N, N] binary adjacency matrix
            edge_weights: [N, N] tensor of topic-aware edge probabilities
            
        Returns:
            [N, embedding_dim] tensor of final node embeddings
        """
        batch_size, num_nodes = node_features.shape[0], adjacency_matrix.shape[0]
        device = node_features.device
        
        # Initialize embeddings to zero as per paper
        embeddings = torch.zeros(batch_size, num_nodes, self.embedding_dim, device=device)
        
        # Iterative embedding updates for T iterations
        for t in range(self.num_iterations):
            new_embeddings = torch.zeros_like(embeddings)
            
            for v in range(num_nodes):
                # Node feature transformation: Θ₁X_v
                feature_term = self.theta1(node_features[:, v])  # [batch_size, embedding_dim]
                
                # Neighbor embedding aggregation: Θ₂ Σ_{u∈N(v)} u_u^(t)
                neighbors = adjacency_matrix[v].nonzero(as_tuple=False).squeeze(-1)
                if len(neighbors) > 0:
                    neighbor_embeddings = embeddings[:, neighbors]  # [batch_size, |N(v)|, embedding_dim]
                    neighbor_sum = torch.sum(neighbor_embeddings, dim=1)  # [batch_size, embedding_dim]
                    neighbor_term = self.theta2(neighbor_sum)
                else:
                    neighbor_term = torch.zeros_like(feature_term)
                
                # Weighted edge aggregation: Θ₃ Σ_{u∈N(v)} relu(Θ₄p(u,v))
                if len(neighbors) > 0:
                    edge_weights_v = edge_weights[neighbors, v].unsqueeze(-1)  # [|N(v)|, 1]
                    edge_weights_v = edge_weights_v.expand(batch_size, -1, -1)  # [batch_size, |N(v)|, 1]
                    
                    edge_embeddings = F.relu(self.theta4(edge_weights_v))  # [batch_size, |N(v)|, embedding_dim]
                    edge_sum = torch.sum(edge_embeddings, dim=1)  # [batch_size, embedding_dim]
                    edge_term = self.theta3(edge_sum)
                else:
                    edge_term = torch.zeros_like(feature_term)
                
                # Combine all terms with ReLU activation
                new_embeddings[:, v] = F.relu(feature_term + neighbor_term + edge_term)
            
            embeddings = new_embeddings
        
        return embeddings  # [batch_size, num_nodes, embedding_dim]
    
    def get_node_features(self, social_network: SocialNetwork, seed_set: Set[int],
                         topics: List[int]) -> torch.Tensor:
        """
        Construct node feature matrix X_v for all nodes.
        
        Args:
            social_network: SocialNetwork instance
            seed_set: Current seed set (selected nodes)
            topics: Query topics
            
        Returns:
            [num_nodes, feature_dim] tensor of node features
        """
        nodes = list(social_network.graph.nodes())
        num_nodes = len(nodes)
        feature_dim = 1 + self.num_topics
        
        features = torch.zeros(num_nodes, feature_dim)
        
        for i, node in enumerate(nodes):
            # Selection indicator: X_v[0] = 1 if v ∈ S, else 0
            features[i, 0] = 1.0 if node in seed_set else 0.0
            
            # User profile: X_v[1:] = P_v
            if node in social_network.user_profiles:
                profile = social_network.user_profiles[node]
                profile_len = min(len(profile), self.num_topics)
                features[i, 1:1+profile_len] = torch.from_numpy(profile[:profile_len]).float()
        
        return features


class DIEM(nn.Module):
    """
    Deep Influence Evaluation Model as described in Section 3.5.
    
    This neural network estimates the marginal influence of adding a candidate node
    to the current seed set under given topic constraints.
    """
    
    def __init__(self, embedding_dim: int):
        """
        Initialize DIEM network.
        
        Args:
            embedding_dim: Dimension of node embeddings from Diffusion2Vec
        """
        super(DIEM, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Network parameters as defined in Equation (4)
        self.theta5 = nn.Linear(2 * embedding_dim, 1)           # Θ₅: final output layer
        self.theta6 = nn.Linear(embedding_dim, embedding_dim)    # Θ₆: seed set transformation
        self.theta7 = nn.Linear(embedding_dim, embedding_dim)    # Θ₇: candidate transformation
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize network parameters."""
        for layer in [self.theta5, self.theta6, self.theta7]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, seed_embeddings: torch.Tensor, candidate_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to estimate marginal influence.
        
        Implements Q̂(S,v;Θ,τ) = Θ₅ relu([Θ₆ Σ_{u∈S} u_u^(T), Θ₇ u_v^(T)])
        
        Args:
            seed_embeddings: [batch_size, |S|, embedding_dim] embeddings of seed set
            candidate_embedding: [batch_size, embedding_dim] embedding of candidate node
            
        Returns:
            [batch_size, 1] estimated marginal influence values
        """
        # Aggregate seed set embeddings: Σ_{u∈S} u_u^(T)
        if seed_embeddings.shape[1] > 0:
            seed_aggregation = torch.sum(seed_embeddings, dim=1)  # [batch_size, embedding_dim]
        else:
            # Empty seed set case
            seed_aggregation = torch.zeros_like(candidate_embedding)
        
        # Transform aggregated seed embeddings: Θ₆ Σ_{u∈S} u_u^(T)
        seed_transform = self.theta6(seed_aggregation)  # [batch_size, embedding_dim]
        
        # Transform candidate embedding: Θ₇ u_v^(T)
        candidate_transform = self.theta7(candidate_embedding)  # [batch_size, embedding_dim]
        
        # Concatenate transformed representations: [Θ₆(...), Θ₇(...)]
        combined = torch.cat([seed_transform, candidate_transform], dim=-1)  # [batch_size, 2*embedding_dim]
        
        # Apply ReLU and final linear transformation: Θ₅ relu([...])
        output = self.theta5(F.relu(combined))  # [batch_size, 1]
        
        return output