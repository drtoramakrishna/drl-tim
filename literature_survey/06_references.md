# Comprehensive References and Further Reading

## Table of Contents
1. [Core Paper References](#1-core-paper-references)
2. [Deep Reinforcement Learning](#2-deep-reinforcement-learning)
3. [Graph Neural Networks](#3-graph-neural-networks)
4. [Influence Maximization](#4-influence-maximization)
5. [Topic Modeling and Social Networks](#5-topic-modeling-and-social-networks)
6. [Combinatorial Optimization](#6-combinatorial-optimization)
7. [Advanced Topics and Extensions](#7-advanced-topics-and-extensions)
8. [Implementation Resources](#8-implementation-resources)
9. [Datasets and Benchmarks](#9-datasets-and-benchmarks)
10. [Software Tools and Libraries](#10-software-tools-and-libraries)

---

## 1. Core Paper References

### Primary Research Paper
- **Rahul, R., Jha, P., & Kumar, A.** (2023). "Deep Reinforcement Learning‑Based Approach to Tackle Topic‑Aware Influence Maximization." *Journal of Computational Science*, 45, 102-115.

### Foundational Works Cited in the Paper

#### Influence Maximization Foundations
- **Kempe, D., Kleinberg, J., & Tardos, E.** (2003). "Maximizing the spread of influence through a social network." *Proceedings of the 9th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 137-146. [DOI: 10.1145/956750.956769]

- **Chen, W., Wang, Y., & Yang, S.** (2009). "Efficient influence maximization in social networks." *Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 199-208.

- **Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C., VanBriesen, J., & Glance, N.** (2007). "Cost-effective outbreak detection in networks." *Proceedings of the 13th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 420-429.

#### Topic-Aware Extensions
- **Chen, W., Collins, A., Cummings, R., Ke, T., Liu, Z., Rincon, D., ... & Yuan, Y.** (2011). "Influence maximization in social networks when negative opinions may emerge and propagate." *Proceedings of the 2011 SIAM International Conference on Data Mining*, 379-390.

- **Barbieri, N., Bonchi, F., & Manco, G.** (2012). "Topic-aware social influence propagation models." *Proceedings of the 2012 IEEE 12th International Conference on Data Mining*, 81-90.

---

## 2. Deep Reinforcement Learning

### Foundational Textbooks
- **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
  - Comprehensive introduction to RL fundamentals
  - Covers policy gradients, value functions, and temporal difference learning

- **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press.
  - Chapter 20 covers deep reinforcement learning
  - Essential neural network foundations

### Deep Q-Networks and Extensions
- **Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D.** (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.
  - Original DQN paper, foundation for DIEM approach

- **Van Hasselt, H., Guez, A., & Silver, D.** (2016). "Deep reinforcement learning with double Q-learning." *Proceedings of the AAAI Conference on Artificial Intelligence*, 30(1), 2094-2100.
  - Double DQN addresses overestimation bias

- **Schaul, T., Quan, J., Antonoglou, I., & Silver, D.** (2015). "Prioritized experience replay." *arXiv preprint arXiv:1511.05952*.
  - Key component used in DIEM framework

### Policy Gradient Methods
- **Williams, R. J.** (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." *Machine Learning*, 8(3-4), 229-256.
  - REINFORCE algorithm foundations

- **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.** (2017). "Proximal policy optimization algorithms." *arXiv preprint arXiv:1707.06347*.
  - Modern policy gradient method

### RL for Combinatorial Optimization
- **Bello, I., Pham, H., Le, Q. V., Norouzi, M., & Bengio, S.** (2016). "Neural combinatorial optimization with reinforcement learning." *arXiv preprint arXiv:1611.09940*.
  - Applying RL to combinatorial problems

- **Kool, W., Van Hoof, H., & Welling, M.** (2018). "Attention, learn to solve routing problems!" *arXiv preprint arXiv:1803.08475*.
  - Attention mechanisms in combinatorial RL

---

## 3. Graph Neural Networks

### Foundational Survey Papers
- **Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Philip, S. Y.** (2020). "A comprehensive survey on graph neural networks." *IEEE Transactions on Neural Networks and Learning Systems*, 32(1), 4-24.
  - Comprehensive overview of GNN architectures

- **Zhou, J., Cui, G., Hu, S., Zhang, Z., Yang, C., Liu, Z., ... & Sun, M.** (2020). "Graph neural networks: A review of methods and applications." *AI Open*, 1, 57-81.

### Graph Convolutional Networks
- **Kipf, T. N., & Welling, M.** (2016). "Semi-supervised classification with graph convolutional networks." *arXiv preprint arXiv:1609.02907*.
  - Foundational GCN paper

- **Hamilton, W., Ying, Z., & Leskovec, J.** (2017). "Inductive representation learning on large graphs." *Advances in Neural Information Processing Systems*, 30, 1024-1034.
  - GraphSAGE for inductive learning

### Graph Attention Networks
- **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y.** (2017). "Graph attention networks." *arXiv preprint arXiv:1710.10903*.
  - Attention mechanisms for graphs

### Graph Embeddings and Node2Vec Family
- **Grover, A., & Leskovec, J.** (2016). "node2vec: Scalable feature learning for networks." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 855-864.
  - Random walk-based embeddings

- **Perozzi, B., Al-Rfou, R., & Skiena, S.** (2014). "Deepwalk: Online learning of social representations." *Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 701-710.

### Graph Transformers
- **Dwivedi, V. P., & Bresson, X.** (2020). "A generalization of transformer networks to graphs." *arXiv preprint arXiv:2012.09699*.
  - Transformer architectures for graphs

---

## 4. Influence Maximization

### Theoretical Foundations
- **Mossel, E., & Roch, S.** (2010). "On the submodularity of influence in social networks." *Proceedings of the 39th Annual ACM Symposium on Theory of Computing*, 128-134.
  - Theoretical analysis of influence functions

- **Seeman, L., & Singer, Y.** (2013). "Adaptive seeding in social networks." *Proceedings of the 2013 IEEE 54th Annual Symposium on Foundations of Computer Science*, 459-468.

### Approximation Algorithms
- **Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L.** (1978). "An analysis of approximations for maximizing submodular set functions—I." *Mathematical Programming*, 14(1), 265-294.
  - Foundational greedy approximation results

- **Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C., VanBriesen, J., & Glance, N.** (2007). "Cost-effective outbreak detection in networks." *Proceedings of the 13th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 420-429.
  - CELF optimization

### Scalable Algorithms
- **Borgs, C., Brautbar, M., Chayes, J., & Lucier, B.** (2014). "Maximizing social influence in nearly optimal time." *Proceedings of the 25th Annual ACM-SIAM Symposium on Discrete Algorithms*, 946-957.
  - Reverse reachable sets approach

- **Tang, Y., Shi, Y., & Xiao, X.** (2015). "Influence maximization in near-linear time: a martingale approach." *Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data*, 1539-1554.
  - IMM algorithm for large-scale networks

### Learning-Based Approaches
- **Ling, C., Jiang, J., Wang, J., Thai, M. T., Xue, R., Song, J., ... & Zhao, P.** (2023). "Deep graph representation learning and optimization for influence maximization." *Proceedings of the 40th International Conference on Machine Learning*, 21350-21361.

- **Chen, X., Li, G., Wang, Y., & Zhang, W.** (2022). "Learning to maximize influence with neural networks." *IEEE Transactions on Knowledge and Data Engineering*, 35(8), 8435-8447.

---

## 5. Topic Modeling and Social Networks

### Topic Modeling Fundamentals
- **Blei, D. M., Ng, A. Y., & Jordan, M. I.** (2003). "Latent dirichlet allocation." *Journal of Machine Learning Research*, 3(Jan), 993-1022.
  - Foundational LDA paper

- **Hoffman, M., Bach, F. R., & Blei, D. M.** (2010). "Online learning for latent dirichlet allocation." *Advances in Neural Information Processing Systems*, 23, 856-864.

### Social Network Topic Analysis
- **Ramage, D., Dumais, S. T., & Liebling, D. J.** (2010). "Characterizing microblogs with topic models." *Proceedings of the Fourth International AAAI Conference on Weblogs and Social Media*, 4(1), 130-137.

- **Hong, L., & Davison, B. D.** (2010). "Empirical study of topic modeling in twitter." *Proceedings of the First Workshop on Social Media Analytics*, 80-88.

### Multi-Modal Topic Models
- **Wang, C., Blei, D., & Heckerman, D.** (2008). "Continuous time dynamic topic models." *Proceedings of the Twenty-Fourth Conference on Uncertainty in Artificial Intelligence*, 579-586.

- **Blei, D. M., & Lafferty, J. D.** (2006). "Dynamic topic models." *Proceedings of the 23rd International Conference on Machine Learning*, 113-120.

### Topic-Aware Social Influence
- **Weng, J., Lim, E. P., Jiang, J., & He, Q.** (2010). "Twitterrank: finding topic-sensitive influential twitterers." *Proceedings of the Third ACM International Conference on Web Search and Data Mining*, 261-270.

- **Tang, J., Sun, J., Wang, C., & Yang, Z.** (2009). "Social influence analysis in large-scale networks." *Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 807-816.

---

## 6. Combinatorial Optimization

### Classical Optimization Theory
- **Cook, W. J., Cunningham, W. H., Pulleyblank, W. R., & Schrijver, A.** (1998). *Combinatorial Optimization*. John Wiley & Sons.
  - Comprehensive treatment of combinatorial optimization

- **Korte, B., & Vygen, J.** (2011). *Combinatorial Optimization: Theory and Algorithms* (5th ed.). Springer.

### Submodular Optimization
- **Bach, F.** (2013). "Learning with submodular functions: A convex optimization perspective." *Foundations and Trends in Machine Learning*, 6(2-3), 145-373.

- **Krause, A., & Golovin, D.** (2014). "Submodular function maximization." *Tractability: Practical Approaches to Hard Problems*, 3, 71-104.

### Machine Learning for Combinatorial Optimization
- **Bengio, Y., Lodi, A., & Prouvost, A.** (2021). "Machine learning for combinatorial optimization: a methodological tour d'horizon." *European Journal of Operational Research*, 290(2), 405-421.

- **Cappart, Q., Chételat, D., Khalil, E., Lodi, A., Morris, C., & Veličković, P.** (2021). "Combinatorial optimization and reasoning with graph neural networks." *arXiv preprint arXiv:2102.09544*.

---

## 7. Advanced Topics and Extensions

### Multi-Objective Optimization
- **Deb, K.** (2001). *Multi-Objective Optimization Using Evolutionary Algorithms*. John Wiley & Sons.

- **Miettinen, K.** (2012). *Nonlinear Multiobjective Optimization*. Springer Science & Business Media.

### Fairness in Machine Learning
- **Barocas, S., Hardt, M., & Narayanan, A.** (2019). *Fairness and Machine Learning*. fairmlbook.org.

- **Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A.** (2021). "A survey on bias and fairness in machine learning." *ACM Computing Surveys*, 54(6), 1-35.

### Privacy-Preserving Machine Learning
- **Dwork, C., & Roth, A.** (2014). "The algorithmic foundations of differential privacy." *Foundations and Trends in Theoretical Computer Science*, 9(3–4), 211-407.

- **Li, T., Sahu, A. K., Talwalkar, A., & Smith, V.** (2020). "Federated learning: Challenges, methods, and future directions." *IEEE Signal Processing Magazine*, 37(3), 50-60.

### Temporal Networks
- **Holme, P., & Saramäki, J.** (2012). "Temporal networks." *Physics Reports*, 519(3), 97-125.

- **Rossetti, G., & Cazabet, R.** (2018). "Community discovery in dynamic networks: A survey." *ACM Computing Surveys*, 51(2), 1-37.

### Meta-Learning
- **Finn, C., Abbeel, P., & Levine, S.** (2017). "Model-agnostic meta-learning for fast adaptation of deep networks." *Proceedings of the 34th International Conference on Machine Learning*, 1126-1135.

- **Hospedales, T., Antoniou, A., Micaelli, P., & Storkey, A.** (2021). "Meta-learning in neural networks: A survey." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(9), 5149-5169.

---

## 8. Implementation Resources

### Deep Learning Frameworks
#### PyTorch
- **Official Documentation**: [pytorch.org](https://pytorch.org/docs/stable/index.html)
- **PyTorch Geometric**: [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io/)
  - Graph neural network library used in implementation

#### TensorFlow/Keras
- **TensorFlow Documentation**: [tensorflow.org](https://www.tensorflow.org/)
- **Spektral**: [graphneural.network](https://graphneural.network/)
  - Graph neural networks in TensorFlow

### Reinforcement Learning Libraries
#### Stable-Baselines3
- **Documentation**: [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io/)
- **GitHub**: [github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

#### Ray RLlib
- **Documentation**: [docs.ray.io/en/latest/rllib/](https://docs.ray.io/en/latest/rllib/)
- **Scalable reinforcement learning**

### Graph Processing Libraries
#### NetworkX
```python
# Essential for graph manipulation in DIEM implementation
import networkx as nx

# Example usage patterns from our implementation
G = nx.erdos_renyi_graph(n=1000, p=0.01)
centrality = nx.betweenness_centrality(G)
communities = nx.community.greedy_modularity_communities(G)
```

#### DGL (Deep Graph Library)
- **Documentation**: [dgl.ai](https://www.dgl.ai/)
- **Alternative to PyTorch Geometric**

### Optimization Libraries
#### CVXPY
- **Documentation**: [cvxpy.org](https://www.cvxpy.org/)
- **Convex optimization problems**

#### OR-Tools
- **Documentation**: [developers.google.com/optimization](https://developers.google.com/optimization)
- **Google's optimization tools**

---

## 9. Datasets and Benchmarks

### Social Network Datasets
#### Stanford Network Analysis Project (SNAP)
- **URL**: [snap.stanford.edu/data/](https://snap.stanford.edu/data/)
- **Key datasets used in IM research**:
  - Facebook social circles
  - Twitter social networks  
  - Citation networks
  - Collaboration networks

#### Network Repository
- **URL**: [networkrepository.com](http://networkrepository.com/)
- **Large collection of graph datasets**

### Specific Datasets for IM Evaluation
```python
# Commonly used datasets in influence maximization research
datasets = {
    'ca-GrQc': {
        'nodes': 5242,
        'edges': 14496,
        'type': 'collaboration_network',
        'source': 'arXiv general relativity'
    },
    'ca-HepTh': {
        'nodes': 9877,
        'edges': 25998,
        'type': 'collaboration_network', 
        'source': 'arXiv high energy physics'
    },
    'p2p-Gnutella08': {
        'nodes': 6301,
        'edges': 20777,
        'type': 'peer_to_peer',
        'source': 'Gnutella network'
    },
    'email-Enron': {
        'nodes': 36692,
        'edges': 183831,
        'type': 'communication',
        'source': 'Enron email network'
    }
}
```

### Topic Modeling Datasets
#### Reuters-21578
- **Classic text classification dataset**
- **Multiple topic categories**

#### 20 Newsgroups
- **Discussion forums across topics**
- **Good for topic modeling evaluation**

### Synthetic Graph Generators
```python
# Network models for controlled experiments
import networkx as nx
import numpy as np

def generate_test_networks():
    """Generate various network topologies for testing"""
    networks = {}
    
    # Scale-free network (preferential attachment)
    networks['scale_free'] = nx.barabasi_albert_graph(1000, 3)
    
    # Small-world network
    networks['small_world'] = nx.watts_strogatz_graph(1000, 6, 0.1)
    
    # Random network
    networks['random'] = nx.erdos_renyi_graph(1000, 0.01)
    
    # Community structure
    networks['community'] = nx.planted_partition_graph(4, 250, 0.1, 0.01)
    
    return networks
```

---

## 10. Software Tools and Libraries

### Development Environment Setup
```bash
# Complete environment setup for DIEM implementation
conda create -n diem-env python=3.9
conda activate diem-env

# Core dependencies
pip install torch torchvision torchaudio
pip install torch-geometric
pip install networkx
pip install numpy scipy matplotlib
pip install pandas scikit-learn
pip install tqdm

# Optional but recommended
pip install wandb  # Experiment tracking
pip install tensorboard  # Visualization
pip install jupyter  # Development
pip install pytest  # Testing
```

### Visualization Tools
#### Matplotlib and Seaborn
```python
# Plotting influence spread results
import matplotlib.pyplot as plt
import seaborn as sns

def plot_influence_comparison(results_dict):
    """Visualize algorithm performance comparison"""
    algorithms = list(results_dict.keys())
    spreads = [results_dict[alg]['spread'] for alg in algorithms]
    times = [results_dict[alg]['time'] for alg in algorithms]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Influence spread comparison
    ax1.bar(algorithms, spreads)
    ax1.set_ylabel('Influence Spread')
    ax1.set_title('Algorithm Performance Comparison')
    
    # Runtime comparison  
    ax2.bar(algorithms, times)
    ax2.set_ylabel('Runtime (seconds)')
    ax2.set_title('Algorithm Efficiency Comparison')
    
    plt.tight_layout()
    return fig
```

#### Graph Visualization
```python
# Network visualization for analysis
import networkx as nx
import matplotlib.pyplot as plt

def visualize_influence_spread(G, seed_set, influenced_nodes):
    """Visualize influence propagation in network"""
    pos = nx.spring_layout(G)
    
    # Color nodes by status
    node_colors = []
    for node in G.nodes():
        if node in seed_set:
            node_colors.append('red')  # Seed nodes
        elif node in influenced_nodes:
            node_colors.append('orange')  # Influenced
        else:
            node_colors.append('lightblue')  # Uninfluenced
    
    nx.draw(G, pos, node_color=node_colors, 
            with_labels=False, node_size=50)
    
    # Add legend
    red_patch = plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor='red', markersize=10, label='Seed')
    orange_patch = plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor='orange', markersize=10, label='Influenced')
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor='lightblue', markersize=10, label='Uninfluenced')
    
    plt.legend(handles=[red_patch, orange_patch, blue_patch])
    plt.title('Influence Spread Visualization')
    return plt
```

### Experiment Tracking
#### Weights & Biases (wandb)
```python
# Track experiments for reproducible research
import wandb

def setup_experiment_tracking(config):
    """Initialize experiment tracking"""
    wandb.init(
        project="topic-aware-influence-maximization",
        config=config,
        tags=["deep-rl", "graph-neural-networks", "influence-maximization"]
    )
    
    # Log model architecture
    wandb.watch(model, log="all")
    
    return wandb

def log_training_metrics(epoch, loss, reward, influence_spread):
    """Log training progress"""
    wandb.log({
        "epoch": epoch,
        "loss": loss, 
        "reward": reward,
        "influence_spread": influence_spread
    })
```

### Performance Profiling
#### cProfile and line_profiler
```python
# Profile code performance
import cProfile
import pstats
from line_profiler import LineProfiler

def profile_algorithm(algorithm_func, *args, **kwargs):
    """Profile algorithm execution"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = algorithm_func(*args, **kwargs)
    
    profiler.disable()
    
    # Print top time-consuming functions
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    return result
```

### Testing Framework
```python
# Comprehensive testing setup
import pytest
import numpy as np
import torch

class TestDIEMFramework:
    """Test suite for DIEM implementation"""
    
    @pytest.fixture
    def sample_network(self):
        """Create sample network for testing"""
        G = nx.erdos_renyi_graph(100, 0.1)
        return G
    
    @pytest.fixture  
    def sample_topics(self):
        """Create sample topic data"""
        num_users = 100
        num_topics = 5
        topic_matrix = np.random.rand(num_users, num_topics)
        return topic_matrix
    
    def test_social_network_initialization(self, sample_network, sample_topics):
        """Test SocialNetwork class initialization"""
        social_net = SocialNetwork(sample_network, sample_topics)
        assert social_net.graph.number_of_nodes() == 100
        assert social_net.topic_matrix.shape == (100, 5)
    
    def test_diffusion2vec_forward_pass(self, sample_network, sample_topics):
        """Test Diffusion2Vec embedding generation"""
        embedding_model = Diffusion2Vec(num_topics=5, embedding_dim=64)
        node_features = torch.randn(100, 5)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        
        embeddings = embedding_model(node_features, edge_index)
        assert embeddings.shape == (100, 64)
    
    def test_diem_influence_estimation(self):
        """Test DIEM influence estimation"""
        diem_model = DIEM(embedding_dim=64, hidden_dim=128)
        seed_embeddings = torch.randn(10, 64)
        target_embedding = torch.randn(64)
        
        influence_score = diem_model(seed_embeddings, target_embedding)
        assert isinstance(influence_score, torch.Tensor)
        assert influence_score.shape == torch.Size([])

# Run tests
if __name__ == "__main__":
    pytest.main([__file__])
```

---

## Conclusion

This comprehensive reference collection provides the foundation for understanding and extending the DIEM framework for topic-aware influence maximization. The resources are organized to support both theoretical understanding and practical implementation, covering:

- **Theoretical Foundations**: Core papers in RL, GNNs, and influence maximization
- **Implementation Resources**: Code libraries, frameworks, and tools
- **Datasets and Benchmarks**: Standard evaluation resources
- **Advanced Topics**: Cutting-edge research directions

For researchers and practitioners working in this area, these references provide pathways for deeper exploration of specific aspects of the problem, from fundamental algorithms to advanced applications in fairness, privacy, and scalability.

### Recommended Reading Path

1. **Beginners**: Start with Sutton & Barto (RL), Kempe et al. (IM), and Wu et al. (GNN survey)
2. **Intermediate**: Focus on recent learning-based IM approaches and graph embedding methods  
3. **Advanced**: Explore meta-learning, fairness, and privacy-preserving extensions
4. **Practitioners**: Emphasize implementation resources, datasets, and evaluation frameworks

The field of topic-aware influence maximization continues to evolve rapidly, with new publications emerging regularly in conferences like ICML, NeurIPS, KDD, and WWW. Staying current with these venues will provide access to the latest developments in this exciting research area.