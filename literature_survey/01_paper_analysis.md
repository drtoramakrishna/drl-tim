# Literature Survey: Deep Reinforcement Learning-Based Approach to Tackle Topic-Aware Influence Maximization

## Executive Summary

This paper presents a novel data-driven framework that combines deep reinforcement learning with graph embedding techniques to solve the Topic-Aware Influence Maximization (TIM) problem. The work addresses critical limitations in existing approaches by proposing a generalized heuristic framework that avoids complex algorithmic design and improves computational efficiency.

## 1. Problem Context and Motivation

### 1.1 Social Network Influence Maximization
The traditional Influence Maximization (IM) problem, introduced by Kempe et al. [15], focuses on selecting a set of k seed users in a social network to maximize the spread of information. However, this generic approach lacks topic awareness, making it unsuitable for personalized advertising and viral marketing scenarios.

### 1.2 Topic-Aware Influence Maximization (TIM)
TIM extends classical IM by incorporating topic relevance, addressing two main branches:
1. **IM for topic-relevant targets**: Maximizing influence on users interested in specific topics
2. **IM for topic-dependent diffusion**: Using topic-aware propagation models where edge probabilities vary by topic

## 2. Key Contributions

### 2.1 Novel Approach Integration
- **Unified Framework**: Combines both branches of TIM into a single comprehensive approach
- **Data-Driven Method**: Replaces manual algorithm design with automated learning
- **Graph Embedding**: Introduces Diffusion2Vec for feature extraction
- **Deep Learning**: Employs Deep Influence Evaluation Model (DIEM) for influence estimation

### 2.2 Technical Innovations

#### 2.2.1 Topic-Aware Propagation Models
- Extended Independent Cascade (IC) and Linear Threshold (LT) models
- Incorporates user similarity, benefits, and contact frequency
- Data-driven probability assignment rather than random values

#### 2.2.2 Deep Influence Evaluation Model (DIEM)
- Neural network-based influence estimation
- Considers graph structure, user attributes, and query topics
- Enables rapid solution construction without heavy online computation

#### 2.2.3 Reinforcement Learning Framework
- Uses Double Deep Q-Networks (DDQN) with prioritized experience replay
- Addresses overestimation and delayed reward challenges
- Enables end-to-end parameter learning

## 3. Algorithmic Framework

### 3.1 Greedy Solution Construction
The framework constructs solutions by sequentially selecting nodes based on an evaluation function Q̂:

```
1. Initialize empty solution set S = ∅
2. For each step until |S| = k:
   - Evaluate all candidates using Q̂((v, S)|τ)
   - Select v* = argmax Q̂((v, S)|τ)
   - Update S := (S, v*)
3. Return final seed set S
```

### 3.2 Core Components

#### Graph Representation (Diffusion2Vec)
- Embedding network that captures node features and graph structure
- Iterative update mechanism for T-hop neighborhood information
- Incorporates user profiles and selection status

#### Deep Influence Evaluation
- Neural network Q̂ that estimates marginal influence
- Takes node embeddings, partial solutions, and query topics as input
- Enables efficient greedy selection without exact influence computation

## 4. Experimental Validation

### 4.1 Datasets and Setup
- **Generated Graphs**: Barabási-Albert networks with 1M-10M nodes
- **Real-World Data**: Twitter dataset with 41.6M users, 476M tweets
- **Topics**: 50 extracted topics using LDA modeling
- **Baselines**: WRIS, RR, and IRR methods

### 4.2 Performance Results

#### Efficiency Gains
- **Speed**: 160x faster than WRIS, comparable to IRR
- **Storage**: 100x less disk space than IRR
- **Scalability**: Generalizes to larger graphs without retraining

#### Effectiveness
- Comparable solution quality to state-of-the-art methods
- Maintains performance across different graph sizes
- Consistent results across varying topic numbers

## 5. Methodological Strengths

### 5.1 Generalization Capability
- Single trained model works across different graph instances
- No need for retraining when graph data changes
- Scalable to larger networks than training instances

### 5.2 Computational Efficiency
- Offline training with online deployment
- Minimal storage requirements (only model parameters)
- Fast query response times suitable for real-time applications

### 5.3 Comprehensive Topic Modeling
- Considers both user interests and dynamic probabilities
- Integrates similarity, benefit, and contact frequency metrics
- Unified approach to both TIM branches

## 6. Technical Architecture

```mermaid
graph TB
    A[Social Network G] --> B[User Profiles & Topics τ]
    A --> C[Topic-Aware Models]
    C --> D[IC Model with p_uv]
    C --> E[LT Model with θ_v]
    B --> F[Diffusion2Vec Embedding]
    F --> G[Node Embeddings u_v^(T)]
    G --> H[DIEM Evaluation Q̂]
    H --> I[Greedy Selection]
    I --> J[Seed Set S*]
    K[RL Training] --> H
    L[Experience Replay] --> K
```

## 7. Research Significance

### 7.1 Theoretical Impact
- First to combine deep RL with graph embedding for TIM
- Unified treatment of topic-aware propagation models
- Novel approach to combinatorial optimization on graphs

### 7.2 Practical Applications
- Real-time advertising and viral marketing
- Social media influence campaigns
- Product recommendation systems
- Opinion propagation analysis

## 8. Limitations and Future Directions

### 8.1 Current Limitations
- Limited to specific diffusion models (IC/LT)
- Requires sufficient training data for generalization
- Topic space changes require model retraining
- Hyperparameter sensitivity not fully explored

### 8.2 Potential Extensions
- Multi-objective influence maximization
- Dynamic network structures
- Adversarial influence scenarios
- Integration with other propagation models
- Real-time adaptation mechanisms

## 9. Related Work Context

### 9.1 Classical Influence Maximization
- Kempe et al. [15]: Original IM formulation
- CELF [19], CELF++ [12]: Efficiency improvements
- RIS [6]: Scalable random sampling approach

### 9.2 Topic-Aware Extensions
- Inflex [2]: Precomputed seed sets approach
- WRIS [23]: Weighted sampling technique
- Various topic-dependent diffusion models [9, 11]

### 9.3 Learning-Based Approaches
- Structure2Vec [16]: Graph embedding for combinatorial problems
- Neural combinatorial optimization [3, 32]
- Deep RL for graph problems

## 10. Implementation Considerations

### 10.1 Training Requirements
- Computational resources: GPU with sufficient memory
- Training time: 3-72 hours depending on topic complexity
- Data requirements: Representative graph instances

### 10.2 Deployment Aspects
- Model size: Compact parameter storage
- Response time: Sub-second query processing
- Scalability: Linear scaling with graph size

This comprehensive analysis provides the foundation for understanding the paper's contributions and implementing the proposed methodology.