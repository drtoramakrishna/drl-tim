# System Architecture and Algorithm Flow Diagrams

## Table of Contents
1. [Overall System Architecture](#1-overall-system-architecture)
2. [Topic-Aware Propagation Models](#2-topic-aware-propagation-models)
3. [Diffusion2Vec Architecture](#3-diffusion2vec-architecture)
4. [DIEM Network Structure](#4-diem-network-structure)
5. [Reinforcement Learning Workflow](#5-reinforcement-learning-workflow)
6. [Training Process Flow](#6-training-process-flow)
7. [Evaluation and Comparison Framework](#7-evaluation-and-comparison-framework)

---

## 1. Overall System Architecture

### High-Level System Overview

```mermaid
graph TB
    subgraph "Input Layer"
        A[Social Network G(V,E)]
        B[User Profiles {P_v}]
        C[Query Topics τ]
        D[Budget k]
    end
    
    subgraph "Topic-Aware Models"
        E[Contact Frequency w_uv]
        F[User Similarity sim(u,v)]
        G[User Benefits B_v^τ]
        H[Topic-Aware IC Model]
        I[Topic-Aware LT Model]
    end
    
    subgraph "Deep Learning Framework"
        J[Diffusion2Vec Embedding]
        K[Node Embeddings u_v^T]
        L[DIEM Evaluation Q̂]
    end
    
    subgraph "RL Training"
        M[Experience Buffer D]
        N[Double DQN]
        O[Prioritized Replay]
        P[Parameter Updates]
    end
    
    subgraph "Output Layer"
        Q[Greedy Selection]
        R[Seed Set S*]
        S[Influence Spread σ_G(S*|τ)]
    end
    
    A --> E
    B --> F
    B --> G
    C --> G
    
    E --> H
    F --> H
    G --> H
    
    E --> I
    F --> I
    G --> I
    
    A --> J
    B --> J
    H --> J
    I --> J
    
    J --> K
    K --> L
    
    L --> M
    M --> N
    N --> O
    O --> P
    P --> L
    
    L --> Q
    Q --> R
    R --> S
    
    style A fill:#ffcccc
    style B fill:#ffcccc
    style C fill:#ffcccc
    style J fill:#ccffcc
    style L fill:#ccffcc
    style N fill:#ccccff
    style R fill:#ffffcc
```

### Component Interaction Flow

```mermaid
sequenceDiagram
    participant U as User Query
    participant TM as Topic-Aware Models
    participant D2V as Diffusion2Vec
    participant DIEM as DIEM Network
    participant RL as RL Agent
    participant GS as Greedy Selector
    
    U->>TM: Social Network + Topics + Profiles
    TM->>TM: Calculate p_uv^(τ), θ_v^(τ), b_uv^(τ)
    TM->>D2V: Topic-aware probabilities
    D2V->>D2V: Iterative embedding updates (T times)
    D2V->>DIEM: Node embeddings u_v^(T)
    
    loop For k seed selection steps
        DIEM->>RL: Marginal influence estimates Q̂(S,v|τ)
        RL->>GS: Action selection (ε-greedy)
        GS->>GS: Select best candidate v*
        GS->>RL: Update state S ← S ∪ {v*}
        RL->>RL: Store experience (S,v*,r,S')
        RL->>DIEM: Update parameters Θ
    end
    
    GS->>U: Return seed set S*
```

---

## 2. Topic-Aware Propagation Models

### Independent Cascade (IC) Model Extension

```mermaid
graph TB
    subgraph "Classical IC Model"
        A1[Edge Probabilities p_uv]
        B1[Random Activation]
        C1[Cascading Spread]
    end
    
    subgraph "Topic-Aware Extension"
        A2[Contact Frequency α₁w_uv]
        B2[User Similarity α₂sim(u,v)]
        C2[Target Benefits α₃B_v^τ]
        D2[Combined Probability]
    end
    
    subgraph "Enhanced IC Process"
        E2[Topic-Specific Activation]
        F2[Benefit-Weighted Influence]
        G2[Targeted Spread σ_G(S|τ)]
    end
    
    A2 --> D2
    B2 --> D2
    C2 --> D2
    
    D2 --> E2
    E2 --> F2
    F2 --> G2
    
    A1 -.-> A2
    B1 -.-> E2
    C1 -.-> F2
    
    style D2 fill:#ffcccc
    style F2 fill:#ccffcc
```

### Linear Threshold (LT) Model Extension

```mermaid
graph TB
    subgraph "Classical LT Model"
        A1[Edge Weights b_uv]
        B1[Node Thresholds θ_v]
        C1[Threshold Activation]
    end
    
    subgraph "Topic-Aware Extension"
        A2["Edge Weights: b_uv^(τ) = α₁w_uv + α₂sim(u,v)"]
        B2["Thresholds: θ_v^(τ) = 1 - α₃B_v^(τ)"]
        C2[Topic-Dependent Activation]
    end
    
    subgraph "Activation Logic"
        D2[Sum Active Neighbors]
        E2[Compare vs Threshold]
        F2[Activate if Sum ≥ θ_v^τ]
    end
    
    A1 -.-> A2
    B1 -.-> B2
    C1 -.-> C2
    
    A2 --> D2
    B2 --> E2
    D2 --> E2
    E2 --> F2
    
    style A2 fill:#ffcccc
    style B2 fill:#ccffcc
    style F2 fill:#ffffcc
```

### Three-Metric Integration

```mermaid
graph LR
    subgraph "User Profiles"
        A[User u: P_u = [p₁, p₂, ..., pₜ]]
        B[User v: P_v = [p₁, p₂, ..., pₜ]]
    end
    
    subgraph "Metric Calculations"
        C["Contact Frequency: w_uv ∈ (0,1]"]
        D["Similarity: sim(u,v) = (P_u·P_v)/(||P_u||||P_v||)"]
        E["Benefit: B_v^τ = Σ_{t∈τ} P_v[t]"]
    end
    
    subgraph "Model Integration"
        F["IC: p_uv^τ = (α₁w_uv + α₂sim(u,v) + α₃B_v^τ)/3"]
        G["LT: b_uv^τ = α₁w_uv + α₂sim(u,v)"]
        H["LT: θ_v^τ = 1 - α₃B_v^τ"]
    end
    
    A --> D
    B --> D
    A --> E
    B --> E
    
    C --> F
    D --> F
    E --> F
    
    C --> G
    D --> G
    
    E --> H
    
    style D fill:#ffcccc
    style E fill:#ccffcc
    style F fill:#ffffcc
```

---

## 3. Diffusion2Vec Architecture

### Iterative Embedding Process

```mermaid
graph TB
    subgraph "Input Features"
        A["X_v = [selection_flag, P_v]"]
        B["Adjacency Matrix A"]
        C["Edge Weights p_uv^(τ)"]
    end
    
    subgraph "Iteration t"
        D["u_v^(t) (Current Embeddings)"]
    end
    
    subgraph "Neural Network Layers"
        E["Θ₁: Feature Transform"]
        F["Θ₂: Neighbor Aggregation"]
        G["Θ₃: Edge Weight Integration"]
        H["Θ₄: Weight Embedding"]
    end
    
    subgraph "Update Computation"
        I["Feature Term: Θ₁X_v"]
        J["Neighbor Term: Θ₂Σ_{u∈N(v)}u_u^(t)"]
        K["Edge Term: Θ₃Σ_{u∈N(v)}relu(Θ₄p_uv^(τ))"]
        L["u_v^(t+1) = relu(I + J + K)"]
    end
    
    subgraph "Output"
        M["Final Embeddings u_v^(T)"]
    end
    
    A --> E
    A --> I
    B --> F
    D --> F
    D --> J
    C --> H
    H --> K
    
    I --> L
    J --> L
    K --> L
    
    L -.-> D
    L --> M
    
    style E fill:#ffcccc
    style F fill:#ccffcc
    style G fill:#ffffcc
    style L fill:#ccccff
```

### Multi-Hop Information Propagation

```mermaid
graph LR
    subgraph "Iteration 0"
        A0["u_v^(0) = 0"]
    end
    
    subgraph "Iteration 1"
        A1["u_v^(1)"]
        B1["1-hop info"]
    end
    
    subgraph "Iteration 2"
        A2["u_v^(2)"]
        B2["2-hop info"]
    end
    
    subgraph "Iteration T"
        AT["u_v^(T)"]
        BT["T-hop info"]
    end
    
    subgraph "Information Scope"
        C1["Direct neighbors"]
        C2["Neighbors of neighbors"]
        CT["T-hop neighborhood"]
    end
    
    A0 --> A1
    A1 --> A2
    A2 -.-> AT
    
    A1 --> B1
    A2 --> B2
    AT --> BT
    
    B1 --> C1
    B2 --> C2
    BT --> CT
    
    style A0 fill:#ffcccc
    style AT fill:#ccffcc
    style BT fill:#ffffcc
```

### Feature Aggregation Mechanism

```mermaid
graph TB
    subgraph "Node v Features"
        A["Selection Status: X_v[0]"]
        B["Topic Interests: X_v[1:]"]
    end
    
    subgraph "Neighborhood Information"
        C["Neighbor Embeddings: {u_u^(t)}_{u∈N(v)}"]
        D["Edge Probabilities: {p_uv^(τ)}_{u∈N(v)}"]
    end
    
    subgraph "Aggregation Functions"
        E["Sum Aggregation: Σ_{u∈N(v)} u_u^(t)"]
        F["Weighted Aggregation: Σ_{u∈N(v)} f(p_uv^(τ))"]
    end
    
    subgraph "Neural Transformations"
        G["Linear Transform + ReLU"]
        H["Multi-layer Processing"]
    end
    
    subgraph "Updated Embedding"
        I["u_v^(t+1)"]
    end
    
    A --> G
    B --> G
    C --> E
    D --> F
    
    E --> H
    F --> H
    G --> H
    
    H --> I
    
    style E fill:#ffcccc
    style F fill:#ccffcc
    style I fill:#ffffcc
```

---

## 4. DIEM Network Structure

### Deep Influence Evaluation Architecture

```mermaid
graph TB
    subgraph "Input Components"
        A["Seed Set Embeddings: {u_u^(T)}_{u∈S}"]
        B["Candidate Embedding: u_v^(T)"]
    end
    
    subgraph "Aggregation Layer"
        C["Seed Aggregation: Σ_{u∈S} u_u^(T)"]
        D["Candidate Processing: u_v^(T)"]
    end
    
    subgraph "Transformation Layers"
        E["Θ₆: Seed Transform"]
        F["Θ₇: Candidate Transform"]
    end
    
    subgraph "Combination Layer"
        G["Concatenation: [Θ₆(...), Θ₇(...)]"]
        H["ReLU Activation"]
    end
    
    subgraph "Output Layer"
        I["Θ₅: Linear Output"]
        J["Marginal Influence Q̂(S,v|τ)"]
    end
    
    A --> C
    B --> D
    
    C --> E
    D --> F
    
    E --> G
    F --> G
    
    G --> H
    H --> I
    I --> J
    
    style C fill:#ffcccc
    style G fill:#ccffcc
    style J fill:#ffffcc
```

### DIEM Forward Pass Flow

```mermaid
sequenceDiagram
    participant S as Seed Set S
    participant C as Candidate v
    participant E as Embedding Layer
    participant A as Aggregation
    participant T as Transform
    participant O as Output
    
    S->>E: Get embeddings {u_u^(T)}
    C->>E: Get embedding u_v^(T)
    
    E->>A: Seed embeddings
    A->>A: Sum aggregation
    
    A->>T: Aggregated seed representation
    E->>T: Candidate representation
    
    T->>T: Apply Θ₆ and Θ₇
    T->>O: Concatenated features
    
    O->>O: ReLU + Θ₅
    O-->>S: Return Q̂(S,v|τ)
```

### Influence Estimation Concept

```mermaid
graph LR
    subgraph "Current State"
        A["Partial Solution S"]
        B["Current Influence σ(S|τ)"]
    end
    
    subgraph "Candidate Addition"
        C["Candidate Node v"]
        D["Extended Solution S∪{v}"]
    end
    
    subgraph "DIEM Estimation"
        E["Deep Network Q̂"]
        F["Estimated Marginal Gain"]
    end
    
    subgraph "True Computation"
        G["Monte Carlo Simulation"]
        H["True Marginal Gain"]
    end
    
    A --> D
    C --> D
    A --> E
    C --> E
    E --> F
    
    D --> G
    G --> H
    
    F -.-> H
    
    style E fill:#ccffcc
    style F fill:#ffffcc
    style H fill:#ffcccc
```

---

## 5. Reinforcement Learning Workflow

### MDP Formulation for TIM

```mermaid
graph TB
    subgraph "MDP Components"
        A["States S: Partial seed sets"]
        B["Actions A: Add node v to S"]
        C["Rewards R: Marginal influence"]
        D["Policy π: Node selection strategy"]
    end
    
    subgraph "State Transitions"
        E["S₀ = ∅ (Empty set)"]
        F["S₁ = {v₁} (First seed)"]
        G["S₂ = {v₁,v₂} (Second seed)"]
        H["Sₖ = {v₁,...,vₖ} (Final set)"]
    end
    
    subgraph "Q-Learning Process"
        I["Q-function Approximation"]
        J["Experience Collection"]
        K["Batch Training"]
        L["Parameter Updates"]
    end
    
    A --> E
    E --> F
    F --> G
    G -.-> H
    
    B --> I
    C --> J
    D --> K
    I --> L
    
    style A fill:#ffcccc
    style I fill:#ccffcc
    style L fill:#ffffcc
```

### Double DQN Architecture

```mermaid
graph TB
    subgraph "Online Networks"
        A["Diffusion2Vec θ"]
        B["DIEM Q̂(s,a;θ)"]
    end
    
    subgraph "Target Networks"
        C["Target Diffusion2Vec θ⁻"]
        D["Target DIEM Q̂(s,a;θ⁻)"]
    end
    
    subgraph "Action Selection"
        E["Online Network"]
        F["argmax_a Q̂(s',a;θ)"]
    end
    
    subgraph "Value Evaluation"
        G["Target Network"]
        H["Q̂(s',a*;θ⁻)"]
    end
    
    subgraph "Target Computation"
        I["y = r + γQ̂(s',a*;θ⁻)"]
        J["Loss: (y - Q̂(s,a;θ))²"]
    end
    
    A --> B
    C --> D
    
    B --> E
    E --> F
    
    D --> G
    F --> H
    
    H --> I
    B --> J
    I --> J
    
    style B fill:#ccffcc
    style F fill:#ffffcc
    style I fill:#ffcccc
```

### Prioritized Experience Replay

```mermaid
graph LR
    subgraph "Experience Buffer"
        A["Experience (s,a,r,s')"]
        B["TD Error |δ|"]
        C["Priority p = |δ| + ε"]
    end
    
    subgraph "Sampling Process"
        D["Priority Distribution"]
        E["Weighted Sampling"]
        F["Importance Weights"]
    end
    
    subgraph "Training Update"
        G["Batch Processing"]
        H["Parameter Update"]
        I["Priority Update"]
    end
    
    A --> B
    B --> C
    C --> D
    
    D --> E
    E --> F
    F --> G
    
    G --> H
    H --> I
    I --> C
    
    style C fill:#ffcccc
    style E fill:#ccffcc
    style H fill:#ffffcc
```

---

## 6. Training Process Flow

### End-to-End Training Pipeline

```mermaid
graph TB
    subgraph "Episode Generation"
        A["Initialize S₀ = ∅"]
        B["Select action aₜ"]
        C["Compute reward rₜ"]
        D["Update state Sₜ₊₁"]
        E["Store experience"]
    end
    
    subgraph "Batch Learning"
        F["Sample from buffer"]
        G["Compute targets"]
        H["Calculate loss"]
        I["Backpropagation"]
        J["Update priorities"]
    end
    
    subgraph "Network Updates"
        K["Update online networks"]
        L["Copy to target networks"]
        M["Decay exploration ε"]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> B
    
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    
    I --> K
    K --> L
    L --> M
    
    style B fill:#ffcccc
    style H fill:#ccffcc
    style L fill:#ffffcc
```

### Training Loop Dynamics

```mermaid
sequenceDiagram
    participant Env as Environment
    participant Agent as RL Agent
    participant Net as Neural Networks
    participant Buf as Replay Buffer
    participant Opt as Optimizer
    
    loop Episode
        Env->>Agent: Current state S
        Agent->>Net: Get Q-values
        Net-->>Agent: Q̂(S,v|τ) for all v
        Agent->>Agent: ε-greedy selection
        Agent->>Env: Action (selected node)
        Env->>Env: Compute true reward
        Env-->>Agent: Reward + new state
        Agent->>Buf: Store experience
    end
    
    loop Batch Training
        Buf->>Opt: Sample prioritized batch
        Opt->>Net: Compute targets (Double DQN)
        Net-->>Opt: Current Q-values
        Opt->>Opt: Compute loss + gradients
        Opt->>Net: Update parameters
        Opt->>Buf: Update priorities
    end
    
    Note over Net: Periodic target network update
```

### Learning Curve Analysis

```mermaid
graph LR
    subgraph "Training Phases"
        A["Exploration Phase"]
        B["Learning Phase"]
        C["Convergence Phase"]
    end
    
    subgraph "Performance Metrics"
        D["Episode Rewards"]
        E["TD Errors"]
        F["Influence Quality"]
    end
    
    subgraph "Optimization Dynamics"
        G["High Variance"]
        H["Decreasing Loss"]
        I["Stable Performance"]
    end
    
    A --> D
    A --> G
    B --> E
    B --> H
    C --> F
    C --> I
    
    style A fill:#ffcccc
    style B fill:#ccffcc
    style C fill:#ffffcc
```

---

## 7. Evaluation and Comparison Framework

### Benchmark Architecture

```mermaid
graph TB
    subgraph "Algorithms Under Test"
        A["DIEM (Proposed)"]
        B["WRIS Baseline"]
        C["CELF Baseline"]
        D["Greedy Baseline"]
    end
    
    subgraph "Test Scenarios"
        E["Different Networks"]
        F["Varying Budget k"]
        G["Multiple Topics"]
        H["Graph Sizes"]
    end
    
    subgraph "Evaluation Metrics"
        I["Influence Spread"]
        J["Execution Time"]
        K["Memory Usage"]
        L["Solution Quality"]
    end
    
    subgraph "Statistical Analysis"
        M["Mean Performance"]
        N["Standard Deviation"]
        O["Statistical Tests"]
        P["Confidence Intervals"]
    end
    
    A --> I
    B --> I
    C --> I
    D --> I
    
    E --> A
    F --> A
    G --> A
    H --> A
    
    I --> M
    J --> N
    K --> O
    L --> P
    
    style A fill:#ccffcc
    style I fill:#ffffcc
    style M fill:#ffcccc
```

### Performance Comparison Visualization

```mermaid
graph LR
    subgraph "Efficiency Analysis"
        A["Time Complexity"]
        B["Space Complexity"]
        C["Scalability"]
    end
    
    subgraph "Quality Analysis"
        D["Approximation Ratio"]
        E["Solution Accuracy"]
        F["Generalization"]
    end
    
    subgraph "Trade-off Analysis"
        G["Speed vs Quality"]
        H["Memory vs Performance"]
        I["Training vs Inference"]
    end
    
    A --> G
    D --> G
    B --> H
    E --> H
    C --> I
    F --> I
    
    style G fill:#ffcccc
    style H fill:#ccffcc
    style I fill:#ffffcc
```

This comprehensive diagram collection provides visual representations of all major components and processes in the DIEM framework, making it easier to understand the complex interactions between different parts of the system.