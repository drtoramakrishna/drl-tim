# Mathematical Formulations and Algorithmic Details

## Table of Contents
1. [Problem Formulation](#1-problem-formulation)
2. [Topic-Aware Propagation Models](#2-topic-aware-propagation-models)
3. [Graph Embedding: Diffusion2Vec](#3-graph-embedding-diffusion2vec)
4. [Deep Influence Evaluation Model (DIEM)](#4-deep-influence-evaluation-model-diem)
5. [Reinforcement Learning Formulation](#5-reinforcement-learning-formulation)
6. [Training Algorithm](#6-training-algorithm)
7. [Complexity Analysis](#7-complexity-analysis)

---

## 1. Problem Formulation

### 1.1 Classical Influence Maximization

**Input**: 
- Graph $G = (V, E)$ with $|V| = n$, $|E| = m$
- Budget constraint $k \in \mathbb{Z}^+$
- Diffusion model $\mathcal{M}$ (IC or LT)

**Output**: Seed set $S^* \subseteq V$ with $|S^*| = k$

**Objective**:
$$S^* = \arg\max_{S \subseteq V, |S|=k} \sigma_G(S)$$

where $\sigma_G(S)$ is the expected influence spread of seed set $S$ under model $\mathcal{M}$.

### 1.2 Topic-Aware Influence Maximization (TIM)

**Enhanced Input**:
- Query topics $\tau \subseteq \mathcal{T}$ where $\mathcal{T}$ is the universal topic set
- User profiles $\{P_v\}_{v \in V}$ where $P_v \in [0,1]^{|\mathcal{T}|}$

**Targeted Influence Function**:
$$\sigma_G(S|\tau) = \mathbb{E}\left[\sum_{v \in \text{Activated}(S)} B_v^{(\tau)}\right]$$

where $B_v^{(\tau)} = \sum_{t \in \tau} P_v[t]$ is the benefit of user $v$ under topics $\tau$.

**TIM Objective**:
$$S^* = \arg\max_{S \subseteq V, |S|=k} \sigma_G(S|\tau)$$

### 1.3 Greedy Framework Formalization

The greedy algorithm maintains a partial solution $S_i$ at step $i$ and selects:

$$v_{i+1} = \arg\max_{v \in V \setminus S_i} \Delta(v|S_i, \tau)$$

where $\Delta(v|S_i, \tau) = \sigma_G(S_i \cup \{v\}|\tau) - \sigma_G(S_i|\tau)$ is the marginal gain.

**Algorithm Template**:
```
Input: G, k, τ, evaluation function Q̂
Output: Seed set S

1. S₀ ← ∅
2. for i = 0 to k-1:
3.    v* ← argmax_{v∈V\Sᵢ} Q̂((v, Sᵢ)|τ)
4.    Sᵢ₊₁ ← Sᵢ ∪ {v*}
5. return Sₖ
```

---

## 2. Topic-Aware Propagation Models

### 2.1 Three Core Metrics

The paper introduces three metrics for topic-aware influence:

#### 2.1.1 Contact Frequency
Represented by initial edge weights $w_{u,v} \in (0,1]$:
- Models frequency of interaction between users $u$ and $v$
- Can be derived from communication logs, social media interactions
- Higher values indicate stronger social ties

#### 2.1.2 User Similarity
Cosine similarity between user profiles:
$$\text{sim}(u,v) = \frac{P_u \cdot P_v}{||P_u||_2 \cdot ||P_v||_2} = \frac{\sum_{t=1}^{|\mathcal{T}|} P_u[t] \cdot P_v[t]}{\sqrt{\sum_{t=1}^{|\mathcal{T}|} P_u[t]^2} \cdot \sqrt{\sum_{t=1}^{|\mathcal{T}|} P_v[t]^2}}$$

**Properties**:
- $\text{sim}(u,v) \in [0,1]$
- $\text{sim}(u,u) = 1$ (reflexive)
- $\text{sim}(u,v) = \text{sim}(v,u)$ (symmetric)

#### 2.1.3 User Benefit
Topic-specific interest measure:
$$B_v^{(\tau)} = \sum_{t \in \tau} P_v[t]$$

Represents how much user $v$ cares about the query topics $\tau$.

### 2.2 Topic-Aware Independent Cascade Model

#### 2.2.1 Activation Probability Formula
For edge $(u,v)$ under topics $\tau$:

$$p_{u,v}^{(\tau)} = \frac{\alpha_1 w_{u,v} + \alpha_2 \text{sim}(u,v) + \alpha_3 B_v^{(\tau)}}{3}$$

**Parameter Constraints**:
- $\alpha_1, \alpha_2, \alpha_3 \in (0,1)$ are balance parameters
- $\alpha_1 + \alpha_2 + \alpha_3 = 3$ (implicit normalization)
- All three terms are in $(0,1]$, ensuring $p_{u,v}^{(\tau)} \in (0,1]$

#### 2.2.2 Diffusion Process
At time step $t$:

1. **Activation Attempt**: For each newly activated user $u \in A_t \setminus A_{t-1}$:
   $$\forall v \in N_{out}(u) \cap I_{t-1}: \text{ activate } v \text{ with probability } p_{u,v}^{(\tau)}$$

2. **State Update**: 
   - $A_t \leftarrow A_{t-1} \cup \{\text{newly activated nodes}\}$
   - $I_t \leftarrow I_{t-1} \setminus \{\text{newly activated nodes}\}$

3. **Termination**: When $A_t = A_{t-1}$ (no new activations)

#### 2.2.3 Expected Influence Computation
The targeted influence under topic-aware IC model:

$$\sigma_{IC}(S|\tau) = \mathbb{E}\left[\sum_{v \in V} \mathbf{1}_{v \text{ activated}} \cdot B_v^{(\tau)}\right]$$

### 2.3 Topic-Aware Linear Threshold Model

#### 2.3.1 Modified Parameters

**Edge Weights**:
$$b_{u,v}^{(\tau)} = \alpha_1 w_{u,v} + \alpha_2 \text{sim}(u,v)$$

with normalization constraint: $\sum_{u \in N_{in}(v)} b_{u,v}^{(\tau)} \leq 1$

**Node Thresholds**:
$$\theta_v^{(\tau)} = 1 - \alpha_3 B_v^{(\tau)}$$

**Intuition**: Users more interested in the topics (higher $B_v^{(\tau)}$) have lower thresholds, making them easier to activate.

#### 2.3.2 Activation Condition
Node $v$ activates at time $t$ if:
$$\sum_{u \in N_{in}(v) \cap A_{t-1}} b_{u,v}^{(\tau)} \geq \theta_v^{(\tau)}$$

#### 2.3.3 Activation Probability Analysis
For a node $v$ with active in-neighbors $S \subseteq N_{in}(v)$:

$$P(\text{v activates}) = P\left(\sum_{u \in S} b_{u,v}^{(\tau)} \geq \theta_v^{(\tau)}\right)$$

Since $\theta_v^{(\tau)}$ can be treated as deterministic (sampled once), this becomes:
$$P(\text{v activates}) = \mathbf{1}\left[\sum_{u \in S} b_{u,v}^{(\tau)} \geq 1 - \alpha_3 B_v^{(\tau)}\right]$$

---

## 3. Graph Embedding: Diffusion2Vec

### 3.1 Node Feature Representation

Each node $v$ is associated with feature vector $X_v$:

$$X_v = [X_v[0], X_v[1], X_v[2], \ldots, X_v[|\mathcal{T}|]]$$

where:
- $X_v[0] = \begin{cases} 1 & \text{if } v \in S \text{ (selected)} \\ 0 & \text{otherwise} \end{cases}$ (Selection indicator)
- $X_v[1:] = P_v$ (User profile vector)

### 3.2 Iterative Embedding Update

The embedding update rule at iteration $t+1$:

$$u_v^{(t+1)} = \text{relu}\left(\Theta_1 X_v + \Theta_2 \sum_{u \in N(v)} u_u^{(t)} + \Theta_3 \sum_{u \in N(v)} \text{relu}(\Theta_4 p_{u,v}^{(\tau)})\right)$$

**Parameter Dimensions**:
- $\Theta_1 \in \mathbb{R}^{p \times (|\mathcal{T}|+1)}$: Node feature transformation
- $\Theta_2 \in \mathbb{R}^{p \times p}$: Neighbor embedding aggregation  
- $\Theta_3 \in \mathbb{R}^{p \times p}$: Weighted aggregation
- $\Theta_4 \in \mathbb{R}^p$: Edge weight transformation
- $p$: Embedding dimension

### 3.3 Neighborhood Aggregation Analysis

The aggregation terms can be expanded as:

**Neighbor Embeddings**:
$$\sum_{u \in N(v)} u_u^{(t)} = \sum_{u \in N_{in}(v)} u_u^{(t)} + \sum_{u \in N_{out}(v)} u_u^{(t)}$$

**Weighted Edge Information**:
$$\sum_{u \in N(v)} \text{relu}(\Theta_4 p_{u,v}^{(\tau)}) = \sum_{u \in N_{in}(v)} \text{relu}(\Theta_4 p_{u,v}^{(\tau)}) + \sum_{u \in N_{out}(v)} \text{relu}(\Theta_4 p_{v,u}^{(\tau)})$$

### 3.4 Multi-Scale Information Capture

After $T$ iterations, embedding $u_v^{(T)}$ contains information from:
- **0-hop**: Node $v$ itself via $\Theta_1 X_v$
- **1-hop**: Direct neighbors via $\Theta_2 \sum_{u \in N(v)} u_u^{(T-1)}$
- **2-hop**: Neighbors of neighbors via recursive propagation
- **T-hop**: All nodes within distance $T$ from $v$

**Information Flow Equation**:
$$u_v^{(T)} = f_T(X_v, \{X_u : u \in B_T(v)\}, \{p_{u,w}^{(\tau)} : (u,w) \in E \cap (B_T(v) \times B_T(v))\})$$

where $B_T(v)$ is the T-hop neighborhood of $v$.

### 3.5 Initialization and Convergence

**Initialization**: $u_v^{(0)} = \mathbf{0} \in \mathbb{R}^p$ for all $v \in V$

**Fixed Point Analysis**: Under certain conditions on $\Theta$ parameters, the iteration converges:
$$\lim_{t \to \infty} ||u_v^{(t+1)} - u_v^{(t)}|| = 0$$

**Practical Termination**: Usually $T = 4$ iterations suffice for good performance.

---

## 4. Deep Influence Evaluation Model (DIEM)

### 4.1 Architecture Design

The DIEM function $\hat{Q}$ estimates marginal influence:

$$\hat{Q}(S, v; \Theta, \tau) = \Theta_5 \text{relu}\left([\Theta_6 \sum_{u \in S} u_u^{(T)}, \Theta_7 u_v^{(T)}]\right)$$

**Component Analysis**:
- **Partial Solution Encoding**: $\sum_{u \in S} u_u^{(T)}$ aggregates information about current seed set
- **Candidate Node Encoding**: $u_v^{(T)}$ represents the candidate node
- **Concatenation**: $[\cdot, \cdot]$ combines both representations
- **Final Mapping**: $\Theta_5$ maps to scalar influence estimate

### 4.2 Parameter Dimensions

- $\Theta_5 \in \mathbb{R}^{2p}$: Final output layer (maps 2p-dimensional vector to scalar)
- $\Theta_6 \in \mathbb{R}^{p \times p}$: Partial solution transformation
- $\Theta_7 \in \mathbb{R}^{p \times p}$: Candidate node transformation

**Total Parameters**: $\Theta = \{\Theta_1, \Theta_2, \Theta_3, \Theta_4, \Theta_5, \Theta_6, \Theta_7\}$

### 4.3 Approximation Quality Analysis

The DIEM approximates the true marginal gain:
$$\hat{Q}(S, v; \Theta, \tau) \approx \sigma_G(S \cup \{v\}|\tau) - \sigma_G(S|\tau)$$

**Approximation Error**:
$$\epsilon(S, v) = |\hat{Q}(S, v; \Theta, \tau) - [\sigma_G(S \cup \{v\}|\tau) - \sigma_G(S|\tau)]|$$

**Generalization Bound**: Under certain assumptions, the expected error over the distribution of graphs can be bounded using standard PAC learning theory.

### 4.4 Computational Complexity

For a single evaluation $\hat{Q}(S, v; \Theta, \tau)$:

1. **Embedding Computation**: $O(T \cdot m \cdot p)$ for all nodes
2. **Aggregation**: $O(|S| \cdot p)$ for partial solution
3. **Neural Network Forward Pass**: $O(p^2)$

**Total per Evaluation**: $O(T \cdot m \cdot p + |S| \cdot p + p^2)$

For greedy selection with $k$ seeds and $n$ candidates per step:
**Total Complexity**: $O(k \cdot n \cdot (T \cdot m \cdot p + k \cdot p + p^2))$

Compare with Monte Carlo simulation: $O(k \cdot n \cdot R \cdot m)$ where $R \gg T \cdot p$.

---

## 5. Reinforcement Learning Formulation

### 5.1 MDP Components

#### 5.1.1 State Space
State $s_i$ represents the current partial solution:
$$s_i = S_i = \{v_1, v_2, \ldots, v_i\} \subseteq V$$

**State Encoding**: Use pooled embedding $\bar{u}_{S_i} = \sum_{u \in S_i} u_u^{(T)}$

**State Space Size**: $|S| = \sum_{i=0}^k \binom{n}{i}$ (exponential in $k$)

#### 5.1.2 Action Space
At state $s_i$ (with $|S_i| = i$), available actions:
$$A(s_i) = V \setminus S_i = \{v \in V : v \notin S_i\}$$

**Action Representation**: Each action $a = v$ is represented by its embedding $u_v^{(T)}$

#### 5.1.3 Transition Function
Deterministic transition:
$$P(s_{i+1}|s_i, a) = \begin{cases} 1 & \text{if } s_{i+1} = s_i \cup \{a\} \\ 0 & \text{otherwise} \end{cases}$$

#### 5.1.4 Reward Function
Immediate reward for action $a = v$ at state $s_i$:
$$r(s_i, v) = \sigma_G(S_i \cup \{v\}|\tau) - \sigma_G(S_i|\tau)$$

**Cumulative Reward**:
$$R(S_k) = \sum_{i=0}^{k-1} r(s_i, v_{i+1}) = \sigma_G(S_k|\tau) - \sigma_G(\emptyset|\tau) = \sigma_G(S_k|\tau)$$

### 5.2 Value Function Approximation

#### 5.2.1 State-Action Value Function
$$Q^*(s, a) = \mathbb{E}\left[\sum_{t=0}^{k-|s|-1} \gamma^t r(s_t, a_t) \mid s_0 = s, a_0 = a, \pi^*\right]$$

For the undiscounted case ($\gamma = 1$):
$$Q^*(s, a) = r(s, a) + \max_{s', a'} Q^*(s', a')$$

#### 5.2.2 Function Approximation
$$Q^*(s, a) \approx \hat{Q}(s, a; \Theta) = \hat{Q}(\sum_{u \in s} u_u^{(T)}, u_a^{(T)}; \Theta)$$

### 5.3 Bellman Equation for TIM

**Optimal Policy**:
$$\pi^*(s) = \arg\max_{a \in A(s)} Q^*(s, a)$$

**Bellman Optimality**:
$$Q^*(s, a) = r(s, a) + \max_{a' \in A(s \cup \{a\})} Q^*(s \cup \{a\}, a')$$

**Terminal Condition**: When $|s| = k-1$, $Q^*(s, a) = r(s, a)$

---

## 6. Training Algorithm

### 6.1 Double DQN with Prioritized Experience Replay

#### 6.1.1 Experience Tuple
Each experience consists of:
$$e_i = (s_i, a_i, r_i, s_{i+1}, \text{done}_i)$$

where $\text{done}_i = \mathbf{1}[|s_{i+1}| = k]$

#### 6.1.2 Target Computation (Double DQN)
$$y_i = r_i + \gamma \cdot (1 - \text{done}_i) \cdot \hat{Q}(s_{i+1}, \arg\max_{a'} \hat{Q}(s_{i+1}, a'; \Theta); \Theta^-)$$

**Key Insight**: Online network $\Theta$ selects action, target network $\Theta^-$ evaluates it.

#### 6.1.3 TD Error and Priority
**Temporal Difference Error**:
$$\delta_i = |y_i - \hat{Q}(s_i, a_i; \Theta)|$$

**Priority**:
$$p_i = |\delta_i| + \epsilon$$

where $\epsilon > 0$ ensures all experiences have non-zero probability.

#### 6.1.4 Sampling Probability
$$P(i) = \frac{p_i^{\alpha}}{\sum_{j=1}^{|D|} p_j^{\alpha}}$$

where $\alpha \in [0, 1]$ controls prioritization strength:
- $\alpha = 0$: Uniform sampling
- $\alpha = 1$: Full prioritization

#### 6.1.5 Importance Sampling Weight
To correct for bias introduced by prioritized sampling:
$$w_i = \left(\frac{1}{|D|} \cdot \frac{1}{P(i)}\right)^{\beta}$$

Normalized: $w_i = \frac{w_i}{\max_j w_j}$

#### 6.1.6 Loss Function
$$L(\Theta) = \mathbb{E}_{i \sim D}\left[w_i \cdot \left(y_i - \hat{Q}(s_i, a_i; \Theta)\right)^2\right]$$

### 6.2 Training Procedure

#### 6.2.1 Episode Generation
```
1. Initialize s₀ = ∅
2. for step t = 0 to k-1:
3.    if random() < ε:  # ε-greedy exploration
4.       a_t = random action from A(s_t)
5.    else:
6.       a_t = argmax_{a∈A(s_t)} Q̂(s_t, a; Θ)
7.    r_t = σ_G(s_t ∪ {a_t}|τ) - σ_G(s_t|τ)  # True reward
8.    s_{t+1} = s_t ∪ {a_t}
9.    Store (s_t, a_t, r_t, s_{t+1}, done_t) in D
10. Return episode reward Σ_t r_t
```

#### 6.2.2 Batch Training
```
1. Sample batch B = {(s_i, a_i, r_i, s'_i, done_i)} from D with priorities
2. Compute targets: y_i = r_i + γ(1-done_i) Q̂(s'_i, argmax_a Q̂(s'_i,a;Θ); Θ⁻)
3. Compute TD errors: δ_i = |y_i - Q̂(s_i, a_i; Θ)|
4. Update priorities: p_i = |δ_i| + ε
5. Compute loss: L = Σ_i w_i(y_i - Q̂(s_i, a_i; Θ))²
6. Update parameters: Θ ← Θ - η∇_Θ L
7. Periodically: Θ⁻ ← Θ
```

### 6.3 Hyperparameter Settings

From the paper's experimental setup:

| Parameter | Value | Description |
|-----------|--------|-------------|
| $\alpha_1, \alpha_2, \alpha_3$ | 0.8, 0.8, 1.0 | Topic-aware model weights |
| Batch size | 64 | Mini-batch size for training |
| N-step | 3 | Multi-step return horizon |
| $T$ | 4 | Diffusion2Vec iterations |
| $p$ | Variable | Embedding dimension |
| $\epsilon$ | 0.1 | Exploration rate |
| $\alpha$ | 0.7 | Prioritization exponent |
| $\beta$ | 0.5 | Importance sampling exponent |

---

## 7. Complexity Analysis

### 7.1 Time Complexity Breakdown

#### 7.1.1 Training Phase

**Per Episode**:
- Graph embedding: $O(T \cdot m \cdot p)$
- Greedy construction: $O(k \cdot n \cdot p^2)$ 
- True reward computation: $O(k \cdot R \cdot m)$ where $R$ is MC samples

**Per Batch Update**:
- Forward pass: $O(|B| \cdot p^2)$
- Backward pass: $O(|B| \cdot p^2)$
- Priority updates: $O(|B| \log |D|)$

**Total Training**: $O(E \cdot k \cdot (n \cdot p^2 + R \cdot m) + U \cdot |B| \cdot p^2)$
where $E$ is episodes, $U$ is updates.

#### 7.1.2 Inference Phase

**Per Query**:
- Graph embedding: $O(T \cdot m \cdot p)$
- Greedy selection: $O(k \cdot n \cdot p^2)$

**Total Inference**: $O(T \cdot m \cdot p + k \cdot n \cdot p^2)$

### 7.2 Space Complexity

#### 7.2.1 Model Parameters
- Diffusion2Vec: $O(p^2 + p \cdot |\mathcal{T}|)$
- DIEM: $O(p^2)$
- **Total**: $O(p^2 + p \cdot |\mathcal{T}|)$

#### 7.2.2 Runtime Memory
- Node embeddings: $O(n \cdot p)$
- Experience replay buffer: $O(|D| \cdot (k + p))$
- **Total**: $O(n \cdot p + |D| \cdot k)$

### 7.3 Comparison with Baselines

| Method | Training | Inference | Storage |
|--------|----------|-----------|---------|
| WRIS | - | $O(k \cdot n \cdot R \cdot m)$ | $O(n + m)$ |
| IRR | $O(R \cdot m)$ | $O(k \cdot \|\mathcal{R}\|)$ | $O(\|\mathcal{R}\| \cdot n)$ |
| DIEM | $O(E \cdot k \cdot R \cdot m)$ | $O(k \cdot n \cdot p^2)$ | $O(p^2)$ |

where $\|\mathcal{R}\|$ is the number of RR sets (typically $10^5 - 10^6$).

**Key Advantages of DIEM**:
1. **Storage**: $O(p^2) \ll O(\|\mathcal{R}\| \cdot n)$ 
2. **Generalization**: No retraining for new graph instances
3. **Scalability**: $p^2 \ll R \cdot m$ for large graphs

### 7.4 Approximation Quality vs. Efficiency Trade-off

**Approximation Ratio**: 
- Exact greedy: $(1 - 1/e)$-approximation
- DIEM: $(1 - 1/e - \delta)$-approximation where $\delta$ depends on neural network approximation error

**Efficiency Gain**: 
- Speedup factor: $\frac{R \cdot m}{p^2} \approx 10^3 - 10^4$ for typical parameters
- Storage reduction: $\frac{\|\mathcal{R}\| \cdot n}{p^2} \approx 10^5 - 10^6$

This mathematical analysis provides the theoretical foundation for understanding the paper's algorithmic contributions and their computational properties.