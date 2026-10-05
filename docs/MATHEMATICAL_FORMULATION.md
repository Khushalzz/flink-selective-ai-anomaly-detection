# Mathematical Formulations & Theoretical Foundations

This document provides the formal mathematical derivations, statistical calibration theorems, and queuing theory proofs underpinning the Big Data Technologies (BDT) project.

---

## 1. Numerically Stable Streaming Statistics: Welford's Algorithm

In high-throughput stream processing, accumulating running sums to compute variance via the textbook formula:
$$s_n^2 = \frac{1}{n-1} \left( \sum_{i=1}^n x_i^2 - \frac{1}{n} \left(\sum_{i=1}^n x_i\right)^2 \right)$$
suffers from **catastrophic cancellation** in IEEE 754 floating-point arithmetic when the mean $\mu \gg \sigma$, resulting in severe precision loss or negative variances.

### Theorem: Welford's Recurrence
Let $x_1, x_2, \dots, x_n$ be an unbounded sequence of sensor observations. The running sample mean $\mu_n$ and sum of squared deviations $M_{2, n} = \sum_{i=1}^n (x_i - \mu_n)^2$ satisfy the following single-pass recurrences:

$$\mu_n = \mu_{n-1} + \frac{x_n - \mu_{n-1}}{n}$$

$$M_{2, n} = M_{2, n-1} + (x_n - \mu_{n-1})(x_n - \mu_n)$$

### Proof
By definition:
$$M_{2, n} = \sum_{i=1}^n (x_i - \mu_n)^2 = \sum_{i=1}^n x_i^2 - n \mu_n^2$$
Expressing $\sum_{i=1}^n x_i^2$ in terms of $M_{2, n-1}$:
$$\sum_{i=1}^n x_i^2 = M_{2, n-1} + (n-1)\mu_{n-1}^2 + x_n^2$$
Substituting $\mu_n = \frac{(n-1)\mu_{n-1} + x_n}{n}$:
$$M_{2, n} - M_{2, n-1} = (n-1)\mu_{n-1}^2 + x_n^2 - n\mu_n^2$$
$$= (n-1)\mu_{n-1}^2 + x_n^2 - n\left(\mu_{n-1} + \frac{x_n - \mu_{n-1}}{n}\right)^2$$
Expanding and factoring:
$$M_{2, n} - M_{2, n-1} = (x_n - \mu_{n-1})\left(x_n - \mu_{n-1} - \frac{x_n - \mu_{n-1}}{n}\right) = (x_n - \mu_{n-1})(x_n - \mu_n) \quad \blacksquare$$

The unbiased sample variance and standard deviation are obtained in $O(1)$ time:
$$s_n^2 = \frac{M_{2, n}}{n - 1}, \quad \sigma_n = \sqrt{s_n^2}$$

---

## 2. Platt Scaling: Sigmoidal Probability Calibration

Raw outputs from unsupervised detectors are non-probabilistic:
- Isolation Forest yields path depth scores $S_{IF} \in [0, 1]$.
- Autoencoder yields reconstruction Mean Squared Error $\text{MSE}_{AE} \in [0, \infty)$.

To compute meaningful ensemble averages and evaluate epistemic uncertainty, raw detector outputs must be mapped to well-calibrated posterior probabilities $P(\text{anomaly} \mid x) \in [0, 1]$.

### Logistic Sigmoid Formulation
We fit scalar parameters $(\alpha, \beta)$ by minimizing the negative log-likelihood (binary cross-entropy) on the strictly held-out validation split $\mathcal{D}_{\text{val}} = \{(x_i, y_i)\}_{i=1}^{N_{\text{val}}}$:

$$\min_{\alpha, \beta} -\sum_{i=1}^{N_{\text{val}}} \left[ y_i \log \sigma(\alpha \cdot s_i + \beta) + (1 - y_i) \log (1 - \sigma(\alpha \cdot s_i + \beta)) \right]$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$.

1. **Isolation Forest Calibration:**
   $$P_{IF}(x) = \sigma(\alpha_{IF} \cdot (-S_{IF}) + \beta_{IF})$$
   (Fitted parameters: $\alpha_{IF} = 22.84, \beta_{IF} = 11.23$)

2. **Autoencoder Reconstruction Calibration:**
   $$P_{AE}(x) = \sigma(\alpha_{AE} \cdot \text{MSE}_{AE} + \beta_{AE})$$
   (Fitted parameters: $\alpha_{AE} = 4.12, \beta_{AE} = -1.85$)

---

## 3. Epistemic Uncertainty Gating Criteria

An observation $x$ is flagged as **uncertain** (triggering Tier 2 fallback) if it satisfies either of two conditions:

### Criterion 1: Margin Uncertainty (Boundary Ambiguity)
When a detector's posterior probability lies in the unconfident region:
$$U_{\text{margin}}(x) \iff (0.35 < P_{IF}(x) < 0.65) \lor (0.35 < P_{AE}(x) < 0.65)$$

### Criterion 2: Structural Disagreement (Detector Conflict)
When two fundamentally distinct detector families (subspace tree partitioning vs. manifold neural reconstruction) arrive at opposing classifications:
$$U_{\text{conflict}}(x) \iff \text{sgn}(P_{IF}(x) - 0.5) \neq \text{sgn}(P_{AE}(x) - 0.5)$$

### Compound Gate Decision
$$\mathcal{G}(x) = \begin{cases} 
\text{Escalate to Tier 2 (XGBoost)}, & \text{if } U_{\text{margin}}(x) \lor U_{\text{conflict}}(x) \\ 
\text{Emit Fast Path } (P_C = \frac{P_{IF} + P_{AE}}{2}), & \text{otherwise} 
\end{cases}$$

Empirically on the Intel Berkeley sensor dataset, this gate yields an escalation rate of **$r = 20.51\%$** (8,204 out of 40,000 windows).

---

## 4. Queuing Theory & Streaming Stability Proof

Consider a high-velocity sensor stream arriving according to a Poisson process with parameter $\lambda$ (events/second).

### Definitions
- Let $S_{\text{fast}}$ be the service time of Tier 1 In-JVM ONNX inference ($S_{\text{fast}} = 1.5\ \mu\text{s}$).
- Let $S_{\text{fallback}}$ be the service time of the Tier 2 fallback engine.
- Let $r \in [0, 1]$ be the escalation rate determined by the uncertainty gate.

The expected per-event service time $\bar{S}(r)$ across the entire streaming pipeline is:
$$\bar{S}(r) = S_{\text{fast}} + r \cdot S_{\text{fallback}}$$

By the **Kingman-Little Stability Theorem** for open queuing networks, a queueing system remains stable (finite buffer size, bounded latency) if and only if the traffic intensity $\rho < 1$:
$$\rho = \lambda \cdot \bar{S}(r) < 1 \iff \lambda < \lambda_{\max} = \frac{1}{S_{\text{fast}} + r \cdot S_{\text{fallback}}}$$

### Case 1: System D (Conventional ML / XGBoost Fallback)
Here, $S_{\text{fallback}} = 1.2\ \mu\text{s}$.
At $r = 0.205$:
$$\bar{S}(0.205) = 1.5\ \mu\text{s} + 0.205 \times 1.2\ \mu\text{s} = 1.746\ \mu\text{s}$$
$$\lambda_{\max} = \frac{1}{1.746 \times 10^{-6}\text{ s}} \approx \mathbf{572,700\text{ events/second}}$$

Because physical Kafka/Flink network line rate is $\approx 22,000\text{ events/s}$, $\rho \ll 1$, guaranteeing **zero backpressure** and deterministic sub-millisecond tail latency.

### Case 2: System F (Heavyweight Transformer / Real Laya Fallback on CPU)
Here, $S_{\text{fallback}} = 833.3\ \text{ms} = 833,333\ \mu\text{s}$.
At $r = 0.205$:
$$\bar{S}(0.205) = 1.5\ \mu\text{s} + 0.205 \times 833,333\ \mu\text{s} \approx 170,835\ \mu\text{s} \approx 0.1708\text{ seconds}$$
$$\lambda_{\max} = \frac{1}{0.1708\text{ s}} \approx \mathbf{5.85\text{ events/second}}$$

### Corollary: Throughput Collapse Theorem
When arrival rate $\lambda = 10,000\text{ events/sec}$ and $r = 0.205$, the traffic intensity for System F is:
$$\rho = 10,000 \times 0.1708 = \mathbf{1,708} \gg 1$$
Because $\rho > 1$, the queue length $L_q(t) \to \infty$ as $t \to \infty$. Flink's credit-based flow control triggers immediate downstream backpressure, propagating up through the Netty transport layer to pause Kafka partition consumption. Ingestion throughput collapses from $9,959\text{ ev/s}$ down to the service ceiling $\lambda_{\max} \approx 6.0\text{ ev/s}$ (**a $99.94\%$ collapse**). $\quad \blacksquare$
