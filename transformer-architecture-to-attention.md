# Understanding The Illustrated Transformer

These are my personal notes on how the Transformer model works, written as I read through Jay Alammar's blog. Written while learning, not after mastering.

**Inspiration and Learning Source:**
- Jay Alammar's blog post: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

## What is the Transformer?

The Transformer is a model that uses **attention** to handle sequence tasks like Neural Machine Translation. The key advantage over Seq2Seq models is that it can be **trained in parallel**, making it much faster to train.

> **Why does this matter?**
> RNN-based Seq2Seq models process words one by one. Transformers process all words at the same time.

---

## High-Level Architecture

At the highest level, the Transformer is still an **encoder-decoder** architecture just like Seq2Seq. But instead of RNNs, it uses stacked layers of attention and feedforward networks.

```
Input Sentence → [Encoder Stack] → [Decoder Stack] → Output Sentence
```

### The Stacks

- The **encoder stack** has 6 identical encoder layers stacked on top of each other
- The **decoder stack** has 6 identical decoder layers stacked on top of each other
- The number 6 is not special - you can experiment with other numbers

> **Important:** The output of the **final encoder layer** is passed to **every decoder layer**, not just the first one.

---

## Inside the Encoder

Each encoder layer has **two components**:

### 1. Self-Attention Module
- Allows each word to look at **other words** in the same sentence
- Helps the model understand context and relationships between words
- Example: In `"The animal didn't cross the street because it was too tired"`, self-attention helps the model figure out that `"it"` refers to `"animal"` and not `"street"`

### 2. Feed-Forward Neural Network (FFNN)
- Applied **independently** to each word position
- No dependencies between positions here, so it can run **in parallel**

```
Input → Self-Attention → FFNN → Output (passed to next encoder)
```

> **Note:** Only the **first encoder** receives word embeddings as input. Every other encoder receives the **output of the encoder below it**.

---

## Inside the Decoder

Each decoder layer has **three components**:

### 1. Self-Attention Module
- Same idea as encoder self-attention
- But only looks at **previously generated words** (can't peek at future words)

### 2. Encoder-Decoder Attention Module
- This is similar to the attention in Seq2Seq models
- Helps the decoder **focus on relevant parts** of the input sentence
- Takes information from the **encoder stack output**

### 3. Feed-Forward Neural Network (FFNN)
- Same as in the encoder

```
Input → Self-Attention → Encoder-Decoder Attention → FFNN → Output
```

---

## Step 1: Input Embeddings

Before anything reaches the encoder, each word needs to be converted into a vector. Here is how it works:

**Step 1:** Take the input sentence and **tokenize** each word

**Step 2:** Convert each token into a vector using an **embedding algorithm**
- Each word becomes a vector of size **512**
- This only happens at the **bottom-most encoder**

**Step 3:** The embedding vectors flow through the encoder layers

### Example:
```
Sentence: "How are you"

After embedding:
"How" → [0.2, 0.5, ..., 0.1]  ← vector of size 512
"are" → [0.8, 0.1, ..., 0.4]  ← vector of size 512
"you" → [0.3, 0.9, ..., 0.7]  ← vector of size 512

Combined input matrix: 3 x 512
```

> **Note:** Positional encoding is also added to these embeddings to tell the model the position of each word. This will be covered separately.

---

## Step 2: Self-Attention Calculation

This is the most important part of the Transformer. Let's break it down step by step.

Say we have a sentence with 4 words. The encoder receives a `4 x 512` input matrix.

### The Three Vectors: Q, K, V

For each word, we need to create three vectors:
- `Q` - **Query vector** (what this word is looking for)
- `K` - **Key vector** (what this word can offer)
- `V` - **Value vector** (the actual content of this word)

These are created using **three weight matrices** `W^Q`, `W^K`, `W^V` that are **learned during training**.

> **Important:** These weight matrices start randomly but get better as the model trains. They are not fixed.

The vectors are smaller than the embedding - size **64** instead of 512. This is an architecture choice to keep computation manageable.

```
For each word x:
Q = x  ×  W^Q
K = x  ×  W^K
V = x  ×  W^V
```

### The Calculation (Step by Step)

**Step 1:** Generate `Q`, `K`, `V` vectors for each word using the weight matrices

**Step 2:** Calculate the **score**
- Take the dot product of the query `Q` with the key `K` of every other word
- This tells us **how much attention** to pay to other words

```
score = Q . K
```

**Step 3:** **Scale** the score
- Divide by the square root of `d_k` (which is 8, since key vectors are size 64)
- This keeps the gradients stable during training

```
scaled score = score / sqrt(d_k)
```

**Step 4:** Apply **softmax**
- Converts scores into weights that add up to 1
- Higher score = more attention to that word

**Step 5:** Multiply softmax output with the **Value vector** `V`
- Words with higher attention scores keep more of their value
- Less relevant words get multiplied by small numbers and fade out

**Step 6:** **Sum** all the weighted value vectors
- This gives us the final attention output `Z` for each word

```
Z = softmax( Q.K / sqrt(d_k) ) × V
```

### What does Z look like?

- `Z` is a matrix where **each row represents one word**
- Each row has size **64** (not 512 - that comes later with multi-head attention)
- Each row contains the attention-weighted information for that word

```
Input:  4 words × 512 embedding
Q, K, V: 4 words × 64
Z output: 4 words × 64
```

---

## Key Takeaways So Far

| Architecture | Self-Attention |
|---|---|
| 6 encoder + 6 decoder layers | Creates Q, K, V vectors of size **64** |
| Only first encoder gets word embeddings | Weight matrices are **learned**, not fixed |
| Final encoder output goes to all decoder layers | Output Z has size **64** per word |

---

## Further Reading

- Jay Alammar's blog: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Original paper: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
