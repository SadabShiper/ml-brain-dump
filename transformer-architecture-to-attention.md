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

## Step 3: Multi-Head Attention

Single self-attention is good but it has limits. **Multi-head attention** solves this by running **multiple self-attention modules in parallel**.

### Why Multi-Head?

It helps in two ways:

**1. Focus on different positions** - A single attention head's output `Z` can be dominated by the word itself. With multiple heads, different heads can focus on different parts of the sentence. For example when encoding `"it"` in `"The animal didn't cross the street because it was too tired"`, one head focuses on `"animal"` and another focuses on `"tired"`.

**2. Learn multiple types of relationships** - Each head learns a different kind of relationship between words in parallel.

### How it Works

The original Transformer uses **8 attention heads**. Each head has its own set of weight matrices:

```
Head 1: W^Q_1, W^K_1, W^V_1  →  Z1
Head 2: W^Q_2, W^K_2, W^V_2  →  Z2
...
Head 8: W^Q_8, W^K_8, W^V_8  →  Z8
```

The self-attention calculation is exactly the same as before, just done **8 times independently** with different weight matrices. Each head produces its own `Z` matrix.

### Combining the Heads

The FFNN expects a **single matrix**, not 8. So we need to combine them:

**Step 1:** Concatenate all 8 Z matrices side by side

```
[Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8]
```

**Step 2:** Multiply by `W_O` - a weight matrix that is **learned during training**

```
Final Z = [Z1, Z2, ..., Z8]  ×  W_O
```

> **What does `W_O` do?**
> It compresses the concatenated matrix back down to the right size so the FFNN can take it as input. Think of it as a learned compression step.

**Step 3:** Pass the final `Z` to the FFNN

### Full Multi-Head Flow

```
Input X
  ↓
8 attention heads run in parallel (each with own W^Q, W^K, W^V)
  ↓
8 separate Z matrices: Z1, Z2, ..., Z8
  ↓
Concatenate → [Z1, Z2, ..., Z8]
  ↓
Multiply by W_O (learned during training)
  ↓
Final Z → passed to FFNN
```
---

## Step 4: Positional Encoding

> **Note on order:** Positional encoding actually happens **before** self-attention in the real model - right after the input embeddings. It is covered here later because that is the order it appears in the blog.

The model has no built-in sense of word order. Positional encoding fixes this by **adding a vector to each input embedding** before it enters the encoder.

### What does it do?
- Gives the model a way to know **where each token sits** in the sentence
- Each position gets a **unique vector** of size 512
- The model uses this to distinguish between the same word appearing at different positions

### How is it calculated?
- The left half of the positional encoding vector uses a **sine function**
- The right half uses a **cosine function**
- These two are combined to form one positional encoding vector per token

```
Position 1 → [sin(...), sin(...), ..., cos(...), cos(...)]  ← size 512
Position 2 → [sin(...), sin(...), ..., cos(...), cos(...)]  ← size 512
...
```

> **Common Misconception:** Positional encoding does not block relationships between distant tokens. The model can still attend to **any position**. What it does is give each position a unique signature so the model knows the order. Relationships between tokens are still decided by the attention scores.

---

## Step 5: Residual Connections and Layer Normalization

Each sub-layer (self-attention and FFNN) inside every encoder and decoder has a **residual connection** wrapped around it, followed by **layer normalization**.

### What is a Residual Connection?
The input to a sub-layer is **added back** to the output of that sub-layer:

```
output = LayerNorm( input + sublayer(input) )
```

### Why does it exist?
The primary reason is to solve the **vanishing gradient problem** during training. Gradients can flow directly through the residual path without passing through the transformation, making it easier to train deep networks.

> A useful side effect is that the original input information is preserved even after transformation.

### Full Encoder Sub-layer Flow:
```
Input
  ↓
Self-Attention
  ↓
Add input + attention output  ← residual connection
  ↓
Layer Normalization
  ↓
FFNN
  ↓
Add input + FFNN output  ← residual connection
  ↓
Layer Normalization
  ↓
Output to next encoder
```

This same pattern applies to **every sub-layer in the decoder** as well.

---

## Step 6: The Decoder - Full Walkthrough

Now that the encoder is done, the decoder takes over. Here is how it works step by step:

**Step 1:** Start with the `<START>` token, calculate its **input embedding** and add **positional encoding**, same as the encoder side

**Step 2:** Pass to the **Masked Self-Attention** layer
- The decoder can only look at **previously generated tokens**, not future ones
- Future positions are set to `-inf` before softmax so they get zeroed out after softmax

**Step 3:** **Add & Layer Normalize** (residual connection)

**Step 4:** Pass to the **Encoder-Decoder Attention** layer
- The **Query** vector comes from the decoder (output of step 3)
- The **Key** and **Value** vectors come from the **final encoder layer output**
- This is how the decoder looks at the input sentence while generating each word

**Step 5:** **Add & Layer Normalize** (residual connection)

**Step 6:** Pass through the **FFNN**

**Step 7:** **Add & Layer Normalize** (residual connection)

**Step 8:** Pass through a **Linear layer + Softmax**
- Linear layer projects the output into a vector the size of the vocabulary (e.g. 10,000 words)
- Softmax converts these into **probabilities** that add up to 1
- The word with the **highest probability** is selected as the output (**greedy decoding**)

> **Alternative to greedy decoding:** Beam search keeps the top N candidates at each step and picks the overall best sequence at the end. This often gives better results than always picking the single best word.

**Step 9:** The output word is **fed back into the bottom decoder** as input for the next step

**Step 10:** Repeat until the `<END>` token is generated

### Full Decoder Flow:
```
<START> token + positional encoding
  ↓
Masked Self-Attention (can't see future tokens)
  ↓
Add & Layer Norm
  ↓
Encoder-Decoder Attention (Q from decoder, K and V from encoder)
  ↓
Add & Layer Norm
  ↓
FFNN
  ↓
Add & Layer Norm
  ↓
Linear + Softmax
  ↓
Output word → fed back to bottom decoder for next step
```

---

## Step 7: Loss Function and Training

### Forward Pass
During training, the model does the same forward pass as during inference. But now we have the **correct answer** to compare against.

### Comparing the Output
- The model outputs a **probability distribution** over the vocabulary at each step
- We compare this with the **target token** (the correct word)
- The difference is measured using **Cross-Entropy** or **KL-Divergence**

```
Model output:  [0.1, 0.6, 0.05, ...]  ← probability for each word
Target:        [0.0, 1.0, 0.00, ...]  ← correct word has probability 1
Loss = difference between the two distributions
```

### Backpropagation
- The loss is used to **update all the weight matrices** in the model via backpropagation
- This includes `W^Q`, `W^K`, `W^V`, `W_O`, and all FFNN weights
- Over many training steps, the model gets better at predicting the correct output

---

## Key Takeaways So Far

| Topic | Key Point |
|---|---|
| Architecture | 6 encoder + 6 decoder layers, final encoder output goes to **all** decoder layers |
| Self-Attention | Q, K, V vectors of size **64**, weight matrices are **learned** |
| Multi-Head Attention | 8 heads in parallel, `W_O` compresses back to size **512** |
| Positional Encoding | Unique vector per position using sine + cosine, does **not** block distant attention |
| Residual Connection | Solves **vanishing gradient**, applied around every sub-layer |
| Decoder | Masked self-attention → encoder-decoder attention → FFNN → linear + softmax |
| Loss Function | Cross-entropy or KL-divergence between output and target distribution |

---

## Further Reading

- Jay Alammar's blog: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Original paper: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
