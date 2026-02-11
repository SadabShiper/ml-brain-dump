# Understanding Seq2Seq Models for Neural Machine Translation

This README explains how **Seq2Seq (Sequence-to-Sequence)** models work, both the basic version and the attention-based version. These notes are written as a learning reference for someone new to these concepts.

**Inspiration and Learning Source:**
- Jay Alammar's blog post: [Visualizing Neural Machine Translation Mechanics of Seq2Seq Models with Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
---

## What are Seq2Seq Models Used For?

Seq2Seq models are used for tasks where we need to **convert one sequence into another sequence**:
- **Neural Machine Translation** (translating from one language to another)
- **Image captioning** (generating text descriptions from images)
- **Text summarization**
- **Question answering**

---

## Basic Seq2Seq Model

The basic Seq2Seq model has **two main parts** working together:

### 1. Encoder RNN
- Takes input sentence **word by word**
- At each time step, it produces a **hidden state**
- Each hidden state gets passed to the next time step
- Continues until it processes the **last token** of the input sentence
- The final hidden state contains information about the **entire input sentence**

### 2. Decoder RNN
- Takes the encoder's final hidden state as a starting point (called the **context vector**)
- This context vector represents the **meaning of the entire input sentence**
- The decoder also maintains its own hidden states
- Produces **output words one at a time**
- Each output word depends on the context vector and previous outputs

### Simple Example:
```
Input (English): "How are you"
Encoder processes: "How" → "are" → "you"
Context vector: [final hidden state containing sentence meaning]
Decoder produces: "Comment" → "allez" → "vous" (French translation)
```

### Limitation of Basic Seq2Seq:
The decoder only sees **one context vector** for the entire input sentence. For long sentences, this single vector might not capture all the important information.

---

## Attention-Based Seq2Seq Model

The **attention mechanism** solves the limitation by letting the decoder **look at all parts** of the input sentence when generating each output word.

### Key Differences from Basic Seq2Seq:

**1. Store All Encoder Hidden States**
- Instead of just keeping the final hidden state, we save **all hidden states** from the encoder
- This gives us access to information from **every word** in the input sentence

**2. How Decoder Works with Attention (Step by Step):**

Let's say we're translating `"How are you"` and we're at time step `t=4` in the decoder.

**Step 1:** Decoder takes a start token and initial hidden state to produce its first hidden state `h4`

**Step 2:** Calculate attention scores
- Use `h4` to compute a **score** with each encoder hidden state
- This score tells us **how relevant** each input word is for generating the current output word

**Step 3:** Apply softmax to scores
- Convert scores into **attention weights** that sum to 1
- Higher weights mean **more important words**

**Step 4:** Create weighted sum
- Multiply each encoder hidden state by its **attention weight**
- This focuses more on important words and less on irrelevant ones

**Step 5:** Sum everything up
- Add all the weighted encoder hidden states together
- This gives us a **context vector** `C4` for this specific time step

**Step 6:** Combine information
- **Concatenate** `C4` with the decoder hidden state `h4`
- This combined vector has information about both the input and where we are in decoding

**Step 7:** Generate output word
- Pass the concatenated vector through a **feedforward neural network**
- This produces the **output word** at this time step

**Step 8:** Repeat
- Move to the next time step and repeat the process for the next word

### Example Flow:
```
Input: "How are you"
Encoder hidden states: h1, h2, h3 (stored)

Decoder at t=4:
- Current decoder hidden state: h4
- Calculate attention scores with h1, h2, h3
- Get attention weights: [0.7, 0.2, 0.1] (focused mostly on first word)
- Weighted sum: C4 = 0.7*h1 + 0.2*h2 + 0.1*h3
- Concatenate: [h4, C4]
- Feed through network → produces output word "Comment"
```

### Why This Works Better:
- The decoder can **focus on different parts** of the input for each output word
- For long sentences, the model **doesn't forget** earlier words
- The attention weights show us **which input words influenced** each output word

---

## Key Takeaways

**Basic Seq2Seq:**
- Simple architecture with **encoder and decoder**
- Uses a **single context vector**
- Works okay for **short sentences**
- Struggles with **longer sequences**

**Attention-Based Seq2Seq:**
- Uses **all encoder hidden states**
- Creates a **different context vector** for each decoder step
- Much **better performance** on longer sentences
- More **interpretable** (we can see what the model is paying attention to)

---

## Further Learning

To dive deeper into these concepts, check out:
- Jay Alammar's visual guide on Seq2Seq and attention mechanisms
- The original "Attention is All You Need" paper for Transformer models
- Tensorflow and PyTorch tutorials on implementing Seq2Seq models
