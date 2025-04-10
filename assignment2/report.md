# Assignment 2: ASR Decoding Methods Analysis

## Introduction
This report summarizes the experiments performed using different CTC decoding methods for a pre-trained wav2vec2 acoustic model for English speech recognition. Four decoding methods were implemented and compared:

1. Greedy decoding
2. Beam search decoding without language model
3. Beam search with language model shallow fusion
4. Beam search with second-pass language model rescoring

## Experiment Results

### 1. Decoding Methods Comparison (Default Parameters)

The following table shows the average Levenshtein distances for each decoding method across all test samples:

| Decoding Method | Average Levenshtein Distance |
|-----------------|------------------------------|
| Greedy          | 37.25                        |
| Beam Search     | 39.88                        |
| Beam Search + LM | 107.63                      |
| Beam Search + LM Rescoring | 39.88             |

The experimental results indicate that:
- Greedy decoding performed surprisingly well, achieving the lowest average Levenshtein distance.
- Basic beam search (without LM) performed slightly worse than greedy decoding.
- Beam search with language model fusion had poor performance with default parameters.
- The LM rescoring approach produced identical results to beam search.

### 2. Hyperparameter Analysis

Multiple experiments were conducted to evaluate the impact of different hyperparameters:

#### 2.1 Effect of Beam Width

| Beam Width | Effect on Performance |
|------------|------------------------|
| 1 | Reduced search space significantly, leading to worse beam search with LM performance |
| 3 (default) | Balanced performance and computation cost |
| 5 | Slightly better beam search results |
| 7 | Similar to beam width 5; without significant improvements |
| 9 | No further improvement over beam width 7 |

#### 2.2 Effect of LM Weight (Alpha)

| Alpha Value | Effect on Performance |
|-------------|------------------------|
| 0.5 | Significant improvement in beam_lm performance |
| 1.0 (default) | Poor performance for beam_lm due to excessive weight on LM scores |
| 2.0 | Drastically worse performance for beam_lm, with output reduced to few words |
| 5.0 | Extreme degradation with beam_lm outputting only 1-2 characters |

#### 2.3 Effect of Word Insertion Bonus (Beta)

| Beta Value | Effect on Performance |
|------------|------------------------|
| 0.5 | Reduced the quality of beam_lm output |
| 1.0 (default) | Standard baseline performance |
| 2.0 | Improved beam_lm output length but not quality |
| 4.0 | No significant improvement over beta=2.0 |

### 3. Best Configuration

The best configuration found through experimentation was:
- Beam width = 3
- Alpha = 0.5 (LM weight)
- Beta = 1.0 (word insertion bonus)
