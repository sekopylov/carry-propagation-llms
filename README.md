# Carry Propagation in LLMs

## 1) Goal
1. Train a small transformer on synthetic integer addition
2. Study whether a causal activation direction is connected to carry propagation
3. Use activation patching and direction ablation on the `=` token representation to test this behavior


## 2) Synthetic dataset and model training (train_small_llm.ipynb)

### 2.1 Data generation
1. Training samples are generated online
2. Each sample has the form a+b=sum;
3. a and b use mixed digit lengths from min_digits=1 to max_digits=7
4. The semicolon ; is used as the answer-end marker

### 2.2 Tokenization
1. The vocabulary is character-level: digits, +, =, ;, and _ for padding

### 2.3 Model configuration
1. Architecture: decoder-only transformer
2. Layers: 6
3. Hidden size: 256
4. Attention heads: 8

### 2.4 Training results
| split | accuracy | n |
| --- | ---: | ---: |
| all | 0.9720 | 2000 |
| carry | 0.9628 | 1480 |
| no-carry | 0.9981 | 520 |

### 2.5 Interpretation
1. The model learned addition well enough for causal analysis
2. Carry examples remain harder than no-carry examples
3. The gap is small enough that the analysis is not dominated by random model failure
4. Training logs show that addition examples with carry propagation are substantially harder than no-carry examples, and that the model learns to solve them later in training

## 3) Counterfactual Patching Method (carry_propagation_analysis.ipynb)
### 3.1 Clean/corrupt pair construction
1. The analysis samples general addition examples across digit lengths
2. A clean example is kept when a + b_clean has more digits than the largest input number
3. A corrupt example is created by reducing b_clean to b_corrupt so that a + b_corrupt goes back inside the original digit length
4. Example pattern:
    1. clean: 518+526=, sum 1044, outside the 3-digit range
    2. corrupt: 518+303=, sum 821, inside the 3-digit range
5. The notebook keeps pairs where the model predicts the clean first token on the clean prompt and does not predict it on the corrupt prompt
6. The run selected 1024 usable pairs and rejected only 12 sampled candidates during filtering
7. This pipeline allows us to sample a diverse set of examples with the same underlying structure: propagation of a carry `1` in the clean prompt

### 3.2 Patching position
1. The current analysis patches the semantic `=` position

### 3.3 Noising and denoising
1. Noising: run the clean prompt, but replace the clean `=` activation with the corrupt `=` activation
2. Denoising: run the corrupt prompt, but replace the corrupt `=` activation with the clean `=` activation
3. Large scores mean the activation at that layer strongly controls the clean-vs-corrupt first-token decision

## 4) Patching Results
### 4.1 General first-token baseline
| split | accuracy | n |
| --- | ---: | ---: |
| all | 0.9971 | 2048 |
| carry | 0.9961 | 1545 |
| no-carry | 1.0000 | 503 |

### 4.2 Layer scores at the `=` token
| layer | noising score | denoising score |
| ---: | ---: | ---: |
| 0 | 0.0240 | -0.0301 |
| 1 | 0.0430 | 0.0080 |
| 2 | 9.4337 | 9.2013 |
| 3 | 10.7452 | 10.7625 |
| 4 | 10.7756 | 10.7854 |
| 5 | 10.7894 | 10.7894 |

### 4.3 Best component
1. The effect becomes large from layer 2 onward and is strongest at the final layer
2. This suggests the relevant information is already present in middle layers and becomes most directly usable by the output head at the end

### 4.4 Necessity and sufficiency check
| condition | accuracy |
| --- | ---: |
| clean baseline | 1.0000 |
| corrupt baseline | 0.0000 |
| corrupt + clean patch | 1.0000 |
| clean + corrupt patch | 0.0000 |

### 4.5 Interpretation
1. The `=` activation at layer 5 is a very strong control point for the selected clean/corrupt task
2. It is sufficient to restore the clean first-token behavior in corrupt prompts
3. It is necessary for keeping the clean first-token behavior in clean prompts
4. This intervention shows that the `=`-token activation has a strong causal effect on the clean-vs-corrupt first-token prediction. **However, it does not by itself prove the existence of a general carry-propagation circuit.** The selected pairs are constructed so that, in the clean prompt, the answer becomes one digit longer and usually starts with a new leading `1`, while in the corrupt prompt this does not happen. **Therefore, the discovered direction may encode a simpler heuristic, such as “output a new leading `1`” or “the answer should have an extra digit,” rather than the algorithmic process of propagating.**
5. The next sections extend this analysis to better understand what kind of information these hidden states capture and how specific their effect is.

## 5) Carry Group Definition
### 5.1 Carry-chain length
1. carry_chain_len(a, b) scans addition from low digits to high digits
2. It tracks consecutive columns where a carry is produced
3. chain=0 means no carry
4. chain=1 means a carry occurs but does not propagate through multiple consecutive columns
5. chain>=2 means carry propagation across at least two consecutive digit positions

### 5.2 Group names
1. no_carry: no carry and the sum does not start with 1
2. starts_with_1_no_carry: no carry, but the first answer digit is 1
3. outside_single_carry: one carry and the sum has more digits than the largest input
4. outside_propagation: propagated carry and the sum has more digits than the largest input
5. inside_single_carry: one carry but the sum stays within the original digit length
6. inside_propagation: propagated carry but the sum stays within the original digit length
7. starts_with_1_inside_single_carry: inside single-carry case where the first answer digit is 1
8. starts_with_1_inside_propagation: inside propagated-carry case where the first answer digit is 1

### 5.3 Why this grouping is important
1. The groups separate carry mechanics from output-format effects
2. inside_* examples test whether the direction is a general carry direction
3. outside_* examples test whether the direction is an overflow/new-leading-digit direction
4. **starts_with_1_* examples test whether the direction partly encodes the first answer token `1`**

## 6) Direction Ablation Results
### 6.1 Ablation method
1. The notebook computes the mean direction clean_activation - corrupt_activation at layer 5, position `=`
2. On fresh examples, it removes only the projection of the `=` activation onto this direction
3. It then measures whether the model still predicts the correct first answer token

### 6.2 Results by group
| group | n | base acc | ablated acc | drop |
| --- | ---: | ---: | ---: | ---: |
| no_carry | 2584 | 0.9973 | 0.9969 | 0.0004 |
| outside_single_carry | 402 | 1.0000 | 0.2214 | 0.7786 |
| outside_propagation | 706 | 1.0000 | 0.1643 | 0.8357 |
| inside_single_carry | 3244 | 0.9969 | 0.9966 | 0.0003 |
| inside_propagation | 2265 | 0.9987 | 0.9982 | 0.0004 |
| starts_with_1_inside_single_carry | 327 | 1.0000 | 0.6820 | 0.3180 |
| starts_with_1_inside_propagation | 202 | 1.0000 | 0.7228 | 0.2772 |
| starts_with_1_no_carry | 270 | 1.0000 | 0.5259 | 0.4741 |

## 7) Main Interpretation
### 7.1 Hypothesis A: generic carry-propagation circuit
1. Prediction: removing the direction should hurt inside_propagation and inside_single_carry substantially
2. Result: drops are almost zero:
1. inside_single_carry: 0.0003;
2. inside_propagation: 0.0004.
3. Conclusion: this hypothesis is not supported
4. The direction is not a general mechanism for all carries inside the number

### 7.2 Hypothesis B: new-leading-digit feature
1. Prediction: removing the direction should mainly hurt cases where the sum becomes one digit longer than the inputs
2. Result: outside drops are very large:
1. outside_single_carry: 0.7786;
2. outside_propagation: 0.8357.
3. Conclusion: this hypothesis is strongly supported
4. The discovered direction is best interpreted as an overflow or new-leading-digit feature at the = position

### 7.3 Hypothesis C: first-token 1 feature is mixed into the direction
1. Prediction: examples whose answer starts with 1 should be sensitive even when they are not outside-overflow examples (e.g. no-carry).
2. Result: starts_with_1_* groups have meaningful drops:
1. starts_with_1_no_carry: 0.4741;
2. starts_with_1_inside_single_carry: 0.3180;
3. starts_with_1_inside_propagation: 0.2772.
3. Conclusion: this hypothesis is partially supported.
4. The direction seems to include a leading-token-1 component, but the outside-overflow effect is much stronger than the general starts-with-1 effect.

### 7.4 Why outside propagation is strongest
1. outside_propagation combines two properties:
1. a carry chain exists;
2. the carry chain changes the total number of digits.
2. This is exactly the structure used to build the clean/corrupt pairs.
3. The model likely stores a high-level decision at =: the answer should begin with the new leading digit.
4. Removing the direction destroys that decision, so accuracy falls from 1.0000 to 0.1643.

### 7.5 Why inside propagation is almost unchanged
1. In inside_propagation, carries happen but the first answer digit usually does not depend on creating a new leading digit.
2. The model can solve these examples using other digit-local or distributed mechanisms.
3. The ablated direction is therefore not needed.
4. This explains the tiny drop from 0.9987 to 0.9982.

### 7.6 Why starts-with-1 no-carry is affected
1. starts_with_1_no_carry has no carry, so the effect cannot be explained by carry propagation.
2. The drop is still large: 0.4741.
3. This is evidence that the direction partly controls whether the first generated answer token is 1.
4. **This is also a warning against calling the direction a pure carry direction.**

## 8) Final claim
Overall, the experiment identifies a strong causal direction at the final-layer `=` token representation. Removing this direction has almost no effect on no-carry examples, but strongly hurts carry examples where the answer becomes longer than the inputs. **This suggests that the direction is meaningfully connected to carry-related behavior, especially cases where a carry creates a new leading digit. However, additional experiments show that the direction is not a pure carry-propagation feature: it also partially overlaps with the model’s tendency to predict the first answer token `1`.** Therefore, the best interpretation is that the direction captures a mixed feature related to length-increasing carry behavior and new-leading-`1` prediction, rather than a complete general arithmetic circuit.

## 9) Future work

The current analysis studies full residual-stream activations, so it does not identify the exact components responsible for the effect. A natural next step is to analyze individual attention heads and MLP layers to see whether the direction is localized or distributed. Future work should also separate carry propagation from the new-leading-`1` effect more carefully, and evaluate later answer tokens, not only the first generated digit.

