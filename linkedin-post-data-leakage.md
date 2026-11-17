# LinkedIn Post — Data Leakage Fix in RNN/LSTM/GRU/BiLSTM Comparison

Been comparing SimpleRNN, LSTM, GRU, and BiLSTM on the same next-word-prediction task — and caught myself making a classic mistake: computing `max_seq_len` from train+val combined, and fitting the tokenizer before splitting. Both let validation quietly leak into preprocessing.

Fixed it: split first, dedupe exact-duplicate sentences, derive padding length from train data only, add dropout to tame overfitting (train/val gap went from ~34pts to ~11pts), and fix a subtle bug where `restore_best_weights` saves the best epoch's weights but `history[-1]` reports the last epoch's metrics — two different epochs, easy to miss.

Result: cleaner, trustworthy comparison. GRU came out on top — best val accuracy (16.1%) and lowest perplexity (459), at less than half LSTM's training time. Sometimes the "expensive" model isn't the best answer.

Small reminder that leakage isn't always dramatic — sometimes it's one line of preprocessing code deciding an architecture constant using data it shouldn't see yet.

#MachineLearning #DeepLearning #NLP #AI

---

## Before vs After (for reference / carousel slide)

**Before (leakage present):**

| Metric | RNN |
|---|---|
| Train acc | 49.4% (still climbing, unstopped) |
| Val acc (best) | 15.74% |
| Val loss (best) | 5.87 |
| Train/val gap | ~34 pts |

**After (leak-free, deduped, dropout added):**

| Metric | RNN | LSTM | GRU | BiLSTM |
|---|---|---|---|---|
| Best epoch | 13 | 23 | 14 | 15 |
| Train acc | 27.3% | 26.8% | 26.8% | 25.4% |
| Val acc | 15.1% | 16.0% | **16.1%** | 16.0% |
| Val loss | 6.18 | 6.58 | **6.13** | 6.29 |
| Val perplexity | 483 | 723 | **459** | 537 |
| Params | 1.79M | 1.88M | 1.85M | 2.98M |
| Train time | 48.7s | 126.6s | 78.7s | 124.2s |
