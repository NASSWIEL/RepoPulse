# Expanding-Window Training Implementation

## Overview

The training strategy has been modified to use an **expanding-window approach** instead of a fixed 4-quarter lookback window. This enables the model to learn from variable-length historical context, better reflecting real-world data availability over time.

## Key Changes

### 1. Configuration (`config/config.yaml`)

**Before:**
```yaml
preprocessing:
  sequence_length: 4  # Fixed number of past quarters
```

**After:**
```yaml
preprocessing:
  expanding_window: true      # Enable progressive expanding windows
  min_lookback: 2            # Minimum historical quarters required
  max_lookback: null         # Maximum lookback (null = unlimited)
```

### 2. Data Preparation (`models/prepare_timeseries_data.py`)

#### Expanding-Window Sequence Generation

The `_create_sequences()` method now generates training examples progressively:

- **Q1-Q2 → Q3**: Use first 2 quarters to predict 3rd quarter
- **Q1-Q2-Q3 → Q4**: Use first 3 quarters to predict 4th quarter  
- **Q1-Q2-Q3-Q4 → Q5**: Use first 4 quarters to predict 5th quarter
- And so on...

#### Variable-Length Sequence Handling

- Sequences are **left-padded** with zeros to a common maximum length
- Actual sequence lengths are stored in `sequence_lengths` array
- Models use packed sequences to ignore padding during training

**Key Features:**
- Tracks `actual_lookback` for each sequence
- Supports both expanding-window and fixed-window modes (backward compatible)
- Saves padded sequences with length information

### 3. Model Architecture (`models/forecaster.py`)

#### Updated LSTM/GRU Models

Both LSTM and GRU forecasters now support variable-length sequences:

```python
def fit(self, X, y, sequence_lengths=None):
    # Uses pack_padded_sequence for efficient variable-length processing
    packed_input = nn.utils.rnn.pack_padded_sequence(
        X_sorted, lengths, batch_first=True
    )
    packed_output, hidden = self.model(packed_input)
    # Use last hidden state (corresponds to actual last timestep)
    predictions = self.output_layer(hidden[-1])
```

**Benefits:**
- Efficiently handles padded sequences
- Only processes actual sequence content (not padding)
- Uses last hidden state corresponding to true sequence end

### 4. Training Scripts (`models/train_forecasters.py`)

#### TimeSeriesDataset

Updated to load and return sequence lengths:

```python
def __getitem__(self, idx):
    return {
        'lookback': ...,
        'target': ...,
        'length': self.sequence_lengths[idx]  # Added
    }
```

#### Model Forward Pass

Models now accept lengths parameter:

```python
def forward(self, x, lengths=None):
    if lengths is not None:
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.tolist(), batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.lstm(packed_input)
        return self.fc(hidden[-1])
    else:
        # Backward compatibility for fixed-length
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
```

### 5. API Service (`api_service.py`)

The API remains compatible:
- Accepts fixed-length inputs (e.g., 4 quarters) for inference
- Models trained with expanding windows can handle any sequence length
- No changes required to API endpoints

## Usage

### Training with Expanding Window (Default)

```bash
# Using config file (expanding_window: true)
python models/prepare_timeseries_data.py --config config/config.yaml
python models/train_forecasters.py
```

### Training with Fixed Window (Backward Compatible)

```bash
# Override config
python models/prepare_timeseries_data.py --fixed-window --lookback 4
```

### Command-Line Options

```bash
python models/prepare_timeseries_data.py \
    --config config/config.yaml \
    --expanding-window          # Enable expanding window
    --fixed-window              # Use fixed window (legacy)
    --lookback 4                # Override lookback (fixed mode only)
```

## Benefits of Expanding-Window Training

1. **More Training Data**: Generates multiple training examples per repository
   - Repository with 10 quarters: generates 8 examples (Q2→Q3, Q3→Q4, ..., Q9→Q10)
   - Fixed window: only 6 examples (with lookback=4)

2. **Better Generalization**: Model learns to handle varying amounts of history
   - Realistic for new repositories (limited history)
   - Adapts to established repositories (extensive history)

3. **Improved Learning**: Progressive training from simple to complex patterns
   - Early examples: 2-3 quarters (recent patterns)
   - Later examples: 4+ quarters (long-term trends)

4. **Flexible Inference**: Trained models work with any sequence length
   - Can predict for repos with 2 quarters
   - Can predict for repos with 10+ quarters

## Technical Details

### Sequence Padding Strategy

- **Left padding** (zeros at beginning): `[0, 0, ..., actual_data]`
- Preserves temporal order: most recent data at end
- RNN processes from left to right, padding doesn't affect final hidden state

### Packed Sequences

Using `pack_padded_sequence`:
- Skips padding computations (faster)
- Only processes actual sequence content
- Automatically handles variable lengths in batches

### Metadata Tracking

Saved in `data/processed/timeseries/metadata.json`:
```json
{
  "expanding_window": true,
  "min_lookback": 2,
  "max_lookback": null,
  "lookback": 10,  // Max length after padding
  "n_features": 8,
  "feature_names": [...]
}
```

## Backward Compatibility

The implementation maintains full backward compatibility:

1. **Fixed-window mode**: Set `expanding_window: false` in config
2. **Legacy scripts**: Work without modification
3. **Existing checkpoints**: Can still be loaded and used
4. **API service**: No changes needed for inference

## Performance Considerations

### Training Time
- **Increased**: More training examples per repository
- **Offset by**: More efficient packed sequence processing

### Memory Usage
- **Similar**: Padded sequences have fixed maximum length
- **Batch processing**: Unchanged

### Model Quality
- **Expected improvement**: More diverse training examples
- **Better handling**: Of repositories with limited history

## Example

### Repository Timeline
```
Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8  Q9  Q10
*---*---*---*---*---*---*---*---*---*
```

### Generated Training Examples (Expanding Window)
```
Example 1: [Q1, Q2] → Q3               (lookback=2)
Example 2: [Q1, Q2, Q3] → Q4           (lookback=3)
Example 3: [Q1, Q2, Q3, Q4] → Q5       (lookback=4)
Example 4: [Q1, Q2, Q3, Q4, Q5] → Q6   (lookback=5)
...
Example 8: [Q1...Q9] → Q10             (lookback=9)
```

### Generated Training Examples (Fixed Window, lookback=4)
```
Example 1: [Q1, Q2, Q3, Q4] → Q5
Example 2: [Q2, Q3, Q4, Q5] → Q6
Example 3: [Q3, Q4, Q5, Q6] → Q7
...
Example 6: [Q6, Q7, Q8, Q9] → Q10
```

**Result**: 8 examples vs 6 examples, with better coverage of varying historical contexts.

## Testing

To verify the implementation works correctly:

```bash
# 1. Prepare data with expanding window
python models/prepare_timeseries_data.py --expanding-window

# 2. Check generated sequences
python -c "
import numpy as np
data = np.load('data/processed/timeseries/train.npz')
print('Sequence shape:', data['lookback_features'].shape)
print('Lengths min/max:', data['sequence_lengths'].min(), data['sequence_lengths'].max())
"

# 3. Train model
python models/train_forecasters.py

# 4. Verify training completes without errors
```

## Future Enhancements

Potential improvements:
1. **Attention mechanisms**: Weight recent quarters more heavily
2. **Curriculum learning**: Start with short sequences, gradually increase
3. **Multi-horizon forecasting**: Predict multiple quarters ahead
4. **Adaptive padding**: Per-batch padding instead of global max
