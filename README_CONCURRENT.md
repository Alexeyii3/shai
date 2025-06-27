# Concurrent Math Task Generator

This repository now includes two versions of the math task generator:

## Files

- `generate.py` - Original sequential version
- `generate_concurrent.py` - New concurrent version with async/await support

## Key Improvements in Concurrent Version

### 🚀 **Performance Benefits**
- **Parallel Processing**: Multiple skills processed simultaneously
- **Async API Calls**: Non-blocking Gemini API requests
- **Batch Processing**: Configurable concurrent worker limits
- **Rate Limiting**: Built-in semaphore to respect API limits

### 🔧 **Technical Features**
- **Thread-Safe**: Uses thread-local storage for API clients
- **Progress Tracking**: Real-time progress updates across workers
- **Error Resilience**: Per-skill error handling won't stop other workers
- **Memory Efficient**: Streaming responses with async processing

### ⚡ **Speed Comparison**
- **Sequential**: ~5-10 minutes per skill → Total: 50-100 minutes for 10 skills
- **Concurrent**: ~5-10 minutes per skill → Total: ~15-25 minutes for 10 skills (4 workers)
- **Speedup**: ~3-4x faster with proper rate limiting

## Usage

### Basic Usage
```bash
# Run concurrent version (recommended)
python generate_concurrent.py

# Run original version
python generate.py
```

### Configuration Options

In `generate_concurrent.py`, you can adjust:

```python
MAX_WORKERS = 4               # Number of concurrent workers
TASKS_PER_SKILL = 100         # Tasks to generate per skill
TEST_MODE = False             # Set to True for quick testing
REQUEST_DELAY = 1.0           # Delay between API requests (seconds)
```

### Worker Recommendations

- **Conservative**: 2-3 workers (safer for rate limits)
- **Balanced**: 4-5 workers (recommended default)
- **Aggressive**: 6-8 workers (may hit rate limits)

## Rate Limiting Strategy

The concurrent version implements multiple rate limiting mechanisms:

1. **Semaphore**: Limits concurrent API calls
2. **Request Delay**: Adds delay between requests
3. **Exponential Backoff**: Handles temporary failures
4. **Thread-Local Clients**: Prevents client conflicts

## Output

Both versions produce identical CSV files:
- Format: `generated_{skill_name}.csv`
- Same data structure and quality
- Thread-safe file writing in concurrent version

## Monitoring Progress

The concurrent version provides enhanced progress tracking:

```
📊 Progress: 3/10 skills completed
[Algebra_Linear_Equations] Completed generation: 100 tasks
[Geometry_Circles] Batch 2/4: Generating 25 tasks...
🚀 Processing batch 2/3 (4 skills)
```

## Error Handling

Improved error resilience:
- Individual skill failures don't stop other workers
- Detailed error reporting with skill context
- Automatic retry with exponential backoff
- Graceful degradation under high load

## Memory Usage

The concurrent version is more memory efficient:
- Streaming API responses
- Async processing reduces blocking
- Thread-local storage prevents memory leaks
- Batch processing controls peak memory usage

## When to Use Each Version

### Use `generate_concurrent.py` when:
- ✅ Processing multiple skills (>2)
- ✅ Want faster completion times
- ✅ Have stable internet connection
- ✅ Running on modern Python (3.7+)

### Use `generate.py` when:
- ✅ Processing single skill
- ✅ Conservative approach needed
- ✅ Debugging generation issues
- ✅ Limited system resources

## API Rate Limits

Both versions respect Gemini API limits, but concurrent version:
- Uses intelligent batching
- Implements request queuing
- Provides better error recovery
- Monitors token usage across workers

## Troubleshooting

### Common Issues

1. **Rate Limit Errors**
   - Reduce `MAX_WORKERS` to 2-3
   - Increase `REQUEST_DELAY` to 2.0
   - Enable `TEST_MODE` for testing

2. **Memory Issues**
   - Reduce batch size
   - Lower `MAX_WORKERS`
   - Process skills individually

3. **Network Timeouts**
   - Check internet connection
   - Increase retry delays
   - Use sequential version as fallback

### Performance Tuning

```python
# Conservative settings (slower but safer)
MAX_WORKERS = 2
REQUEST_DELAY = 2.0

# Aggressive settings (faster but may hit limits)
MAX_WORKERS = 6
REQUEST_DELAY = 0.5

# Balanced settings (recommended)
MAX_WORKERS = 4
REQUEST_DELAY = 1.0
```

## Dependencies

Same requirements for both versions:
- `google-genai>=0.3.0`
- `google-auth>=2.0.0`
- Python 3.7+ (for asyncio support)

Install with:
```bash
pip install -r requirements.txt
```

## Example Output

```
=== Concurrent Math Task Generator ===
Input file: SAT Questions - SAT_Math_No_Graph.csv
Max concurrent workers: 4

📊 Found 3 already generated skills
⏳ Will generate tasks for 7 skill(s) using 4 concurrent workers

🚀 Processing batch 1/2 (4 skills)
   Skills: Algebra_Linear_Equations, Geometry_Circles, Statistics_Mean, Calculus_Derivatives

[Algebra_Linear_Equations] Starting generation of 100 tasks
[Geometry_Circles] Starting generation of 100 tasks
...

✅ Batch 1 completed in 142.3s, generated 400 tasks
📊 Progress: 4/7 skills completed

🎉 All skills processed!
📊 Total time: 284.7s
📊 Total tasks generated: 700
📊 Average time per skill: 40.7s
```
