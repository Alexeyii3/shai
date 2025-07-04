# SAT Math Task Generator Documentation

## Overview

The `generate_concurrent.py` script is a sophisticated AI-powered tool that generates high-quality SAT math problems using Google's Gemini 2.5 Pro model through Vertex AI. It reads existing SAT problems from CSV files, analyzes them by skill and difficulty, and generates new problems that match the style, format, and difficulty distribution of the original dataset.

## Key Features

- **Concurrent Generation**: Processes multiple skills simultaneously with configurable worker limits
- **Difficulty-Based Distribution**: Generates equal numbers of Easy, Medium, and Hard problems
- **Smart Column Filtering**: Only loads necessary columns from input CSV to optimize memory usage
- **Robust Error Handling**: Comprehensive retry logic with exponential backoff
- **JSON Parsing Fallbacks**: Multiple strategies to handle Gemini's JSON responses
- **LaTeX Processing**: Automatic fixing of common LaTeX escaping issues in JSON
- **Progress Tracking**: Real-time monitoring of generation progress across skills
- **Resume Capability**: Can continue from where it left off by checking existing generated files

## Architecture

### Core Components

#### 1. ConcurrentMathTaskGenerator Class
The main class that orchestrates the entire generation process.

**Key Attributes:**
- `project_id`: Google Cloud project ID for Vertex AI
- `model`: Gemini model version (gemini-2.5-pro)
- `max_workers`: Number of concurrent workers (default: 1)
- `rate_limiter`: Async semaphore to control API request rate
- `request_delay`: Delay between API requests (1.5 seconds)

#### 2. Data Processing Pipeline

```
CSV Input → Column Filtering → Skill Grouping → Example Preparation → Prompt Generation → API Calls → JSON Parsing → Task Processing → CSV Output
```

### Input Data Structure

The script expects CSV files with the following **required columns**:
- `test`: Test type (e.g., "Math")
- `domain`: Subject domain (e.g., "Algebra")
- `skill`: Specific skill (e.g., "Linear equations in two variables")
- `difficulty`: Problem difficulty ("Easy", "Medium", "Hard")
- `question_text_latex`: Problem statement in LaTeX format
- `option_A_latex`, `option_B_latex`, `option_C_latex`, `option_D_latex`: Multiple choice options
- `correct_answer`: Correct answer (A, B, C, D, or direct value)
- `correct_answer_spr_latex`: Correct answer in LaTeX format

**Filtered Out Columns** (unused):
- `question_text_plain`: Plain text version (LaTeX version used instead)
- `option_A`, `option_B`, `option_C`, `option_D`: Plain text options (LaTeX versions used)
- `explanation_latex`: Original explanation (replaced with step_1 through step_6)

### Output Data Structure

Generated tasks include all input columns plus:
- `step_1` through `step_6`: Individual explanation steps in LaTeX format
- Each step is a complete, standalone explanation
- Up to 6 steps maximum (can be fewer, unused steps are null)

## Configuration

### Main Configuration Variables

```python
# In main() function
INPUT_CSV = "SAT Questions - SAT_Math_No_Graph.csv"  # Input file path
TASKS_PER_DIFFICULTY = 100        # Tasks to generate per difficulty level
TEST_MODE = False                 # Enable test mode (generates fewer tasks)
TEST_TASKS = 10                   # Number of tasks in test mode
ONLY_REMAINING = True             # Only generate for skills not yet completed
BATCH_SIZE = 10                   # Tasks per API batch
PROJECT_ID = "studyhall-dev-383420"  # Google Cloud project ID
MAX_WORKERS = 1                   # Concurrent workers
```

### Model Configuration

```python
# In __init__ method
self.model = "gemini-2.5-pro"     # Gemini model version
self.max_workers = 1              # Concurrent processing limit
self.rate_limiter = asyncio.Semaphore(1)  # API rate limiting
self.request_delay = 1.5          # Delay between requests (seconds)
```

## Key Methods

### Data Processing Methods

#### `read_tasks_by_skill(csv_file: str)`
- Reads and filters CSV input, keeping only necessary columns
- Groups tasks by skill for processing
- Reports column filtering and task counts
- Returns: `Dict[str, List[Dict[str, Any]]]`

#### `prepare_examples_for_prompt(tasks: List, num_examples: int = 15)`
- Randomly samples examples for prompt generation
- Cleans math content using `clean_math_content()`
- Ensures consistent formatting across examples
- Returns: `List[Dict[str, Any]]`

### Generation Methods

#### `create_generation_prompt(examples: List, skill: str, existing_generated: List = None, num_to_generate: int = 10)`
- Creates detailed prompts for Gemini API
- Includes examples, requirements, and formatting instructions
- Provides comprehensive LaTeX escaping guidelines
- Handles difficulty distribution requirements
- Returns: `str` (formatted prompt)

#### `generate_batch_with_backoff(prompt: str, skill: str, max_retries: int = 5, base_delay: float = 1.0)`
- Executes API calls with retry logic and exponential backoff
- Handles various error types (RetryError, TimeoutError, etc.)
- Tracks token usage for monitoring
- Returns: `Tuple[Optional[List[Dict]], int, int, int]` (tasks, input_tokens, output_tokens, total_tokens)

#### `generate_tasks_for_skill_by_difficulty(skill: str, examples: List, tasks_per_difficulty: int = 100, batch_size: int = 10)`
- Generates tasks with equal distribution across difficulty levels
- Groups examples by difficulty for targeted generation
- Processes each difficulty level separately
- Returns: `List[Dict[str, Any]]`

### Processing Methods

#### `fix_common_latex_escaping(json_text: str)`
- Fixes common LaTeX escaping issues in JSON responses
- Handles arrow symbols (→, ⇒, etc.) and math commands
- Uses regex to fix single backslashes in JSON strings
- Applied before JSON parsing to prevent errors
- Returns: `str` (corrected JSON text)

#### `parse_json_with_fallback(json_text: str)`
- Multiple parsing strategies for robust JSON handling
- Attempts direct parsing, regex extraction, and error correction
- Handles mixed content (text + JSON) responses
- Returns: `List[Dict[str, Any]]`

#### `process_generated_tasks(tasks: List)`
- Validates and cleans generated task data
- Ensures required fields are present
- Standardizes format and content
- Returns: `List[Dict[str, Any]]`

### File Management Methods

#### `save_tasks_to_csv(tasks: List, filename: str)`
- Saves generated tasks to CSV with proper column ordering
- Handles field validation and error reporting
- Thread-safe file writing with locks
- Creates detailed error reports for debugging

#### `check_existing_tasks_by_difficulty(skill: str)`
- Counts existing tasks by difficulty level
- Determines how many additional tasks needed
- Supports resume functionality
- Returns: `Dict[str, int]` (difficulty → tasks_needed)

## Usage Examples

### Basic Usage

```python
# Initialize generator
generator = ConcurrentMathTaskGenerator(
    project_id="your-project-id",
    max_workers=1
)

# Generate tasks for all skills
await generator.process_all_skills(
    input_csv="SAT Questions - SAT_Math_No_Graph.csv",
    tasks_per_difficulty=100,
    test_mode=False,
    only_remaining=True
)
```

### Test Mode

```python
# Quick test with fewer tasks
await generator.process_all_skills(
    input_csv="SAT Questions - SAT_Math_No_Graph.csv",
    tasks_per_difficulty=100,
    test_mode=True,
    test_tasks=5,
    only_remaining=False
)
```

### Custom Configuration

```python
# Custom batch processing
generator = ConcurrentMathTaskGenerator(
    project_id="your-project-id",
    max_workers=2  # More concurrent workers
)

# Process with custom batch size
await generator.process_all_skills(
    input_csv="your_input.csv",
    tasks_per_difficulty=50,
    batch_size=5,  # Smaller batches
    only_remaining=True
)
```

## Error Handling

### Common Error Types

1. **JSON Parsing Errors**
   - Invalid LaTeX escaping in JSON
   - Mixed content responses
   - Malformed JSON structure

2. **API Errors**
   - RetryError: Network/connection issues
   - TimeoutError: Request timeout
   - ResourceExhausted: Quota exceeded
   - ClientError: Authentication/permission issues

3. **Data Validation Errors**
   - Missing required fields
   - Invalid task structure
   - Empty response arrays

### Error Recovery Strategies

- **Exponential Backoff**: Increasing delays between retries
- **Client Reset**: Reinitializes API client on persistent errors
- **Fallback Parsing**: Multiple JSON parsing strategies
- **Graceful Degradation**: Continues processing other skills on individual failures

## Output Files

### Generated CSV Structure

```
generated_{skill_name}.csv
```

**Columns:**
- All original input columns (test, domain, skill, difficulty, etc.)
- `step_1` through `step_6`: Individual explanation steps
- Proper LaTeX formatting throughout

### Example Output

```csv
test,domain,skill,difficulty,question_text_latex,option_A_latex,option_B_latex,option_C_latex,option_D_latex,correct_answer,correct_answer_spr_latex,step_1,step_2,step_3,step_4,step_5,step_6
Math,Algebra,Linear equations in two variables,Medium,"Find the slope of $3x + 4y = 12$","$-\frac{3}{4}$","$\frac{3}{4}$","$-\frac{4}{3}$","$\frac{4}{3}$",A,"","Convert to slope-intercept form $y = mx + b$","Subtract $3x$ from both sides: $4y = -3x + 12$","Divide by 4: $y = -\frac{3}{4}x + 3$","The slope is the coefficient of $x$","Therefore, the slope is $-\frac{3}{4}$",null
```

## Performance Considerations

### Optimization Settings

- **Batch Size**: 10 tasks per API call (balance between efficiency and reliability)
- **Concurrent Workers**: 1 (prevents API rate limiting)
- **Request Delay**: 1.5 seconds between calls
- **Memory Optimization**: Column filtering reduces memory usage by ~60%

### Token Usage

- **Input Tokens**: ~3,000-4,000 per batch (examples + prompt)
- **Output Tokens**: ~4,000-6,000 per batch (generated tasks)
- **Total**: ~7,000-10,000 tokens per batch of 10 tasks

### Generation Speed

- **Per Task**: ~30-45 seconds (including retries and delays)
- **Per Skill**: 50-75 minutes for 300 tasks (100 per difficulty)
- **Full Dataset**: 12-18 hours for ~20 skills

## Monitoring and Debugging

### Progress Tracking

```
[Linear equations in two variables] Starting generation of 100 tasks per difficulty level
[Linear equations in two variables] Available difficulties: ['Easy', 'Medium', 'Hard']
[Linear equations in two variables] Easy: 45 examples
[Linear equations in two variables] Medium: 38 examples  
[Linear equations in two variables] Hard: 22 examples

[Linear equations in two variables] Generating 100 tasks for difficulty: Easy
[Linear equations in two variables] Easy - Batch 1/10: Generating 10 tasks...
  [Linear equations in two variables-Easy] Token usage - Input: 3615, Output: 4373, Total: 7988
  [Linear equations in two variables-Easy] Successfully generated 10 tasks
```

### Error Reporting

```
❌ CSV Writing Error: Extra fields detected in tasks for generated_Linear_functions.csv
🔍 DEBUGGING: Full content of problematic tasks:
📋 PROBLEMATIC TASK #1:
   Extra fields detected: ['invalid_field']
   FULL TASK CONTENT:
   🚨 EXTRA → invalid_field: "unexpected content"
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```bash
   gcloud auth application-default login
   gcloud config set project your-project-id
   ```

2. **Memory Issues**
   - Reduce batch_size to 5
   - Process skills individually
   - Use test_mode for debugging

3. **Rate Limiting**
   - Increase request_delay to 2.0+
   - Reduce max_workers to 1
   - Check Vertex AI quotas

4. **JSON Parsing Failures**
   - Check Gemini response format
   - Verify LaTeX escaping
   - Enable debug output

### Debug Mode

```python
# Add debug output to parse_json_with_fallback
except json.JSONDecodeError as e:
    print(f"  Direct JSON parsing failed: {str(e)[:100]}")
    print(f"  JSON text: {json_text}")  # Enable this for debugging
```

## Best Practices

1. **Start Small**: Use test_mode=True for initial runs
2. **Monitor Progress**: Check generated files regularly
3. **Resume Capability**: Use only_remaining=True to continue interrupted runs
4. **Resource Management**: Monitor API quotas and costs
5. **Quality Control**: Review generated tasks for accuracy and consistency

## Dependencies

```python
# Required packages
import asyncio
import csv
import json
import random
import threading
import time
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

# Google Cloud dependencies
import google.genai as genai
from google.genai import types
```

## Installation

```bash
# Install required packages
pip install google-genai

# Authenticate with Google Cloud
gcloud auth application-default login
gcloud config set project your-project-id
```

This documentation provides a comprehensive guide to understanding, configuring, and using the SAT math task generator effectively. 