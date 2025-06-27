# Math Task Generator Documentation

## Overview

The `generate_concurrent.py` script is an AI-powered math task generator that uses Google's Gemini API to create high-quality math problems for standardized tests (SAT, PSAT). It reads existing math problems from CSV files and generates new, similar problems while maintaining consistency in format, difficulty, and mathematical accuracy.

## Features

- **AI-Powered Generation**: Uses Gemini 2.5 Pro for high-quality math problem creation
- **Concurrent Processing**: Supports multiple skills being processed simultaneously
- **Difficulty-Based Generation**: Generates problems across Easy, Medium, and Hard difficulty levels
- **LaTeX Support**: Properly handles mathematical notation with LaTeX/MathJax formatting
- **Robust Error Handling**: Comprehensive error handling with retry mechanisms
- **Progress Tracking**: Real-time progress monitoring and token usage tracking
- **CSV Integration**: Reads from and writes to CSV files for easy data management
- **Memory Optimization**: Only loads necessary columns to reduce memory usage

## Prerequisites

### 1. Google Cloud Setup

1. **Create a Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one
   - Note your project ID (e.g., "studyhall-dev-383420")

2. **Enable Vertex AI API**:
   ```bash
   gcloud services enable aiplatform.googleapis.com
   ```

3. **Set up Authentication**:
   ```bash
   # Install gcloud CLI if not already installed
   # Then authenticate
   gcloud auth application-default login
   ```

### 2. Python Dependencies

Install required packages:
```bash
pip install google-cloud-aiplatform
pip install asyncio
pip install csv
pip install json
pip install random
pip install threading
```

## File Structure

```
SAT_Gemini/
├── sat/
│   ├── generate_concurrent.py
│   ├── SAT Questions - SAT_Math_No_Graph.csv
│   └── generated_*.csv (output files)
├── psat/
│   ├── generate_concurrent.py
│   └── input CSV files
├── sat_with_images/
│   ├── generate_concurrent.py
│   └── input CSV files
└── README_generate_concurrent.md
```

## Input CSV Format

The input CSV file must contain the following columns:

### Required Columns:
- `test`: Test type (e.g., "Math", "SAT")
- `domain`: Subject domain (e.g., "Algebra", "Geometry")
- `skill`: Specific skill (e.g., "Linear equations in two variables")
- `difficulty`: Problem difficulty ("Easy", "Medium", "Hard")
- `question_text_latex`: Question text with LaTeX formatting
- `option_A_latex`, `option_B_latex`, `option_C_latex`, `option_D_latex`: Multiple choice options
- `correct_answer`: Correct answer (A, B, C, D, or direct numerical answer)
- `correct_answer_spr_latex`: Correct answer in LaTeX format

### Unused Columns (Filtered Out):
- `question_text_plain`: Plain text version (not used)
- `option_A`, `option_B`, `option_C`, `option_D`: Plain text options (not used)
- `explanation_latex`: Original explanation (replaced with step_1 through step_6)

## Usage

### Basic Usage

1. **Navigate to the appropriate directory**:
   ```bash
   cd sat/  # or psat/ or sat_with_images/
   ```

2. **Run the generator**:
   ```bash
   python generate_concurrent.py
   ```

### Configuration Options

Edit the configuration section in `generate_concurrent.py`:

```python
# Configuration
INPUT_CSV = "SAT Questions - SAT_Math_No_Graph.csv"  # Input file path
TASKS_PER_DIFFICULTY = 100    # Tasks to generate per difficulty level
TEST_MODE = False             # Set to True for testing with fewer tasks
TEST_TASKS = 10               # Number of tasks in test mode
PROJECT_ID = "studyhall-dev-383420"  # Your Google Cloud project ID
MAX_WORKERS = 1               # Number of concurrent workers
```

### Test Mode

For testing or development, enable test mode:

```python
TEST_MODE = True
TEST_TASKS = 5  # Generate only 5 tasks per skill for testing
```

## Output Format

Generated tasks are saved as CSV files with the naming pattern:
```
generated_{skill_name}.csv
```

### Output Columns:
- `test`, `domain`, `skill`, `difficulty`: Same as input
- `question_text_latex`: Generated question with LaTeX
- `option_A_latex` through `option_D_latex`: Generated multiple choice options
- `correct_answer`: Correct answer identifier
- `correct_answer_spr_latex`: Correct answer in LaTeX format
- `step_1` through `step_6`: Individual solution steps (up to 6 steps)

## Advanced Features

### Difficulty-Based Generation

The generator ensures equal distribution across difficulty levels:
- **Easy**: Fundamental concepts, straightforward calculations
- **Medium**: Multi-step problems, moderate complexity
- **Hard**: Complex scenarios, advanced mathematical reasoning

### Incremental Generation

The script intelligently handles existing files:
- Reads existing generated tasks
- Only generates additional tasks if needed
- Maintains target of 100 tasks per difficulty level per skill

### Error Handling and Retries

Comprehensive error handling includes:
- **JSON Parsing Errors**: Multiple fallback parsing strategies
- **API Rate Limits**: Exponential backoff with jitter
- **Network Issues**: Automatic retries with increasing delays
- **LaTeX Escaping**: Automatic fixing of common LaTeX formatting issues

## Monitoring and Debugging

### Progress Tracking

The script provides real-time feedback:
```
[Linear functions] Starting generation of 100 tasks per difficulty level
[Linear functions] Available difficulties: ['Easy', 'Medium', 'Hard']
[Linear functions] Easy: 15 examples
[Linear functions] Medium: 12 examples  
[Linear functions] Hard: 8 examples
```

### Token Usage Monitoring

Track API usage and costs:
```
[Linear functions] Token usage - Input: 3615, Output: 4373, Total: 15051
```

### Error Analysis

Detailed error reporting helps with troubleshooting:
```
[Skill] JSON parsing error: Invalid \escape: line 7 column 127
[Skill] Raw response (first 500 chars): [{"test": "Math"...
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   ```bash
   # Re-authenticate with Google Cloud
   gcloud auth application-default login
   ```

2. **API Quota Exceeded**:
   - Reduce `MAX_WORKERS` to 1
   - Increase delays between requests
   - Check quota limits in Google Cloud Console

3. **JSON Parsing Errors**:
   - The script automatically fixes common LaTeX escaping issues
   - Check the raw response output for malformed JSON
   - Consider reducing batch size if issues persist

4. **Memory Issues**:
   - The script filters unused columns automatically
   - Reduce batch size if processing very large files
   - Consider processing skills individually

### Performance Optimization

1. **Adjust Concurrency**:
   ```python
   MAX_WORKERS = 1  # Start with 1, increase gradually if stable
   ```

2. **Optimize Batch Sizes**:
   ```python
   batch_size = 10  # Smaller batches = more stable, larger = faster
   ```

3. **Model Selection**:
   ```python
   self.model = "gemini-2.5-pro"    # Higher quality, slower
   self.model = "gemini-2.5-flash"  # Faster, lower cost
   ```

## Cost Considerations

### Token Usage

- **Input tokens**: Depend on example problems and prompts (~3000-5000 per batch)
- **Output tokens**: Depend on generated content (~4000-6000 per batch)
- **Total cost**: Approximately $0.01-0.05 per batch of 10 problems

### Optimization Tips

1. Use test mode for development
2. Start with fewer examples in prompts
3. Monitor token usage and adjust batch sizes
4. Consider using Gemini Flash for cost savings

## File Variants

### SAT Math (No Graphics)
- **File**: `sat/generate_concurrent.py`
- **Input**: Text-based math problems
- **Focus**: Algebra, geometry, statistics

### PSAT Math
- **File**: `psat/generate_concurrent.py`
- **Input**: PSAT-level problems
- **Configuration**: Reduced timeout (300s), fewer workers (2)

### SAT Math with Images
- **File**: `sat_with_images/generate_concurrent.py`
- **Input**: Problems with figure descriptions
- **Additional column**: `figure_description`

## Example Workflow

1. **Setup**:
   ```bash
   cd sat/
   # Verify input CSV exists
   ls "SAT Questions - SAT_Math_No_Graph.csv"
   ```

2. **Test Run**:
   ```python
   # Edit generate_concurrent.py
   TEST_MODE = True
   TEST_TASKS = 3
   ```
   ```bash
   python generate_concurrent.py
   ```

3. **Full Generation**:
   ```python
   # Edit generate_concurrent.py
   TEST_MODE = False
   TASKS_PER_DIFFICULTY = 100
   ```
   ```bash
   python generate_concurrent.py
   ```

4. **Monitor Output**:
   ```bash
   ls generated_*.csv
   # Check file contents
   head -5 generated_Linear_functions.csv
   ```

## Support and Maintenance

### Logs and Debugging

The script provides extensive logging. Key information includes:
- Skill processing progress
- Token usage statistics
- Error details and retry attempts
- Final completion status

### Regular Maintenance

1. **Monitor API quotas** in Google Cloud Console
2. **Review generated content** for quality
3. **Update prompts** as needed for better results
4. **Archive old generated files** to manage disk space

## Version History

- **v1.0**: Initial implementation with basic generation
- **v1.1**: Added difficulty-based generation
- **v1.2**: Improved error handling and retry logic
- **v1.3**: Added LaTeX escaping fixes and memory optimization
- **v1.4**: Enhanced concurrent processing and progress tracking 