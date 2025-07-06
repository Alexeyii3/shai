import csv
import os
import json
import random
import time
import asyncio
import re
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
import glob

# Ensure proper error handling for imports
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("\n" + "="*80)
    print("❌ Error: Required Google AI packages not found.")
    print("\nPlease install the required packages using:")
    print("\n    pip install google-cloud-aiplatform")
    print("\nIf you continue to see this error after installation, ensure you have:")
    print("1. Properly set up Google Cloud credentials")
    print("2. Run 'gcloud auth application-default login'")
    print("3. Enabled the Vertex AI API in your Google Cloud project")
    print("\nFor more information, visit: https://cloud.google.com/vertex-ai/docs/start/client-libraries")
    print("="*80 + "\n")
    sys.exit(1)


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup comprehensive logging with file output"""
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup main logger
    logger = logging.getLogger('SAT_Images_Generator')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler for detailed logs (all messages)
    detailed_file = Path(log_dir) / f"generation_detailed_{timestamp}.log"
    detailed_handler = logging.FileHandler(detailed_file, encoding='utf-8')
    detailed_handler.setLevel(logging.DEBUG)
    detailed_handler.setFormatter(detailed_formatter)
    logger.addHandler(detailed_handler)
    
    # File handler for errors only
    error_file = Path(log_dir) / f"generation_errors_{timestamp}.log"
    error_handler = logging.FileHandler(error_file, encoding='utf-8')
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    # File handler for progress tracking (INFO and WARNING)
    progress_file = Path(log_dir) / f"generation_progress_{timestamp}.log"
    progress_handler = logging.FileHandler(progress_file, encoding='utf-8')
    progress_handler.setLevel(logging.INFO)
    progress_handler.setFormatter(detailed_formatter)
    logger.addHandler(progress_handler)
    
    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Log initialization
    logger.info("=" * 80)
    logger.info("SAT Math Task Generator (with Images) - Logging Initialized")
    logger.info(f"Detailed logs: {detailed_file}")
    logger.info(f"Error logs: {error_file}")
    logger.info(f"Progress logs: {progress_file}")
    logger.info("=" * 80)
    
    return logger


class ConcurrentMathTaskGenerator:
    def __init__(self, project_id: str = "studyhall-dev-383420", max_workers: int = 2):
        """Initialize the Gemini API client with Vertex AI and concurrency settings"""
        self.project_id = project_id
        self.location = "global"
        self.model = "gemini-2.5-flash"
        self.max_workers = max_workers  # Default reduced to 3 to avoid rate limits
        
        # Setup logging
        self.logger = setup_logging()
        self.logger.info(f"Initializing SAT Math Task Generator (with Images)")
        self.logger.info(f"Project ID: {project_id}")
        self.logger.info(f"Model: {self.model}")
        self.logger.info(f"Max workers: {max_workers}")
        
        # Thread-local storage for API clients (since Gemini client might not be thread-safe)
        self._local = threading.local()
        
        # Rate limiting with more conservative defaults
        self.rate_limiter = asyncio.Semaphore(max(1, max_workers))
        self.request_delay = 3.0  # Increased delay between requests to avoid hitting rate limits
        
        # Thread pool for API calls
        self.executor = ThreadPoolExecutor(max_workers=max(1, max_workers * 2), 
                                          thread_name_prefix="gemini_")
        
        # Session management for API stability
        self._session_reset_counter = 0
        self._max_requests_per_session = 30  # Reset client after this many requests
        
        # Progress tracking
        self.total_skills = 0
        self.completed_skills = 0
        self.lock = threading.Lock()
        
        # Error tracking
        self.error_counters = defaultdict(int)
    
    def get_client(self, force_reset=False):
        """Get or create a thread-local Gemini client with session management"""
        thread_id = threading.get_ident()
        
        # Force reset or create new client if needed
        if force_reset or not hasattr(self._local, 'client') or not hasattr(self._local, 'request_counter'):
            try:
                # Close any existing client if applicable (for future compatibility)
                if hasattr(self._local, 'client') and hasattr(self._local.client, 'close'):
                    self._local.client.close()
            except Exception:
                pass  # Ignore errors when closing
                
            # Create a fresh client
            self._local.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
            )
            self._local.request_counter = 0
            self.logger.debug(f"[Thread {thread_id % 1000}] Created new Gemini API client")
        
        # Increment counter
        self._local.request_counter += 1
        
        # Check if we should reset the client after a number of requests
        if self._local.request_counter >= self._max_requests_per_session:
            self.logger.debug(f"[Thread {thread_id % 1000}] Resetting API client after {self._local.request_counter} requests")
            return self.get_client(force_reset=True)
            
        return self._local.client
    
    def read_tasks_by_skill(self, csv_file: str) -> Dict[str, List[Dict[str, Any]]]:
        """Read tasks from CSV and group them by skill, keeping only needed columns"""
        tasks_by_skill = defaultdict(list)
        
        # Define the columns we actually need for generation
        needed_columns = {
            'test', 'domain', 'skill', 'difficulty', 'question_text_latex',
            'option_A_latex', 'option_B_latex', 'option_C_latex', 'option_D_latex',
            'correct_answer', 'correct_answer_spr_latex', 'figure_description'
        }
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                # Check which needed columns are available in the CSV
                available_columns = set(reader.fieldnames or [])
                missing_columns = needed_columns - available_columns
                unused_columns = available_columns - needed_columns
                
                if missing_columns:
                    self.logger.warning(f"Missing expected columns: {sorted(missing_columns)}")
                
                if unused_columns:
                    self.logger.info(f"Filtering out unused columns: {sorted(unused_columns)}")
                
                for row in reader:
                    skill = row.get('skill', '').strip()
                    if skill:
                        # Create a filtered row with only the needed columns
                        filtered_row = {col: row.get(col, '') for col in needed_columns if col in available_columns}
                        tasks_by_skill[skill].append(filtered_row)
            
            self.logger.info(f"Successfully read tasks from {csv_file}")
            for skill, tasks in tasks_by_skill.items():
                self.logger.info(f"Skill loaded - {skill}: {len(tasks)} tasks")
                
        except FileNotFoundError:
            self.logger.error(f"CSV file {csv_file} not found")
        except Exception as e:
            self.logger.error(f"Error reading CSV file: {e}")
        
        return dict(tasks_by_skill)
    
    def clean_math_content(self, content: str) -> str:
        """Clean math content to use proper MathJax syntax"""
        if not content:
            return content
            
        # Fix common issues
        content = content.replace('[Math: ', '$').replace(']', '$')
        content = content.replace('\\left(', '(').replace('\\right)', ')')
        content = content.replace('\\left[', '[').replace('\\right]', ']')
        
        # Ensure proper LaTeX formatting
        content = content.replace('$$$', '$').replace('$$', '$')
        
        return content.strip()
    
    def fix_common_latex_escaping(self, json_text: str) -> str:
        """Fix only the most common LaTeX escaping issues in JSON strings"""
        import re
        
        # Define only the most problematic LaTeX patterns that commonly break JSON
        critical_fixes = [
            # Common arrow symbols that break JSON
            (r'\\Rightarrow', r'\\\\Rightarrow'),
            (r'\\Leftarrow', r'\\\\Leftarrow'),
            (r'\\rightarrow', r'\\\\rightarrow'),
            (r'\\leftarrow', r'\\\\leftarrow'),
            
            # Common LaTeX commands that often appear unescaped
            (r'\\frac\{', r'\\\\frac{'),
            (r'\\sqrt\{', r'\\\\sqrt{'),
            (r'\\cdot', r'\\\\cdot'),
            (r'\\times', r'\\\\times'),
            
            # Fix single backslashes that are NOT already part of valid JSON escapes
            # This regex looks for backslashes not followed by valid JSON escape characters or already doubled
            (r'(?<!\\)\\(?![\\"/bfnrtu])', r'\\\\'),
        ]
        
        # Apply fixes only within JSON string values (between quotes)
        def fix_in_string(match):
            # The content of the string, without the quotes
            content = match.group(1)
            for pattern, replacement in critical_fixes:
                content = re.sub(pattern, replacement, content)
            # Return the fixed content, re-wrapped in quotes
            return f'"{content}"'
        
        # This regex is a bit simple and can fail on escaped quotes. A better one is below.
        # We will fix the one in the more comprehensive function.
        # This function is now redundant, but let's fix it for completeness.
        json_text = re.sub(r'"((?:\\"|[^"])*)"', fix_in_string, json_text)
        
        return json_text
    
    def fix_json_escaping(self, json_text: str) -> str:
        """Fix common JSON escaping issues with LaTeX content"""
        import re
        
        # First, let's try a more comprehensive approach to fix LaTeX escaping
        # Common LaTeX sequences that need proper escaping in JSON
        latex_replacements = [
            # Fix common LaTeX commands that aren't properly escaped
            (r'\\frac\{', r'\\\\frac{'),
            (r'\\sqrt\{', r'\\\\sqrt{'),
            (r'\\left\(', r'\\\\left('),
            (r'\\right\)', r'\\\\right)'),
            (r'\\left\[', r'\\\\left['),
            (r'\\right\]', r'\\\\right]'),
            (r'\\begin\{', r'\\\\begin{'),
            (r'\\end\{', r'\\\\end{'),
            (r'\\cdot', r'\\\\cdot'),
            (r'\\times', r'\\\\times'),
            (r'\\div', r'\\\\div'),
            (r'\\pm', r'\\\\pm'),
            (r'\\mp', r'\\\\mp'),
            (r'\\leq', r'\\\\leq'),
            (r'\\geq', r'\\\\geq'),
            (r'\\neq', r'\\\\neq'),
            (r'\\approx', r'\\\\approx'),
            (r'\\infty', r'\\\\infty'),
            (r'\\alpha', r'\\\\alpha'),
            (r'\\beta', r'\\\\beta'),
            (r'\\gamma', r'\\\\gamma'),
            (r'\\delta', r'\\\\delta'),
            (r'\\theta', r'\\\\theta'),
            (r'\\pi', r'\\\\pi'),
            (r'\\sigma', r'\\\\sigma'),
            (r'\\omega', r'\\\\omega'),
            (r'\\hline', r'\\\\hline'),
            (r'(?<!\\)\\\\(?!\\)', r'\\\\\\\\'),  # Fix double backslashes in tables
        ]
        
        # Apply LaTeX-specific fixes within JSON string values
        def fix_latex_in_string(match):
            content = match.group(1)
            
            # Apply LaTeX replacements
            for pattern, replacement in latex_replacements:
                content = re.sub(pattern, replacement, content)
            
            # Fix any remaining single backslashes that aren't part of valid JSON escapes
            # This regex looks for backslashes not followed by valid JSON escape characters
            content = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', content)
            
            return f'"{content}"'
        
        # Apply fixes to content within JSON strings
        json_text = re.sub(r'"((?:\\"|[^"])*)"', fix_latex_in_string, json_text)
        
        return json_text
    
    def parse_json_with_fallback(self, json_text: str) -> List[Dict[str, Any]]:
        """Parse JSON with multiple fallback strategies, handling text before JSON"""
        
        # Try 1: Direct parsing
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            self.logger.debug(f"Direct JSON parsing failed: {str(e)[:100]}")
            self.logger.debug(f"JSON text: {json_text}")
        
        # Try 2: Extract JSON array from mixed content (improved regex)
        try:
            import re
            
            # Look for JSON array that starts with [ and ends with ]
            # This handles cases where there's explanatory text before/after JSON
            patterns = [
                # Pattern 1: Find complete JSON array (greedy match)
                r'\[\s*\{.*?\}\s*\]',
                # Pattern 2: Find JSON array with proper nesting
                r'\[(?:[^[\]{}]|\{[^{}]*\}|\[[^\]]*\])*\]',
                # Pattern 3: Find JSON starting with [ and ending with ] (non-greedy)
                r'\[.*?\]',
                # Pattern 4: More specific pattern for task arrays
                r'\[\s*\{\s*"test".*?\}\s*\]',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, json_text, re.DOTALL)
                if matches:
                    # Try to parse the first match
                    for match in matches:
                        try:
                            parsed = json.loads(match)
                            if isinstance(parsed, list) and len(parsed) > 0:
                                self.logger.debug(f"Successfully extracted JSON using pattern: {pattern}")
                                return parsed
                        except json.JSONDecodeError:
                            continue
            
        except Exception as e:
            self.logger.debug(f"Regex extraction failed: {str(e)[:100]}")
        
        # Try 3: Apply LaTeX escaping fixes and try again
        try:
            fixed_json = self.fix_json_escaping(json_text)
            
            # Try direct parsing of fixed JSON
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                pass
            
            # Try regex extraction on fixed JSON
            import re
            patterns = [
                r'\[\s*\{.*?\}\s*\]',
                r'\[(?:[^[\]{}]|\{[^{}]*\}|\[[^\]]*\])*\]',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, fixed_json, re.DOTALL)
                if matches:
                    for match in matches:
                        try:
                            parsed = json.loads(match)
                            if isinstance(parsed, list) and len(parsed) > 0:
                                return parsed
                        except json.JSONDecodeError:
                            continue
            
        except Exception as e:
            self.logger.debug(f"Fallback JSON parsing failed: {str(e)[:100]}")
        
        # Try 4: Index-based extraction (find first [ and last ])
        try:
            start_idx = json_text.find('[')
            end_idx = json_text.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                extracted = json_text[start_idx:end_idx + 1]
                
                # Apply escaping fixes
                extracted = self.fix_json_escaping(extracted)
                
                parsed = json.loads(extracted)
                if isinstance(parsed, list):
                    self.logger.debug(f"Successfully extracted JSON using index-based extraction")
                    return parsed
            
        except Exception as e:
            self.logger.debug(f"Index-based extraction failed: {str(e)[:100]}")
        
        # Try 5: Line-by-line parsing to handle malformed JSON
        try:
            lines = json_text.split('\n')
            json_lines = []
            in_json = False
            
            for line in lines:
                if '[' in line and not in_json:
                    in_json = True
                if in_json:
                    json_lines.append(line)
                if ']' in line and in_json:
                    break
            
            if json_lines:
                extracted = '\n'.join(json_lines)
                extracted = self.fix_json_escaping(extracted)
                parsed = json.loads(extracted)
                if isinstance(parsed, list):
                    self.logger.debug(f"Successfully extracted JSON using line-by-line parsing")
                    return parsed
            
        except Exception as e:
            self.logger.debug(f"Line-by-line extraction failed: {str(e)[:100]}")
        
        # If all methods fail, raise an exception
        self.logger.error("=== PARSE JSON FALLBACK FAILURE ===")
        self.logger.error(f"All JSON parsing methods failed for response:")
        self.logger.error(f"Response length: {len(json_text)} characters")
        self.logger.error(f"Full response content:")
        self.logger.error("-" * 80)
        self.logger.error(json_text)
        self.logger.error("-" * 80)
        
        # Also log first and last 200 characters for quick scanning
        if len(json_text) > 400:
            self.logger.error(f"Response preview (first 200 chars): {json_text[:200]}...")
            self.logger.error(f"Response preview (last 200 chars): ...{json_text[-200:]}")
        
        raise json.JSONDecodeError("All JSON parsing methods failed", json_text, 0)
    
    def prepare_examples_for_prompt(self, tasks: List[Dict[str, Any]], num_examples: int = 15) -> List[Dict[str, Any]]:
        """Prepare a subset of examples for the prompt, ensuring variety"""
        if len(tasks) <= num_examples:
            return tasks
        
        # Try to get a good mix of difficulties if available
        difficulties = {}
        for task in tasks:
            diff = task.get('difficulty', 'Unknown')
            if diff not in difficulties:
                difficulties[diff] = []
            difficulties[diff].append(task)
        
        # If we have multiple difficulties, try to get examples from each
        if len(difficulties) > 1:
            examples_per_difficulty = num_examples // len(difficulties)
            selected = []
            for diff_tasks in difficulties.values():
                selected.extend(random.sample(diff_tasks, min(examples_per_difficulty, len(diff_tasks))))
            # Fill remaining slots randomly
            remaining = num_examples - len(selected)
            if remaining > 0:
                available = [t for t in tasks if t not in selected]
                selected.extend(random.sample(available, min(remaining, len(available))))
            return selected[:num_examples]
        return random.sample(tasks, num_examples)

    def extract_question_texts_only(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Extract only the question text from tasks for prompt optimization"""
        question_texts = []
        for task in tasks:
            question_text = task.get('question_text_latex', '')
            if question_text:
                # Clean up the question text for better readability in prompts
                cleaned_text = question_text.strip()
                # Remove excessive whitespace
                cleaned_text = ' '.join(cleaned_text.split())
                question_texts.append(cleaned_text)
        return question_texts
    
    def create_generation_prompt(self, examples: List[Dict[str, Any]], skill: str, 
                                existing_generated: List[Dict[str, Any]] = None, num_to_generate: int = 10) -> str:
        """Create a detailed prompt for generating math tasks with figure descriptions"""
        
        test_name = examples[0].get('test', 'SAT') if examples else 'SAT'
        domain = examples[0].get('domain', 'Math') if examples else 'Math'
        
        prompt = f"""You are an expert math educator creating high-quality {skill} problems with detailed figure descriptions. 

TASK: Generate {num_to_generate} new math problems for the skill "{skill}" that match the style and structure of the provided examples.

EXAMPLES FROM EXISTING DATASET:
{json.dumps(examples, indent=2)}

REQUIREMENTS:
1. **Maintain Consistency**: Follow the exact same format and structure as the examples
2. **Mathematical Accuracy**: Ensure all calculations are correct
3. **Varied Difficulty**: Include a mix of Easy, Medium, and Hard problems
4. **Complete Structure**: Include all required fields
5. **Proper LaTeX**: Use correct LaTeX formatting for mathematical expressions
6. **ULTRA-DETAILED FIGURE DESCRIPTIONS**: Every problem must include an extremely detailed figure description"""

        if existing_generated:
            question_texts = self.extract_question_texts_only(existing_generated)
            if question_texts:
                prompt += f"""

AVOID REPETITION: You have already generated problems with the following question texts. DO NOT create similar problems:
"""
                for i, question_text in enumerate(question_texts, 1):
                    prompt += f"{i}. {question_text}\n"
                prompt += "\n"

        prompt += f"""

FINAL INSTRUCTION:
First, complete all your internal thinking, analysis, and problem creation steps.
After you have finished thinking, provide your response which MUST ONLY BE the final, clean, valid JSON array containing exactly {num_to_generate} problems. Do not include any of your thought process, notes, or any other text before or after the JSON array. The response must start with `[` and end with `]`.

THINKING AND OUTPUT FORMAT:
You may think through the problem creation process before generating the JSON. Feel free to:
- Analyze the examples to understand the pattern
- Consider mathematical concepts and difficulty levels
- Plan the variety of problems you'll create
- Design detailed figure descriptions

However, after your thinking is complete, provide ONLY the JSON array with this exact format:

[
  {{
    "test": "{test_name}",
    "domain": "{domain}",
    "skill": "{skill}",
    "difficulty": "Easy|Medium|Hard",
    "question_text_latex": "Question with proper $LaTeX$ formatting",
    "option_A_latex": "Option A",
    "option_B_latex": "Option B", 
    "option_C_latex": "Option C",
    "option_D_latex": "Option D",
    "correct_answer": "A|B|C|D or direct answer",
    "correct_answer_spr_latex": "Answer in LaTeX format",
    "figure_description": "ULTRA-DETAILED figure description with complete drawing specifications",
    "step_1": "First step explanation with LaTeX",
    "step_2": "Second step explanation with LaTeX",
    "step_3": "Third step explanation with LaTeX (if needed)",
    "step_4": "Fourth step explanation with LaTeX (if needed)",
    "step_5": "Fifth step explanation with LaTeX (if needed)",
    "step_6": "Sixth step explanation with LaTeX (if needed)"
  }}
]

CRITICAL REQUIREMENTS: 
- **figure_description is MANDATORY and must be ULTRA-DETAILED**: Every problem must have an extremely detailed figure description
- **DRAWING-LEVEL PRECISION**: Include enough detail that a technical illustrator could recreate the exact figure
- **SPECIFY EVERYTHING**: Exact coordinates, line styles, colors, fonts, positioning, spacing, borders, alignments, tick marks, labels, orientations, scales, and visual formatting
- Use separate step_1, step_2, etc. fields for each explanation step
- Each step should be complete and standalone
- Fill UP TO 6 STEPS MAXIMUM (can be fewer, leave unused steps empty or null)
- IMPORTANT: In JSON, escape backslashes properly (use \\\\\\ for LaTeX \\\\, \\\\$ for $, etc.)
- Ensure all LaTeX expressions are properly escaped for valid JSON format
- Do not include any text before the opening [ bracket
- Do not include any text after the closing ] bracket
- The JSON must contain exactly {num_to_generate} complete problem objects


FINAL INSTRUCTION:
First, complete all your internal thinking, analysis, and problem creation steps.
After you have finished thinking, provide your response which MUST ONLY BE the final, clean, valid JSON array containing exactly {num_to_generate} problems. Do not include any of your thought process, notes, or any other text before or after the JSON array. The response must start with `[` and end with `]`."""
        
        return prompt
    
    async def generate_batch_with_backoff(self, prompt: str, skill: str, max_retries: int = 5, 
                                        base_delay: float = 2.0) -> Tuple[Optional[List[Dict[str, Any]]], int, int, int]:
        """Generate a batch of tasks with exponential backoff and rate limiting"""
        
        # Initialize retry tracking
        retry_count = 0
        max_total_retries = max_retries + 2  # Allow a few extra retries for certain error types
        client_reset_count = 0
        max_client_resets = 2
        
        # Track specific API errors
        api_errors = {
            "RetryError": 0,
            "ClientError": 0,
            "TimeoutError": 0,
            "ResourceExhausted": 0,
            "Other": 0
        }
        
        async with self.rate_limiter:
            # Add delay between requests
            await asyncio.sleep(self.request_delay)
            
            while retry_count < max_total_retries:
                should_reset_client = False
                response_text = ""
                
                try:
                    self.logger.info(f" [{skill}] Generating batch (attempt {retry_count + 1}/{max_total_retries})...")
                    
                    # Get thread-local client (potentially reset if needed from previous errors)
                    client = self.get_client(force_reset=(client_reset_count > 0))
                    
                    # Prepare content for the new API
                    contents = [
                        types.Content(
                            role="user",
                            parts=[types.Part(text=prompt)]
                        )
                    ]
                    
                    # Configure generation settings with more conservative settings
                    generate_content_config = types.GenerateContentConfig(
                        temperature=0.9,  # Slightly reduced temperature for more predictable results
                        top_p=0.9,     # Slightly reduced top_p for more predictable results
                        max_output_tokens=65535,
                        safety_settings=[
                            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
                        ]
                    )
                    
                    # Generate content using streaming (but collect all chunks)
                    input_tokens = 0
                    output_tokens = 0
                    total_tokens = 0
                    
                    # Run the synchronous API call in a thread pool
                    loop = asyncio.get_event_loop()
                    
                    def generate_content():
                        nonlocal response_text, input_tokens, output_tokens, total_tokens
                        
                        try:
                            for chunk in client.models.generate_content_stream(
                                model=self.model,
                                contents=contents,
                                config=generate_content_config,
                                # Note: timeout is handled at the asyncio level instead
                            ):
                                if chunk.text:
                                    response_text += chunk.text
                                
                                # Track token usage if available
                                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                                    if hasattr(chunk.usage_metadata, 'prompt_token_count'):
                                        input_tokens = chunk.usage_metadata.prompt_token_count
                                    if hasattr(chunk.usage_metadata, 'candidates_token_count'):
                                        output_tokens = chunk.usage_metadata.candidates_token_count
                                    if hasattr(chunk.usage_metadata, 'total_token_count'):
                                        total_tokens = chunk.usage_metadata.total_token_count
                        except Exception as e:
                            error_name = type(e).__name__
                            self.logger.error(f" API Stream Error: {error_name}: {str(e)}")
                            
                            # Flag for client reset on specific errors
                            if ("RetryError" in error_name or 
                                "ClientError" in error_name or 
                                "ConnectionError" in error_name or
                                "ResourceExhausted" in str(e)):
                                nonlocal should_reset_client
                                should_reset_client = True
                                
                            # Re-raise to be handled in the main try/except
                            raise
                    
                    # Execute the API call asynchronously with a longer timeout using our managed thread pool
                    try:
                        await asyncio.wait_for(
                            loop.run_in_executor(self.executor, generate_content), 
                            timeout=240  # Extended timeout to 4 minutes
                        )
                    except asyncio.TimeoutError:
                        self.logger.info(f" [{skill}] API request timed out after 240 seconds")
                        api_errors["TimeoutError"] += 1
                        should_reset_client = True
                        raise
                    except Exception as e:
                        self.logger.error(f" [{skill}] API execution error: {type(e).__name__}: {str(e)}")
                        raise
                    
                    response_text = response_text.strip()
                    
                    # Print token usage information
                    if total_tokens > 0:
                        self.logger.info(f"[{skill}] Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
                    else:
                        self.logger.warning(f"[{skill}] Token usage information not available")
                    
                    # Clean up response
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]
                    
                    response_text = response_text.strip()
                    
                    # Apply minimal LaTeX escaping fixes for common JSON issues
                    response_text = self.fix_common_latex_escaping(response_text)
                    
                    # Parse JSON with fallback methods
                    new_tasks = self.parse_json_with_fallback(response_text)
                    
                    # Validate structure
                    if not isinstance(new_tasks, list):
                        raise ValueError("Response is not a list")
                    
                    if not new_tasks:  # Empty list
                        raise ValueError("Response contains empty list of tasks")
                    
                    # Process and clean generated tasks
                    processed_tasks = self.process_generated_tasks(new_tasks)
                    
                    # Check if we got a reasonable number of tasks
                    if len(processed_tasks) < 1:
                        raise ValueError(f"Too few tasks generated: {len(processed_tasks)}")
                    
                    self.logger.info(f"[{skill}] Successfully generated {len(processed_tasks)} tasks")
                    
                    # Reset error counts on success
                    with self.lock:
                        self.error_counters["success"] = self.error_counters.get("success", 0) + 1
                        
                    return processed_tasks, input_tokens, output_tokens, total_tokens
                    
                except json.JSONDecodeError as e:
                    error_msg = f"JSON parsing error (attempt {retry_count + 1}): {e}"
                    self.logger.error(f"[{skill}] {error_msg}")
                    
                    # Log the prompt that caused the error
                    self.logger.error(f"[{skill}] === PROMPT THAT CAUSED ERROR ===")
                    self.logger.error(f"[{skill}] Prompt length: {len(prompt)} characters")
                    self.logger.error(f"[{skill}] Prompt content (first 500 chars): {prompt[:500]}...")
                    if len(prompt) > 500:
                        self.logger.error(f"[{skill}] Prompt content (last 200 chars): ...{prompt[-200:]}")
                    
                    # Log the full Gemini response for debugging
                    self.logger.error(f"[{skill}] === FULL GEMINI RESPONSE (JSON Parse Error) ===")
                    self.logger.error(f"[{skill}] Response length: {len(response_text)} characters")
                    self.logger.error(f"[{skill}] Full response content:")
                    self.logger.error(f"[{skill}] {'-' * 80}")
                    self.logger.error(f"[{skill}] {response_text}")
                    self.logger.error(f"[{skill}] {'-' * 80}")
                    
                    # Also log first and last 200 characters for quick scanning
                    if len(response_text) > 400:
                        self.logger.error(f"[{skill}] Response preview (first 200 chars): {response_text[:200]}...")
                        self.logger.error(f"[{skill}] Response preview (last 200 chars): ...{response_text[-200:]}")
                    
                    # Count error type
                    error_type = f"JSONDecodeError_{type(e).__name__}"
                    self.error_counters[error_type] += 1
                    
                except ValueError as e:
                    error_msg = f"Value error (attempt {retry_count + 1}): {e}"
                    self.logger.error(f"[{skill}] {error_msg}")
                    
                    # Log the prompt that caused the error
                    self.logger.error(f"[{skill}] === PROMPT THAT CAUSED ERROR ===")
                    self.logger.error(f"[{skill}] Prompt length: {len(prompt)} characters")
                    self.logger.error(f"[{skill}] Prompt content (first 500 chars): {prompt[:500]}...")
                    if len(prompt) > 500:
                        self.logger.error(f"[{skill}] Prompt content (last 200 chars): ...{prompt[-200:]}")
                    
                    # Log the full Gemini response for debugging
                    self.logger.error(f"[{skill}] === FULL GEMINI RESPONSE (Value Error) ===")
                    self.logger.error(f"[{skill}] Response length: {len(response_text)} characters")
                    self.logger.error(f"[{skill}] Full response content:")
                    self.logger.error(f"[{skill}] {'-' * 80}")
                    self.logger.error(f"[{skill}] {response_text}")
                    self.logger.error(f"[{skill}] {'-' * 80}")
                    
                    # Count error type
                    error_type = f"ValueError_{str(e)[:50]}"
                    self.error_counters[error_type] += 1
                    
                except Exception as e:
                    error_name = type(e).__name__
                    error_message = str(e)
                    
                    error_msg = f"API execution error: {error_name}: {error_message}"
                    self.logger.error(f"[{skill}] {error_msg}")
                    
                    # Log the prompt that caused the error
                    self.logger.error(f"[{skill}] === PROMPT THAT CAUSED ERROR ===")
                    self.logger.error(f"[{skill}] Prompt length: {len(prompt)} characters")
                    self.logger.error(f"[{skill}] Prompt content (first 500 chars): {prompt[:500]}...")
                    if len(prompt) > 500:
                        self.logger.error(f"[{skill}] Prompt content (last 200 chars): ...{prompt[-200:]}")
                    
                    # Log the full Gemini response for debugging if we have one
                    if 'response_text' in locals() and response_text:
                        self.logger.error(f"[{skill}] === FULL GEMINI RESPONSE (API Error) ===")
                        self.logger.error(f"[{skill}] Response length: {len(response_text)} characters")
                        self.logger.error(f"[{skill}] Full response content:")
                        self.logger.error(f"[{skill}] {'-' * 80}")
                        self.logger.error(f"[{skill}] {response_text}")
                        self.logger.error(f"[{skill}] {'-' * 80}")
                    else:
                        self.logger.error(f"[{skill}] No response text available (error occurred before response)")
                    
                    # Count error type
                    error_type = f"{error_name}_{error_message[:50]}"
                    self.error_counters[error_type] += 1
                
                retry_count += 1
                
                # Reset client if needed (but limit the number of resets)
                if should_reset_client and client_reset_count < max_client_resets:
                    client_reset_count += 1
                    self.logger.debug(f"[{skill}] Resetting API client (reset #{client_reset_count})...")
                    self.get_client(force_reset=True)
                
                if retry_count < max_total_retries:
                    # Calculate delay with more aggressive exponential backoff and jitter
                    delay = base_delay * (3 ** min(retry_count, 3)) + random.uniform(2, 10)
                    
                    # Add extra delay for specific error types
                    if api_errors["RetryError"] > 0 or api_errors["ClientError"] > 0:
                        delay += 10 + (api_errors["RetryError"] + api_errors["ClientError"]) * 5
                    
                    # Cap the max delay at 2 minutes
                    delay = min(delay, 120)
                    
                    self.logger.info(f"[{skill}] Waiting {delay:.1f} seconds before retry {retry_count + 1}/{max_total_retries}...")
                    
                    # Track errors in the global counter
                    with self.lock:
                        for err_type, count in api_errors.items():
                            if count > 0:
                                self.error_counters[err_type] = self.error_counters.get(err_type, 0) + 1
                    
                    await asyncio.sleep(delay)
            
            # If we get here, all attempts failed
            self.logger.error(f"[{skill}] All {max_total_retries} attempts failed")
            
            # Log error summary
            if api_errors:
                self.logger.error(f"[{skill}] Error summary after {max_total_retries} attempts:")
                for error_type, count in api_errors.items():
                    if count > 0:
                        self.logger.error(f"[{skill}]   - {error_type}: {count} occurrences")
            
            return None, 0, 0, 0
    
    async def generate_tasks_for_skill(self, skill: str, examples: List[Dict[str, Any]], 
                                     total_tasks: int = 100, batch_size: int = 25) -> List[Dict[str, Any]]:
        """Generate tasks for a specific skill in batches"""
        all_generated = []
        examples_for_prompt = self.prepare_examples_for_prompt(examples, 15)
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens_used = 0
        
        self.logger.info(f"n[{skill}] Starting generation of {total_tasks} tasks")
        self.logger.info(f"{skill}] Available examples: {len(examples)}, Using: {len(examples_for_prompt)} examples")
        
        num_batches = (total_tasks + batch_size - 1) // batch_size
        
        for batch_num in range(num_batches):
            remaining_tasks = total_tasks - len(all_generated)
            current_batch_size = min(batch_size, remaining_tasks)
            
            self.logger.info(f"{skill}] Batch {batch_num + 1}/{num_batches}: Generating {current_batch_size} tasks...")
            
            prompt = self.create_generation_prompt(
                examples_for_prompt, 
                skill, 
                all_generated,  # Pass existing generated tasks to avoid repetition
                current_batch_size
            )
            
            batch_result = await self.generate_batch_with_backoff(prompt, skill)
            
            if batch_result[0]:  # batch_tasks is not None
                batch_tasks, input_tokens, output_tokens, total_tokens = batch_result
                all_generated.extend(batch_tasks)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_tokens_used += total_tokens
                self.logger.info(f" [{skill}] Total generated so far: {len(all_generated)}/{total_tasks}")
            else:
                self.logger.error(f" [{skill}] Failed to generate batch {batch_num + 1}")
                break
            
            # Small delay between batches
            if batch_num < num_batches - 1:
                await asyncio.sleep(2)
        
        self.logger.info(f"n[{skill}] Completed generation: {len(all_generated)} tasks")
        self.logger.info(f"{skill}] Total token usage: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_tokens_used}")
        
        # Update progress
        with self.lock:
            self.completed_skills += 1
            self.logger.info(f"n📊 Progress: {self.completed_skills}/{self.total_skills} skills completed")
        
        return all_generated
    
    def save_tasks_incrementally(self, tasks: List[Dict[str, Any]], skill: str, batch_count: int):
        """Save tasks incrementally every 10 tasks to avoid losing progress"""
        if not tasks:
            return
            
        # Create filename
        safe_skill_name = "".join(c for c in skill if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_skill_name = safe_skill_name.replace(' ', '_')
        filename = f"generated_{safe_skill_name}.csv"
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(filename)
        
        try:
            # Process tasks before saving
            processed_tasks = self.process_generated_tasks(tasks)
            
            if not processed_tasks:
                self.logger.warning(f"No valid tasks to save incrementally for {skill}")
                return
            
            # Define expected fieldnames
            fieldnames = [
                'test', 'domain', 'skill', 'difficulty', 'question_text_latex',
                'option_A_latex', 'option_B_latex', 'option_C_latex', 'option_D_latex',
                'correct_answer', 'correct_answer_spr_latex', 'figure_description',
                'step_1', 'step_2', 'step_3', 'step_4', 'step_5', 'step_6'
            ]
            
            # Append to existing file or create new one
            mode = 'a' if file_exists else 'w'
            with open(filename, mode, newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                
                # Write header only if file is new
                if not file_exists:
                    writer.writeheader()
                
                writer.writerows(processed_tasks)
            
            self.logger.info(f"💾 Incrementally saved {len(processed_tasks)} tasks to {filename} (batch #{batch_count})")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving incremental tasks for {skill}: {e}")
    
    async def generate_tasks_for_skill_by_difficulty(self, skill: str, examples: List[Dict[str, Any]], 
                                                   tasks_per_difficulty: int = 100, batch_size: int = 25) -> List[Dict[str, Any]]:
        """Generate tasks for a specific skill with equal distribution across difficulty levels"""
        
        # Group examples by difficulty
        examples_by_difficulty = {}
        for example in examples:
            difficulty = example.get('difficulty', '').strip()
            if difficulty:
                if difficulty not in examples_by_difficulty:
                    examples_by_difficulty[difficulty] = []
                examples_by_difficulty[difficulty].append(example)
        
        self.logger.info(f"n[{skill}] Starting generation of {tasks_per_difficulty} tasks per difficulty level")
        self.logger.info(f"{skill}] Available difficulties: {list(examples_by_difficulty.keys())}")
        for difficulty, examples_list in examples_by_difficulty.items():
            self.logger.info(f"{skill}] {difficulty}: {len(examples_list)} examples")
        
        all_generated = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens_used = 0
        
        # Counter for incremental saving
        total_saved_count = 0
        incremental_batch_count = 0
        
        # Generate tasks for each difficulty level
        for difficulty in sorted(examples_by_difficulty.keys()):
            difficulty_examples = examples_by_difficulty[difficulty]
            
            if len(difficulty_examples) < 3:  # Need minimum examples for this difficulty
                self.logger.warning(f"️  [{skill}] Skipping {difficulty}: only {len(difficulty_examples)} examples (need at least 3)")
                continue
            
            self.logger.info(f"n[{skill}] Generating {tasks_per_difficulty} tasks for difficulty: {difficulty}")
            
            # Prepare examples for this difficulty
            examples_for_prompt = self.prepare_examples_for_prompt(difficulty_examples, min(15, len(difficulty_examples)))
            
            # Generate tasks in batches for this difficulty
            difficulty_generated = []
            num_batches = (tasks_per_difficulty + batch_size - 1) // batch_size
            
            for batch_num in range(num_batches):
                remaining_tasks = tasks_per_difficulty - len(difficulty_generated)
                current_batch_size = min(batch_size, remaining_tasks)
                
                self.logger.info(f"{skill}] {difficulty} - Batch {batch_num + 1}/{num_batches}: Generating {current_batch_size} tasks...")
                
                prompt = self.create_difficulty_specific_prompt(
                    examples_for_prompt, 
                    skill, 
                    difficulty,
                    difficulty_generated,  # Pass existing generated tasks for this difficulty
                    current_batch_size
                )
                
                # Add retry for the entire batch with increasing delays
                max_batch_retries = 2
                for batch_retry in range(max_batch_retries + 1):
                    try:
                        if batch_retry > 0:
                            retry_delay = 30 * batch_retry
                            self.logger.info(f" [{skill}] {difficulty} - Batch retry {batch_retry}/{max_batch_retries} after {retry_delay}s delay")
                            await asyncio.sleep(retry_delay)
                            
                        batch_result = await self.generate_batch_with_backoff(prompt, f"{skill}-{difficulty}")
                        
                        if batch_result[0]:  # batch_tasks is not None
                            batch_tasks, input_tokens, output_tokens, total_tokens = batch_result
                            
                            # Ensure all generated tasks have the correct difficulty
                            for task in batch_tasks:
                                task['difficulty'] = difficulty
                            
                            difficulty_generated.extend(batch_tasks)
                            all_generated.extend(batch_tasks)
                            
                            total_input_tokens += input_tokens
                            total_output_tokens += output_tokens
                            total_tokens_used += total_tokens
                            
                            self.logger.info(f" [{skill}] {difficulty} - Generated so far: {len(difficulty_generated)}/{tasks_per_difficulty}")
                            
                            # Check if we should save incrementally (every 10 tasks)
                            if len(all_generated) - total_saved_count >= 10:
                                incremental_batch_count += 1
                                tasks_to_save = all_generated[total_saved_count:]
                                self.save_tasks_incrementally(tasks_to_save, skill, incremental_batch_count)
                                total_saved_count = len(all_generated)
                            
                            break  # Success, exit retry loop
                        else:
                            self.logger.error(f" [{skill}] {difficulty} - Failed to generate batch {batch_num + 1} (attempt {batch_retry + 1})")
                            if batch_retry == max_batch_retries:
                                # All retries failed
                                self.logger.error(f" ❌ [{skill}] {difficulty} - All batch attempts failed, moving to next batch")
                                # Don't break the outer batch loop, try the next batch
                    except Exception as e:
                        self.logger.error(f" ❌ [{skill}] {difficulty} - Error in batch {batch_num + 1} (attempt {batch_retry + 1}): {e}")
                        if batch_retry == max_batch_retries:
                            self.logger.error(f" ❌ [{skill}] {difficulty} - All batch attempts failed with errors")
                            # Continue to next batch
                
                # Small delay between batches
                if batch_num < num_batches - 1:
                    await asyncio.sleep(2)
            
            self.logger.info(f"{skill}] {difficulty} - Completed: {len(difficulty_generated)} tasks")
        
        # Save any remaining tasks that haven't been saved incrementally
        if len(all_generated) > total_saved_count:
            incremental_batch_count += 1
            remaining_tasks = all_generated[total_saved_count:]
            self.save_tasks_incrementally(remaining_tasks, skill, incremental_batch_count)
            self.logger.info(f"💾 Saved final {len(remaining_tasks)} remaining tasks for {skill}")
        
        self.logger.info(f"n[{skill}] TOTAL COMPLETED: {len(all_generated)} tasks across all difficulties")
        self.logger.info(f"{skill}] Total token usage: Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_tokens_used}")
        self.logger.info(f"💾 Total incremental saves: {incremental_batch_count} batches")
        
        # Update progress
        with self.lock:
            self.completed_skills += 1
            self.logger.info(f"n📊 Progress: {self.completed_skills}/{self.total_skills} skills completed")
        
        return all_generated

    def create_difficulty_specific_prompt(self, examples: List[Dict[str, Any]], skill: str, difficulty: str,
                                        existing_generated: List[Dict[str, Any]] = None, num_to_generate: int = 10) -> str:
        """Create a detailed prompt for generating math tasks for a specific difficulty level"""
        
        test_name = examples[0].get('test', 'Math') if examples else 'Math'
        domain = examples[0].get('domain', 'Algebra') if examples else 'Algebra'
        
        prompt = f"""You are an expert math educator creating high-quality {skill} problems with visual elements. 

TASK: Generate {num_to_generate} new math problems for the skill "{skill}" at {difficulty} difficulty level that match the style and structure of the provided examples.

EXAMPLES FROM EXISTING DATASET (ALL {difficulty.upper()} DIFFICULTY):
{json.dumps(examples, indent=2)}

REQUIREMENTS:
1. **Maintain Consistency**: Follow the exact same format and structure as the examples
2. **Specific Difficulty**: ALL problems must be {difficulty} difficulty level - no other difficulties allowed
3. **Mathematical Accuracy**: Ensure all calculations are correct
4. **Complete Structure**: Include all required fields with proper difficulty: "{difficulty}"
5. **Figure Descriptions**: Create ULTRA-DETAILED figure descriptions for complete visual recreation
6. **{difficulty} Difficulty Level**: Ensure complexity matches the difficulty level exactly"""

        if existing_generated:
            prompt += f"""

AVOID REPETITION: You have already generated the following {difficulty} problems. DO NOT create similar problems:
{json.dumps(existing_generated, indent=2)} 
"""

        prompt += f"""

OUTPUT FORMAT: Return ONLY a valid JSON array of {num_to_generate} problems, no additional text:

[
  {{
    "test": "{test_name}",
    "domain": "{domain}",
    "skill": "{skill}",
    "difficulty": "{difficulty}",
    "question_text_latex": "Question with proper $LaTeX$ formatting",
    "figure_description": "ULTRA-DETAILED, DRAWING-READY description of visual elements. Must be so specific and comprehensive that an artist could recreate the exact figure without any ambiguity. Include: precise coordinate systems with exact ranges and tick mark positions, detailed axis labels and positioning, complete gridline specifications, exact line styles (solid/dashed/dotted), precise curve descriptions with all key points and coordinates, detailed table structures with borders/formatting/alignment, geometric figure orientations with exact measurements and label positions, color specifications, font details, spacing measurements, and any other visual elements. Use technical drawing precision.",
    "option_A_latex": "Option A (include tables in MathJax if needed)",
    "option_B_latex": "Option B (include tables in MathJax if needed)", 
    "option_C_latex": "Option C (include tables in MathJax if needed)",
    "option_D_latex": "Option D (include tables in MathJax if needed)",
    "correct_answer": "A|B|C|D or direct answer",
    "correct_answer_spr_latex": "Answer in LaTeX format",
    "step_1": "First step explanation with LaTeX",
    "step_2": "Second step explanation with LaTeX",
    "step_3": "Third step explanation with LaTeX (if needed)",
    "step_4": "Fourth step explanation with LaTeX (if needed)",
    "step_5": "Fifth step explanation with LaTeX (if needed)",
    "step_6": "Sixth step explanation with LaTeX (if needed)"
  }}
]

CRITICAL REQUIREMENTS: 
- **ALL problems must be {difficulty} difficulty - no exceptions**
- **figure_description is MANDATORY and must be ULTRA-DETAILED**: Every problem must have an extremely detailed figure description
- **DRAWING-LEVEL PRECISION**: Include enough detail that a technical illustrator could recreate the exact figure
- Use separate step_1, step_2, etc. fields for each explanation step
- Each step should be complete and standalone
- Fill UP TO 6 STEPS MAXIMUM (can be fewer, leave unused steps empty or null)
- IMPORTANT: In JSON, escape backslashes properly (use \\\\\\ for LaTeX \\\\, \\\\$ for $, etc.)
- Ensure all LaTeX expressions are properly escaped for valid JSON format"""
        
        return prompt

    def get_generated_skill_counts(self) -> Dict[str, int]:
        """Get a dictionary of skill names and their generated task counts."""
        skill_counts = defaultdict(int)
        pattern = "generated_*.csv"
        csv_files = glob.glob(pattern)
        
        self.logger.info(f"Checking for existing skill files with pattern: '{pattern}'")

        for filepath in csv_files:
            try:
                filename = os.path.basename(filepath)
                skill_name = filename[len("generated_"):-len(".csv")]
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    count = len(list(reader))
                skill_counts[skill_name] = count
            except Exception as e:
                self.logger.warning(f"Could not count rows in {filepath}: {e}")
        
        if skill_counts:
            self.logger.info(f"Found task counts for {len(skill_counts)} existing skills.")
        else:
            self.logger.info("No existing generated skill files found.")
            
        return skill_counts
    
    async def process_skill_batch(self, skill_batch: List[Tuple[str, List[Dict[str, Any]]]], 
                                tasks_per_difficulty: int, test_mode: bool, test_tasks: int):
        """Process a batch of skills concurrently using difficulty-based generation"""
        async def process_single_skill(skill, examples):
            try:
                # Generate tasks by difficulty level (100 per difficulty = 300 total)
                if test_mode:
                    # In test mode, generate fewer tasks per difficulty
                    target_tasks_per_difficulty = max(1, test_tasks // 3)  # Divide test tasks across difficulties
                else:
                    target_tasks_per_difficulty = tasks_per_difficulty
                
                # Add a small staggered delay before starting each skill to avoid burst API requests
                thread_id = threading.get_ident() % 100
                await asyncio.sleep(thread_id * 0.5)  # Stagger starts by 0.5s per thread
                
                generated_tasks = await self.generate_tasks_for_skill_by_difficulty(
                    skill, examples, target_tasks_per_difficulty
                )
                
                if generated_tasks:
                    # Tasks are already saved incrementally during generation
                    self.logger.info(f"✅ Completed generation for skill '{skill}' with {len(generated_tasks)} tasks (saved incrementally)")
                    return skill, len(generated_tasks)
                else:
                    self.logger.info(f"️ No tasks were generated for skill '{skill}'")
                    return skill, 0
            except asyncio.TimeoutError as e:
                self.logger.error(f" Timeout error processing skill '{skill}': {e}")
                return skill, f"TIMEOUT: {str(e)}"
            except Exception as e:
                error_type = type(e).__name__
                self.logger.error(f" Error processing skill '{skill}': {error_type}: {e}")
                
                # Track specific error types
                with self.lock:
                    self.error_counters[error_type] = self.error_counters.get(error_type, 0) + 1
                
                # Return error type for reporting
                return skill, f"ERROR: {error_type}: {str(e)[:100]}"
        
        # Process all skills in the batch
        tasks = []
        for skill, examples in skill_batch:
            # Add each task to be processed
            tasks.append(process_single_skill(skill, examples))
        
        # Process all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results - handle any exceptions caught by return_exceptions=True
        processed_results = []
        for i, result in enumerate(results):
            skill = skill_batch[i][0]
            if isinstance(result, Exception):
                error_type = type(result).__name__
                error_msg = str(result)
                self.logger.error(f" Unhandled exception for skill '{skill}': {error_type}: {error_msg}")
                processed_results.append((skill, f"EXCEPTION: {error_type}"))
                
                # Track it in error counters
                with self.lock:
                    self.error_counters[error_type] = self.error_counters.get(error_type, 0) + 1
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def process_all_skills(self, input_csv: str, tasks_per_difficulty: int = 100, 
                               test_mode: bool = False, test_tasks: int = 10, 
                               only_remaining: bool = True, batch_size: int = None):
        """Process all skills from the input CSV concurrently with difficulty-based generation"""
        self.logger.info("=== Concurrent Math Task Generator (Difficulty-Based) ===")
        self.logger.info(f"Input file: {input_csv}")
        self.logger.info(f"Tasks per difficulty level: {tasks_per_difficulty} × 3 difficulties = {tasks_per_difficulty * 3} total per skill")
        self.logger.info(f"Test mode: {'Enabled - ' + str(test_tasks) + ' tasks per skill' if test_mode else 'Disabled'}")
        self.logger.info(f"Project ID: {self.project_id}")
        self.logger.info(f"Concurrent workers: {self.max_workers}")
        self.logger.info(f"Skills batch size: {batch_size}")
        
        # Read tasks by skill
        tasks_by_skill = self.read_tasks_by_skill(input_csv)
        
        if not tasks_by_skill:
            self.logger.info("No tasks found in input file")
            return
        
        # Get already generated skills if requested
        skill_counts = {}
        if only_remaining:
            skill_counts = self.get_generated_skill_counts()

        # Filter skills to process
        skills_to_process = []
        if only_remaining and not skill_counts:
            self.logger.info("No existing files found. Processing all skills.")
            skills_to_process = list(tasks_by_skill.items())
        else:
            for skill, examples in tasks_by_skill.items():
                if only_remaining and skill_counts.get(skill, 0) >= 90:
                    self.logger.warning(f"️  Skipping {skill}: already has {skill_counts.get(skill, 0)} tasks (≥ 90)")
                    continue
                
                if len(examples) < 5:  # Need minimum examples
                    self.logger.warning(f"️  Skipping {skill}: only {len(examples)} examples (need at least 5)")
                    continue
                
                skills_to_process.append((skill, examples))
        
        if not skills_to_process:
            self.logger.info("🎉 No remaining skills to generate! All skills are already done.")
            return
        
        self.total_skills = len(skills_to_process)
        self.completed_skills = 0
        
        self.logger.info(f"n⏳ Will generate tasks for {len(skills_to_process)} skill(s) using {self.max_workers} concurrent workers:")
        for skill, _ in skills_to_process:
            self.logger.info(f" 🔄 {skill}")
        
        # If batch_size is not specified, use max_workers
        if batch_size is None:
            batch_size = self.max_workers
        
        # Process skills in batches to control concurrency
        start_time = time.time()
        total_generated = 0
        
        for i in range(0, len(skills_to_process), batch_size):
            batch = skills_to_process[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(skills_to_process) + batch_size - 1) // batch_size
            
            self.logger.info(f"n🚀 Processing batch {batch_num}/{total_batches} ({len(batch)} skills)")
            self.logger.info(f"  Skills: {', '.join([skill for skill, _ in batch])}")
            
            # If this isn't the first batch, add a cooling period between batches to avoid rate limits
            if i > 0:
                cooling_period = 30  # 30 seconds between batches to avoid hitting API limits
                self.logger.info(f"️ Adding cooling period of {cooling_period}s between batches to avoid API rate limits...")
                await asyncio.sleep(cooling_period)
            
            batch_start = time.time()
            results = await self.process_skill_batch(batch, tasks_per_difficulty, test_mode, test_tasks)
            batch_time = time.time() - batch_start
            
            # Count successful generations in this batch
            batch_generated = sum(count for skill, count in results if isinstance(count, int) and count > 0)
            total_generated += batch_generated
            
            # Track errors
            batch_errors = sum(1 for skill, count in results if not isinstance(count, int) or count == 0)
            
            self.logger.info(f" Batch {batch_num} completed in {batch_time:.1f}s, generated {batch_generated} tasks")
            if batch_errors > 0:
                self.logger.error(f"️ {batch_errors} skills had errors in this batch")
        
        total_time = time.time() - start_time
        
        self.logger.info(f"n🎉 All skills processed!")
        self.logger.info(f" Total time: {total_time:.1f}s")
        self.logger.info(f" Total tasks generated: {total_generated}")
        self.logger.info(f" Average time per skill: {total_time / len(skills_to_process):.1f}s")
        self.logger.info(f" Skills processed: {self.completed_skills}/{self.total_skills}")

    def process_generated_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean generated tasks before saving"""
        processed_tasks = []
        
        for i, task in enumerate(tasks):
            # First, validate that the task is a dictionary
            if not isinstance(task, dict):
                self.logger.error(f"=== PROBLEMATIC TASK ITEM ===")
                self.logger.error(f"Task at index {i} is not a dictionary")
                self.logger.error(f"Task type: {type(task)}")
                self.logger.error(f"Task content: {task}")
                self.logger.error(f"Task length: {len(str(task))} characters")
                continue
            
            # Clean up all string values
            cleaned_task = {}
            for key, value in task.items():
                if isinstance(value, str):
                    # Clean math content and trim whitespace
                    cleaned_value = self.clean_math_content(value.strip())
                    cleaned_task[key] = cleaned_value
                else:
                    cleaned_task[key] = value
            
            # Ensure required fields are present
            required_fields = ["skill", "difficulty", "question_text_latex", "correct_answer", "figure_description"]
            missing_fields = [field for field in required_fields if not cleaned_task.get(field)]
            
            if missing_fields:
                self.logger.warning(f"Task missing required fields: {', '.join(missing_fields)}. Adding defaults.")
                self.logger.warning(f"=== TASK WITH MISSING FIELDS ===")
                self.logger.warning(f"Task data: {cleaned_task}")
                
                # Add default values for missing fields
                for field in missing_fields:
                    if field == "skill":
                        cleaned_task["skill"] = "Unknown Skill"
                    elif field == "difficulty":
                        cleaned_task["difficulty"] = "Medium"
                    elif field == "question_text_latex":
                        cleaned_task["question_text_latex"] = "Question text missing"
                    elif field == "correct_answer":
                        cleaned_task["correct_answer"] = "Unknown"
                    elif field == "figure_description":
                        cleaned_task["figure_description"] = "Figure description missing"
            
            # Convert format issues in LaTeX fields
            for key in cleaned_task:
                if key.endswith('_latex') and isinstance(cleaned_task[key], str):
                    # Fix common LaTeX formatting issues
                    cleaned_task[key] = cleaned_task[key].replace('[Math: ', '$').replace(']', '$')
                    # Ensure proper MathJax delimiters
                    cleaned_task[key] = cleaned_task[key].replace('$$$', '$').replace('$$', '$')
            
            processed_tasks.append(cleaned_task)
        
        return processed_tasks
    
    def save_tasks_to_csv(self, tasks: List[Dict[str, Any]], filename: str):
        """Save tasks to CSV file with detailed error diagnostics"""
        if not tasks:
            self.logger.info(f"No tasks to save for {filename}")
            return
        
        # Define the column order
        columns = [
            'test', 'domain', 'skill', 'difficulty', 'question_text_latex',
            'figure_description',  # Include figure_description for images version
            'option_A_latex', 'option_B_latex', 'option_C_latex', 'option_D_latex',
            'correct_answer', 'correct_answer_spr_latex', 
            'step_1', 'step_2', 'step_3', 'step_4', 'step_5', 'step_6'
        ]
        
        try:
            with self.lock:  # Thread-safe file writing
                with open(filename, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(tasks)
            self.logger.info(f" Saved {len(tasks)} tasks to {filename}")
        except ValueError as e:
            if "fields not in fieldnames" in str(e):
                self.logger.error(f" CSV Writing Error: Extra fields detected in tasks for {filename}")
                self.logger.info(f"rror message: {e}")
                self.logger.info("\n🔍 DEBUGGING: Full content of problematic tasks:")
                self.logger.info("=" * 80)
                
                # Find and print the complete content of tasks with extra fields
                problematic_count = 0
                for i, task in enumerate(tasks):
                    extra_fields = set(task.keys()) - set(columns)
                    if extra_fields:
                        problematic_count += 1
                        self.logger.info(f"n📋 PROBLEMATIC TASK #{i + 1}:")
                        self.logger.info(f"  Extra fields detected: {sorted(extra_fields)}")
                        self.logger.info(f"  FULL TASK CONTENT:")
                        self.logger.info("-" * 60)
                        
                        # Print all fields in the task for complete debugging info
                        for key, value in task.items():
                            # Truncate very long values for readability but keep essential info
                            if isinstance(value, str) and len(value) > 200:
                                truncated_value = value[:200] + "... [TRUNCATED]"
                            else:
                                truncated_value = value
                            
                            # Highlight extra fields
                            marker = "🚨 EXTRA → " if key in extra_fields else "         "
                            self.logger.info(f"  {marker}{key}: {repr(truncated_value)}")
                        
                        self.logger.info("-" * 60)
                
                self.logger.info(f"n📊 Summary: Found {problematic_count} tasks with extra fields out of {len(tasks)} total tasks")
                self.logger.info("❌ This error should not occur after validation - please check the validation logic")
            else:
                self.logger.error(f" ValueError saving to {filename}: {e}")
        except Exception as e:
            self.logger.error(f" General error saving to {filename}: {e}")

async def main():
    # Setup logging for the main execution context
    logger = setup_logging()

    # Configuration
    INPUT_CSV = "SAT_questions_with_figures.csv"  # Your input CSV file
    TASKS_PER_DIFFICULTY = 100    # Number of tasks to generate per difficulty level (100 each for Easy, Medium, Hard = 300 total per skill)
    TEST_MODE = False             # Set to True to generate only 10 tasks for testing
    TEST_TASKS = 10              # Number of tasks in test mode
    PROJECT_ID = "studyhall-dev-383420"  # Your Google Cloud project ID
    MAX_WORKERS = 2              # Reduced concurrent workers to avoid API rate limits
    BATCH_SIZE = 5               # Number of skills to process concurrently (reduced to avoid rate limits)
    
    logger.info("=== SAT Math Task Generator with Figures (Difficulty-Based) ===")
    logger.info(f"Input file: {INPUT_CSV}")
    logger.info(f"Tasks per difficulty: {TASKS_PER_DIFFICULTY} × 3 difficulties = {TASKS_PER_DIFFICULTY * 3} total per skill")
    logger.info(f"Test mode: {'Enabled - ' + str(TEST_TASKS) + ' tasks per skill' if TEST_MODE else 'Disabled'}")
    logger.info(f"Project ID: {PROJECT_ID}")
    logger.info(f"Concurrent workers: {MAX_WORKERS}")
    logger.info(f"Skills batch size: {BATCH_SIZE}")
    
    generator = None  # Initialize generator to ensure it's always defined
    # Initialize generator with robust error handling
    try:
        generator = ConcurrentMathTaskGenerator(project_id=PROJECT_ID, max_workers=MAX_WORKERS)
        logger.info("✅ Successfully initialized Concurrent Gemini API with Vertex AI")
    except ImportError as e:
        logger.info("\n" + "="*80)
        logger.info(f"❌ Error importing required modules: {e}")
        logger.info("\nPlease install the required packages using:")
        logger.info("\n    pip install google-generativeai")
        logger.info("\nIf you continue to see this error after installation, ensure you have:")
        logger.info("1. Properly set up Google Cloud credentials")
        logger.info("2. Run 'gcloud auth application-default login'")
        logger.info("3. Enabled the Vertex AI API in your Google Cloud project")
        logger.info("\nFor more information, visit: https://cloud.google.com/vertex-ai/docs/start/client-libraries")
        logger.info("="*80 + "\n")
        return
    except Exception as e:
        logger.error(f"❌ Error initializing Gemini API: {e}")
        logger.error("Make sure you have run 'gcloud auth application-default login'")
        logger.error("And ensure your project has Vertex AI API enabled")
        return
    
    # Process all skills with improved robustness
    start_time = time.time()
    try:
        await generator.process_all_skills(
            input_csv=INPUT_CSV,
            tasks_per_difficulty=TASKS_PER_DIFFICULTY,  # 100 per difficulty level
            test_mode=TEST_MODE,
            test_tasks=TEST_TASKS,
            only_remaining=True,   # Set to False to regenerate all skills
            batch_size=BATCH_SIZE  # Process this many skills concurrently
        )
        logger.info("\n✅ Generation completed successfully!")
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Process interrupted by user. Partial results may have been saved.")
    except Exception as e:
        logger.error(f"\n❌ An error occurred during regeneration: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Print error statistics
        total_time = time.time() - start_time
        logger.info("\n=== Gemini API Error Statistics ===")
        if generator and hasattr(generator, 'error_counters') and generator.error_counters:
            total_errors = sum(count for err_type, count in generator.error_counters.items() if err_type != "success")
            total_success = generator.error_counters.get("success", 0)
            
            logger.info(f" Total API calls: {total_errors + total_success}")
            logger.info(f" Successful calls: {total_success}")
            logger.error(f" Failed calls: {total_errors}")
            
            if total_errors > 0:
                logger.info("\nError breakdown:")
                for err_type, count in sorted(generator.error_counters.items(), key=lambda x: x[1], reverse=True):
                    if err_type != "success":
                        logger.info(f" • {err_type}: {count} occurrences")
                        
            logger.info(f"\nRuntime: {total_time:.1f}s")
            
            if "RetryError" in generator.error_counters or "ClientError" in generator.error_counters:
                logger.warning("\n⚠️ Notable API issues detected ⚠️")
                logger.warning("Recommendations:")
                logger.warning("1. Reduce MAX_WORKERS and BATCH_SIZE even further")
                logger.warning("2. Increase request delays (self.request_delay in the generator) to 5-10 seconds")
                logger.warning("3. Add longer pauses between batches")
                logger.warning("4. Check your Vertex AI API quotas in the Google Cloud Console")
                logger.warning("5. Consider running at non-peak hours or spreading generation across multiple days")
        
        logger.info("\n=== Concurrent Generation Process Finished ===")


if __name__ == "__main__":
    asyncio.run(main())