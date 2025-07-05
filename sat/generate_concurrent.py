import csv
import os
import json
import random
import re
import time
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
from google import genai
from google.genai import types


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup comprehensive logging with file output"""
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup main logger
    logger = logging.getLogger('SAT_Generator')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler for detailed logs (DEBUG and above)
    detailed_log_file = Path(log_dir) / f"generation_detailed_{timestamp}.log"
    file_handler = logging.FileHandler(detailed_log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # File handler for errors only
    error_log_file = Path(log_dir) / f"generation_errors_{timestamp}.log"
    error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    # File handler for progress tracking (INFO and above)
    progress_log_file = Path(log_dir) / f"generation_progress_{timestamp}.log"
    progress_handler = logging.FileHandler(progress_log_file, encoding='utf-8')
    progress_handler.setLevel(logging.INFO)
    progress_handler.setFormatter(simple_formatter)
    logger.addHandler(progress_handler)
    
    # Console handler (WARNING and above to reduce console noise)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Log the setup
    logger.info("="*80)
    logger.info("SAT Math Task Generator - Logging Initialized")
    logger.info(f"Detailed logs: {detailed_log_file}")
    logger.info(f"Error logs: {error_log_file}")
    logger.info(f"Progress logs: {progress_log_file}")
    logger.info("="*80)
    
    return logger


class ConcurrentMathTaskGenerator:
    def __init__(self, project_id: str = "studyhall-dev-383420", max_workers: int = 2):
        """Initialize the Gemini API client with Vertex AI and concurrency settings"""
        self.project_id = project_id
        self.location = "global"
        self.model = "gemini-2.5-flash"
        self.max_workers = max_workers
        
        # Setup logging
        self.logger = setup_logging()
        self.logger.info(f"Initializing SAT Math Task Generator")
        self.logger.info(f"Project ID: {project_id}")
        self.logger.info(f"Model: {self.model}")
        self.logger.info(f"Max workers: {max_workers}")
        
        # Thread-local storage for API clients (since Gemini client might not be thread-safe)
        self._local = threading.local()
        
        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(max_workers)
        self.request_delay = 1.0  # Delay between requests to avoid hitting rate limits
        
        # Progress tracking
        self.total_skills = 0
        self.completed_skills = 0
        self.lock = threading.Lock()
    
    def get_client(self):
        """Get or create a thread-local Gemini client"""
        if not hasattr(self._local, 'client'):
            self._local.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
            )
        return self._local.client
    
    def read_tasks_by_skill(self, csv_file: str) -> Dict[str, List[Dict[str, Any]]]:
        """Read tasks from CSV and group them by skill, keeping only needed columns"""
        self.logger.info(f"Reading tasks from CSV file: {csv_file}")
        tasks_by_skill = defaultdict(list)
        
        # Define the columns we actually need for generation
        needed_columns = {
            'test', 'domain', 'skill', 'difficulty', 'question_text_latex',
            'option_A_latex', 'option_B_latex', 'option_C_latex', 'option_D_latex',
            'correct_answer', 'correct_answer_spr_latex'
        }
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                # Check which needed columns are available in the CSV
                available_columns = set(reader.fieldnames or [])
                missing_columns = needed_columns - available_columns
                unused_columns = available_columns - needed_columns
                
                if missing_columns:
                    warning_msg = f"Missing expected columns: {sorted(missing_columns)}"
                    self.logger.warning(warning_msg)
                
                if unused_columns:
                    info_msg = f"Filtering out unused columns: {sorted(unused_columns)}"
                    self.logger.info(info_msg)
                
                for row in reader:
                    skill = row.get('skill', '').strip()
                    if skill:
                        # Create a filtered row with only the needed columns
                        filtered_row = {col: row.get(col, '') for col in needed_columns if col in available_columns}
                        tasks_by_skill[skill].append(filtered_row)
            
            success_msg = f"Successfully read tasks from {csv_file}"
            self.logger.info(success_msg)
            
            for skill, tasks in tasks_by_skill.items():
                skill_info = f"Skill loaded - {skill}: {len(tasks)} tasks"
                self.logger.info(skill_info)
                
        except FileNotFoundError:
            error_msg = f"CSV file {csv_file} not found"
            self.logger.error(error_msg)
        except Exception as e:
            error_msg = f"Error reading CSV file: {e}"
            self.logger.error(error_msg)
        
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
            return f'"{content}"' # <-- FIX HERE

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
            
            # <<< FIX #1: RETURN THE FIXED CONTENT, WRAPPED IN QUOTES >>>
            return f'"{content}"'
        
        # Apply fixes to content within JSON strings
        # <<< FIX #2: USE A MORE ROBUST REGEX TO HANDLE ESCAPED QUOTES (\") >>>
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
    
    def prepare_examples_for_prompt(self, tasks: List[Dict[str, Any]], num_examples: int = 25) -> List[Dict[str, Any]]:
        """Prepare example tasks for the prompt, cleaning math content"""
        # Group tasks by difficulty
        difficulties = {}
        for task in tasks:
            difficulty = task.get('difficulty', '').strip()
            if difficulty:
                if difficulty not in difficulties:
                    difficulties[difficulty] = []
                difficulties[difficulty].append(task)
        
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
        """Extract only the question text (with LaTeX) from existing generated tasks"""
        if not tasks:
            return []
        
        question_texts = []
        for task in tasks:
            question_text = task.get('question_text_latex', '')
            if question_text:
                question_texts.append(question_text)
        
        return question_texts
    
    def create_generation_prompt(self, examples: List[Dict[str, Any]], skill: str, 
                                existing_generated: List[Dict[str, Any]] = None, num_to_generate: int = 10) -> str:
        """Create a detailed prompt for generating math tasks"""
        
        # Analyze examples to understand patterns
        difficulties = set(task.get('difficulty', '').strip() for task in examples)
        test_name = examples[0].get('test', 'Math') if examples else 'Math'
        domain = examples[0].get('domain', 'Algebra') if examples else 'Algebra'
        
        prompt = f"""You are an expert math educator creating high-quality {skill} problems. 

TASK: Generate {num_to_generate} new math problems for the skill "{skill}" that match the style and difficulty of the provided examples.

EXAMPLES FROM EXISTING DATASET:
{json.dumps(examples, indent=2)}

REQUIREMENTS:
1. **Maintain Consistency**: Follow the exact same format and structure as the examples
2. **Difficulty Distribution**: Include problems across difficulties: {', '.join(difficulties)}
3. **Mathematical Accuracy**: Ensure all calculations are correct
4. **MathJax Format**: Use proper LaTeX/MathJax syntax (e.g., $x^2$, $\\frac{{a}}{{b}}$, $\\sqrt{{x}}$)
5. **Complete Structure**: Include all required fields:
   - test: "{test_name}"
   - domain: "{domain}"
   - skill: "{skill}"
   - difficulty: Choose from {list(difficulties)}
   - question_text_latex: The main question with proper LaTeX
   - option_A_latex, option_B_latex, option_C_latex, option_D_latex: Multiple choice options (if applicable, include tables in MathJax syntax if needed)
   - correct_answer: The correct answer (A, B, C, D, or direct answer)
   - correct_answer_spr_latex: The correct answer in LaTeX format
   - step_1, step_2, step_3, step_4, step_5, step_6: Individual explanation steps (UP TO 6 STEPS MAXIMUM)

6. **Variety**: Create diverse problems within the skill area
7. **Avoid Common Issues**: 
   - Don't use [Math: ...] notation
   - Use proper LaTeX delimiters ($...$)
   - Ensure fractions use \\frac{{numerator}}{{denominator}}
   - Use \\left( and \\right) for large parentheses when needed

8. **Tables and Data**: 
   - If problems involve tables or data, include them in answer options using MathJax syntax
   - Format tables using \\begin{{array}} and \\end{{array}} with proper alignment
   - Example: $\\begin{{array}}{{|c|c|}} \\hline x & y \\\\ \\hline 1 & 2 \\\\ 3 & 4 \\\\ \\hline \\end{{array}}$

9. **Step Formatting**:
   - Each step should be a complete, standalone explanation
   - Use clear mathematical notation in each step
   - Each step should logically lead to the next
   - Fill only the necessary steps (can be fewer than 6)

10. **Problem Types**: Mix different types of problems within the skill:
   - Some with multiple choice options (A, B, C, D)
   - Some with direct numerical answers
   - Include word problems and abstract algebraic problems
   - Vary complexity within difficulty levels
   
11. **Ignore Figures Problems**:
   - If examples include problems with figures, ignore them for this task
   - Focus on examples that are purely text-based math problems"""

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
After you have finished thinking, your response MUST ONLY BE the final, clean, valid JSON array containing exactly {num_to_generate} problems. Do not include any of your thought process, notes, or any other text before or after the JSON array. The response must start with `[` and end with `]`.

THINKING AND OUTPUT FORMAT:
You may think through the problem creation process before generating the JSON. Feel free to:
- Analyze the skill requirements
- Consider difficulty distribution
- Plan problem types and variations
- Think about mathematical concepts to cover

However, after your thinking, you MUST provide a valid JSON array in the EXACT format specified below.

**IMPORTANT**: The JSON array must be complete, valid, and contain exactly {num_to_generate} problems.

OUTPUT FORMAT (after any thinking/analysis):

[
  {{
    "test": "{test_name}",
    "domain": "{domain}",
    "skill": "{skill}",
    "difficulty": "Easy|Medium|Hard",
    "question_text_latex": "Question with proper $LaTeX$ formatting",
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

CRITICAL JSON REQUIREMENTS: 
- The JSON array must be syntactically perfect and parseable
- Use separate step_1, step_2, etc. fields for each explanation step
- Each step should be complete and standalone
- Use UP TO 6 STEPS MAXIMUM (can be fewer, leave unused steps empty or null)
- Include tables in answer options using MathJax array syntax when needed

CRITICAL LATEX ESCAPING IN JSON:
- EVERY backslash in LaTeX must be doubled for JSON: \\\\ instead of \\
- Examples of correct escaping:
  * \\frac{{1}}{{2}} → \\\\frac{{1}}{{2}}
  * \\sqrt{{x}} → \\\\sqrt{{x}}
  * \\begin{{array}} → \\\\begin{{array}}
  * \\\\ (line break) → \\\\\\\\
  * \\$ → \\\\$
- The JSON parser will fail if LaTeX backslashes are not properly escaped
- Test your JSON by ensuring it can be parsed by standard JSON parsers

- Do not include any text after the closing ] bracket
- The JSON must contain exactly {num_to_generate} complete problem objects


FINAL INSTRUCTION:
First, complete all your internal thinking, analysis, and problem creation steps.
After you have finished thinking, provide your response which MUST ONLY BE the final, clean, valid JSON array containing exactly {num_to_generate} problems. Do not include any of your thought process, notes, or any other text before or after the JSON array. The response must start with `[` and end with `]`."""
        
        return prompt
    
    async def generate_batch_with_backoff(self, prompt: str, skill: str, max_retries: int = 5, 
                                        base_delay: float = 1.0) -> Tuple[Optional[List[Dict[str, Any]]], int, int, int]:
        """Generate a batch of tasks with exponential backoff and rate limiting"""
        self.logger.info(f"Starting batch generation for skill: {skill}")
        
        async with self.rate_limiter:
            # Add delay between requests
            await asyncio.sleep(self.request_delay)
            
            # Track errors for summary
            error_counts = {}
            
            for attempt in range(max_retries):
                try:
                    attempt_msg = f"Generating batch (attempt {attempt + 1}/{max_retries}) for skill: {skill}"
                    self.logger.debug(attempt_msg)
                    
                    # Get thread-local client
                    client = self.get_client()
                    
                    # Prepare content for the new API
                    contents = [
                        types.Content(
                            role="user",
                            parts=[types.Part(text=prompt)]
                        )
                    ]
                    
                    # Configure generation settings
                    generate_content_config = types.GenerateContentConfig(
                        temperature=1,
                        top_p=1,
                        max_output_tokens=65535,
                        safety_settings=[
                            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
                        ]
                    )
                    
                    # Generate content using streaming (but collect all chunks)
                    response_text = ""
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
                        except Exception as stream_error:
                            error_msg = f"API Stream Error: {type(stream_error).__name__}: {stream_error}"
                            self.logger.error(f"[{skill}] {error_msg}")
                            # Detailed analysis of streaming errors
                            if "RetryError" in str(type(stream_error).__name__):
                                self.logger.error(f"[{skill}] RetryError Details detected")
                                if hasattr(stream_error, 'last_attempt') and stream_error.last_attempt:
                                    if hasattr(stream_error.last_attempt, 'exception'):
                                        underlying = stream_error.last_attempt.exception
                                        self.logger.error(f"[{skill}] Stream underlying error: {type(underlying).__name__}: {underlying}")
                                        if hasattr(underlying, 'code'):
                                            self.logger.error(f"[{skill}] Stream error code: {underlying.code}")
                                        if hasattr(underlying, 'message'):
                                            self.logger.error(f"[{skill}] Stream error message: {underlying.message}")
                            # Re-raise to be caught by the outer exception handler
                            raise
                    
                    # Execute the API call asynchronously
                    await loop.run_in_executor(None, generate_content)
                    
                    response_text = response_text.strip()
                    
                    # Log token usage information
                    if total_tokens > 0:
                        token_msg = f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}"
                        self.logger.info(f"[{skill}] {token_msg}")
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
                    
                    # Process and clean generated tasks
                    processed_tasks = self.process_generated_tasks(new_tasks)
                    
                    success_msg = f"Successfully generated {len(processed_tasks)} tasks"
                    self.logger.info(f"[{skill}] {success_msg}")
                    return processed_tasks, input_tokens, output_tokens, total_tokens
                    
                except json.JSONDecodeError as e:
                    error_msg = f"JSON parsing error (attempt {attempt + 1}): {e}"
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
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                    
                except ValueError as e:
                    error_msg = f"Value error (attempt {attempt + 1}): {e}"
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
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                    
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
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    self.logger.info(f"[{skill}] Waiting {delay:.1f} seconds before retry...")
                    await asyncio.sleep(delay)
            
            # If we get here, all attempts failed
            self.logger.error(f"[{skill}] All {max_retries} attempts failed")
            
            # Log error summary
            if error_counts:
                self.logger.error(f"[{skill}] Error summary after {max_retries} attempts:")
                for error_type, count in error_counts.items():
                    self.logger.error(f"[{skill}]   - {error_type}: {count} occurrences")
            
            return None, 0, 0, 0
    
    async def generate_tasks_for_skill(self, skill: str, examples: List[Dict[str, Any]], 
                                     total_tasks: int = 100, batch_size: int = 10) -> List[Dict[str, Any]]:
        """Generate tasks for a specific skill in batches"""
        all_generated = []
        examples_for_prompt = self.prepare_examples_for_prompt(examples, 15)
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens_used = 0
        
        self.logger.info(f"[{skill}] Starting generation of {total_tasks} tasks")
        self.logger.info(f"[{skill}] Available examples: {len(examples)}, Using: {len(examples_for_prompt)} examples")
        
        num_batches = (total_tasks + batch_size - 1) // batch_size
        
        for batch_num in range(num_batches):
            remaining_tasks = total_tasks - len(all_generated)
            current_batch_size = min(batch_size, remaining_tasks)
            
            self.logger.debug(f"[{skill}] Batch {batch_num + 1}/{num_batches}: Generating {current_batch_size} tasks...")
            
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
                self.logger.info(f"[{skill}] Total generated so far: {len(all_generated)}/{total_tasks}")
            else:
                self.logger.error(f"[{skill}] Failed to generate batch {batch_num + 1}")
                break
            
            # Small delay between batches
            if batch_num < num_batches - 1:
                await asyncio.sleep(2)
        
        self.logger.info(f"[{skill}] Completed generation: {len(all_generated)} tasks")
        self.logger.info(f"[{skill}] Total token usage: Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_tokens_used}")
        
        # Update progress
        with self.lock:
            self.completed_skills += 1
            self.logger.info(f"Progress: {self.completed_skills}/{self.total_skills} skills completed")
        
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
                'correct_answer', 'correct_answer_spr_latex',
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
                                                   tasks_per_difficulty: int = 100, batch_size: int = 10) -> List[Dict[str, Any]]:
        """Generate tasks for a specific skill with equal distribution across difficulty levels"""
        
        # Group examples by difficulty
        examples_by_difficulty = {}
        for example in examples:
            difficulty = example.get('difficulty', '').strip()
            if difficulty:
                if difficulty not in examples_by_difficulty:
                    examples_by_difficulty[difficulty] = []
                examples_by_difficulty[difficulty].append(example)
        
        self.logger.info(f"[{skill}] Starting generation of {tasks_per_difficulty} tasks per difficulty level")
        self.logger.info(f"[{skill}] Available difficulties: {list(examples_by_difficulty.keys())}")
        for difficulty, examples_list in examples_by_difficulty.items():
            self.logger.info(f"[{skill}] {difficulty}: {len(examples_list)} examples")
        
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
                self.logger.warning(f"[{skill}] Skipping {difficulty}: only {len(difficulty_examples)} examples (need at least 3)")
                continue
            
            self.logger.info(f"[{skill}] Generating {tasks_per_difficulty} tasks for difficulty: {difficulty}")
            
            # Prepare examples for this difficulty
            examples_for_prompt = self.prepare_examples_for_prompt(difficulty_examples, min(15, len(difficulty_examples)))
            
            # Generate tasks in batches for this difficulty
            difficulty_generated = []
            num_batches = (tasks_per_difficulty + batch_size - 1) // batch_size
            
            for batch_num in range(num_batches):
                remaining_tasks = tasks_per_difficulty - len(difficulty_generated)
                current_batch_size = min(batch_size, remaining_tasks)
                
                self.logger.debug(f"[{skill}] {difficulty} - Batch {batch_num + 1}/{num_batches}: Generating {current_batch_size} tasks...")
                
                prompt = self.create_difficulty_specific_prompt(
                    examples_for_prompt, 
                    skill, 
                    difficulty,
                    difficulty_generated,  # Pass existing generated tasks for this difficulty
                    current_batch_size
                )
                
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
                    
                    self.logger.info(f"[{skill}] {difficulty} - Generated so far: {len(difficulty_generated)}/{tasks_per_difficulty}")
                    
                    # Check if we should save incrementally (every 10 tasks)
                    if len(all_generated) - total_saved_count >= 10:
                        incremental_batch_count += 1
                        tasks_to_save = all_generated[total_saved_count:]
                        self.save_tasks_incrementally(tasks_to_save, skill, incremental_batch_count)
                        total_saved_count = len(all_generated)
                    
                else:
                    self.logger.error(f"[{skill}] {difficulty} - Failed to generate batch {batch_num + 1}")
                    break
                
                # Small delay between batches
                if batch_num < num_batches - 1:
                    await asyncio.sleep(2)
            
            self.logger.info(f"[{skill}] {difficulty} - Completed: {len(difficulty_generated)} tasks")
        
        # Save any remaining tasks that haven't been saved incrementally
        if len(all_generated) > total_saved_count:
            incremental_batch_count += 1
            remaining_tasks = all_generated[total_saved_count:]
            self.save_tasks_incrementally(remaining_tasks, skill, incremental_batch_count)
            self.logger.info(f"💾 Saved final {len(remaining_tasks)} remaining tasks for {skill}")
        
        self.logger.info(f"[{skill}] TOTAL COMPLETED: {len(all_generated)} tasks across all difficulties")
        self.logger.info(f"[{skill}] Total token usage: Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_tokens_used}")
        self.logger.info(f"💾 Total incremental saves: {incremental_batch_count} batches")
        
        # Update progress
        with self.lock:
            self.completed_skills += 1
            self.logger.info(f"Progress: {self.completed_skills}/{self.total_skills} skills completed")
        
        return all_generated

    def create_difficulty_specific_prompt(self, examples: List[Dict[str, Any]], skill: str, difficulty: str,
                                        existing_generated: List[Dict[str, Any]] = None, num_to_generate: int = 10) -> str:
        """Create a detailed prompt for generating math tasks for a specific difficulty level"""
        
        test_name = examples[0].get('test', 'Math') if examples else 'Math'
        domain = examples[0].get('domain', 'Algebra') if examples else 'Algebra'
        
        prompt = f"""You are an expert math educator creating high-quality {skill} problems. 

TASK: Generate {num_to_generate} new math problems for the skill "{skill}" at {difficulty} difficulty level that match the style and structure of the provided examples.

EXAMPLES FROM EXISTING DATASET (ALL {difficulty.upper()} DIFFICULTY):
{json.dumps(examples, indent=2)}

REQUIREMENTS:
1. **Maintain Consistency**: Follow the exact same format and structure as the examples
2. **Specific Difficulty**: ALL problems must be {difficulty} difficulty level - no other difficulties allowed
3. **Mathematical Accuracy**: Ensure all calculations are correct
4. **Complete Structure**: Include all required fields with proper difficulty: "{difficulty}"
5. **{difficulty} Difficulty Level**: Ensure complexity matches the difficulty level exactly"""

        if existing_generated:
            question_texts = self.extract_question_texts_only(existing_generated)
            if question_texts:
                prompt += f"""

AVOID REPETITION: You have already generated {difficulty} problems with the following question texts. DO NOT create similar problems:
"""
                for i, question_text in enumerate(question_texts, 1):
                    prompt += f"{i}. {question_text}\n"
                prompt += "\n"

        prompt += f"""

OUTPUT FORMAT: Return ONLY a valid JSON array of {num_to_generate} problems, no additional text:

[
  {{
    "test": "{test_name}",
    "domain": "{domain}",
    "skill": "{skill}",
    "difficulty": "{difficulty}",
    "question_text_latex": "Question with proper $LaTeX$ formatting",
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
- Use separate step_1, step_2, etc. fields for each explanation step
- Each step should be complete and standalone
- Fill UP TO 6 STEPS MAXIMUM (can be fewer, leave unused steps empty or null)
- IMPORTANT: In JSON, escape backslashes properly (use \\\\ for LaTeX \\, \\$ for \$, etc.)
- Ensure all LaTeX expressions are properly escaped for valid JSON format"""
        
        return prompt
    

    
    async def generate_additional_tasks_by_difficulty(self, skill: str, examples: List[Dict[str, Any]],
                                           tasks_to_generate: Dict[str, int], batch_size: int = 10) -> List[Dict[str, Any]]:
        """Generate additional tasks for specific difficulty levels"""
        
        # Group examples by difficulty
        examples_by_difficulty = {}
        for example in examples:
            difficulty = example.get('difficulty', '').strip()
            if difficulty:
                if difficulty not in examples_by_difficulty:
                    examples_by_difficulty[difficulty] = []
                examples_by_difficulty[difficulty].append(example)
        
        all_generated = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens_used = 0
        
        # Counter for incremental saving
        total_saved_count = 0
        incremental_batch_count = 0
        
        # Generate tasks only for difficulties that need more tasks
        for difficulty, target_count in tasks_to_generate.items():
            if target_count <= 0:  # Skip if no additional tasks needed
                self.logger.info(f"[{skill}] No additional {difficulty} tasks needed")
                continue
                
            if difficulty not in examples_by_difficulty:
                self.logger.warning(f"[{skill}] No examples for {difficulty} difficulty, skipping")
                continue
                
            if len(examples_by_difficulty[difficulty]) < 3:  # Need minimum examples
                self.logger.warning(f"[{skill}] Not enough examples for {difficulty} difficulty, skipping")
                continue
            
            self.logger.info(f"[{skill}] Generating {target_count} additional tasks for difficulty: {difficulty}")
            
            # Prepare examples for this difficulty
            examples_for_prompt = self.prepare_examples_for_prompt(
                examples_by_difficulty[difficulty], 
                min(15, len(examples_by_difficulty[difficulty]))
            )
            
            # Generate tasks in batches for this difficulty
            difficulty_generated = []
            num_batches = (target_count + batch_size - 1) // batch_size
            
            for batch_num in range(num_batches):
                remaining_tasks = target_count - len(difficulty_generated)
                current_batch_size = min(batch_size, remaining_tasks)
                
                self.logger.debug(f"[{skill}] {difficulty} - Batch {batch_num + 1}/{num_batches}: Generating {current_batch_size} tasks...")
                
                prompt = self.create_difficulty_specific_prompt(
                    examples_for_prompt, 
                    skill, 
                    difficulty,
                    difficulty_generated,  # Pass existing generated tasks for this difficulty
                    current_batch_size
                )
                
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
                    
                    self.logger.info(f"[{skill}] {difficulty} - Generated so far: {len(difficulty_generated)}/{target_count}")
                    
                    # Check if we should save incrementally (every 10 tasks)
                    if len(all_generated) - total_saved_count >= 10:
                        incremental_batch_count += 1
                        tasks_to_save = all_generated[total_saved_count:]
                        self.save_tasks_incrementally(tasks_to_save, skill, incremental_batch_count)
                        total_saved_count = len(all_generated)
                    
                else:
                    self.logger.error(f"[{skill}] {difficulty} - Failed to generate batch {batch_num + 1}")
                    break
                
                # Small delay between batches
                if batch_num < num_batches - 1:
                    await asyncio.sleep(2)
            
            self.logger.info(f"[{skill}] {difficulty} - Completed: {len(difficulty_generated)} tasks")
        
        # Save any remaining tasks that haven't been saved incrementally
        if len(all_generated) > total_saved_count:
            incremental_batch_count += 1
            remaining_tasks = all_generated[total_saved_count:]
            self.save_tasks_incrementally(remaining_tasks, skill, incremental_batch_count)
            self.logger.info(f"💾 Saved final {len(remaining_tasks)} remaining tasks for {skill}")
        
        self.logger.info(f"[{skill}] TOTAL ADDITIONAL TASKS: {len(all_generated)}")
        self.logger.info(f"[{skill}] Total token usage: Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_tokens_used}")
        self.logger.info(f"💾 Total incremental saves: {incremental_batch_count} batches")
        
        # Update progress
        with self.lock:
            self.completed_skills += 1
            self.logger.info(f"Progress: {self.completed_skills}/{self.total_skills} skills completed")
        
        return all_generated

    def save_tasks_to_csv(self, tasks: List[Dict[str, Any]], filename: str):
        """Save tasks to CSV file"""
        if not tasks:
            warning_msg = f"No tasks to save for {filename}"
            self.logger.warning(warning_msg)
            return
        
        self.logger.info(f"Saving {len(tasks)} tasks to {filename}")
        
        # Define the column order
        columns = [
            'test', 'domain', 'skill', 'difficulty', 'question_text_latex',
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
            success_msg = f"Saved {len(tasks)} tasks to {filename}"
            self.logger.info(success_msg)
        except ValueError as e:
            if "fields not in fieldnames" in str(e):
                error_msg = f"CSV Writing Error: Extra fields detected in tasks for {filename}"
                self.logger.error(error_msg)
                self.logger.error(f"ValueError details: {e}")
                self.logger.debug("Full content of problematic tasks:")
                
                # Find and print the complete content of tasks with extra fields
                problematic_count = 0
                for i, task in enumerate(tasks):
                    extra_fields = set(task.keys()) - set(columns)
                    if extra_fields:
                        problematic_count += 1
                        task_info = f"PROBLEMATIC TASK #{i + 1}: Extra fields: {sorted(extra_fields)}"
                        self.logger.error(task_info)
                        self.logger.debug(f"FULL TASK CONTENT:")
                        
                        # Print all fields in the task for complete debugging info
                        for key, value in task.items():
                            # Truncate very long values for readability but keep essential info
                            if isinstance(value, str) and len(value) > 200:
                                truncated_value = value[:200] + "... [TRUNCATED]"
                            else:
                                truncated_value = value
                            
                            # Highlight extra fields
                            marker = "EXTRA → " if key in extra_fields else "       "
                            self.logger.debug(f"{marker}{key}: {repr(truncated_value)}")
                
                summary_msg = f"Found {problematic_count} tasks with extra fields out of {len(tasks)} total tasks"
                self.logger.error(summary_msg)
                self.logger.error("This error should not occur after validation - please check the validation logic")
            else:
                error_msg = f"ValueError saving to {filename}: {e}"
                self.logger.error(error_msg)
        except Exception as e:
            error_msg = f"General error saving to {filename}: {e}"
            self.logger.error(error_msg)
    
    def get_already_generated_skills(self) -> set:
        """Get list of skills that have already been generated"""
        import glob
        
        # Track skills that have already been generated with sufficient questions
        generated_skills = set()
        # Track skills that need additional questions
        incomplete_skills = set()
        
        generated_files = glob.glob("generated_*.csv")
        
        for file in generated_files:
            try:
                # Count tasks by difficulty for this file
                tasks_by_difficulty = defaultdict(int)
                skill_name = None
                
                with open(file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        if i == 0 and 'skill' in row:
                            skill_name = row['skill'].strip()
                        
                        difficulty = row.get('difficulty', '').strip()
                        if difficulty:
                            tasks_by_difficulty[difficulty] += 1
                
                if skill_name:
                    # Consider a skill "complete" if it has at least 90 questions per difficulty level
                    # Otherwise mark it as "incomplete" so we can generate more questions
                    if all(tasks_by_difficulty.get(diff, 0) >= 90 for diff in ["Easy", "Medium", "Hard"]):
                        generated_skills.add(skill_name)
                        self.logger.info(f"Complete skill: {skill_name} ({dict(tasks_by_difficulty)})")
                    else:
                        incomplete_skills.add(skill_name)
                        self.logger.info(f"Incomplete skill: {skill_name} ({dict(tasks_by_difficulty)})")
            except Exception as e:
                self.logger.warning(f"Could not read {file}: {e}")
        
        # For backwards compatibility, return just the complete skills
        # The incomplete skills will be handled by check_existing_tasks_by_difficulty
        return generated_skills
    
    async def process_skill_batch(self, skill_batch: List[Tuple[str, List[Dict[str, Any]]]], 
                                tasks_per_difficulty: int, test_mode: bool, test_tasks: int):
        """Process a batch of skills concurrently using difficulty-based generation"""
        async def process_single_skill(skill, examples):
            try:
                # Check existing tasks to determine how many more to generate per difficulty
                if test_mode:
                    # In test mode, generate fewer tasks per difficulty
                    target_tasks_per_difficulty = max(1, test_tasks // 3)  # Divide test tasks across difficulties
                    # Just generate test tasks without checking existing files
                    tasks_to_generate = {"Easy": target_tasks_per_difficulty, 
                                        "Medium": target_tasks_per_difficulty, 
                                        "Hard": target_tasks_per_difficulty}
                    generated_tasks = await self.generate_tasks_for_skill_by_difficulty(
                        skill, examples, target_tasks_per_difficulty
                    )
                else:
                    # In normal mode, check existing tasks and generate additional ones if needed
                    tasks_to_generate = self.check_existing_tasks_by_difficulty(skill)
                    
                    # If all difficulties have enough tasks, no need to generate more
                    if all(count <= 0 for count in tasks_to_generate.values()):
                        self.logger.info(f"[{skill}] All difficulties have sufficient tasks (≥90), skipping generation")
                        return skill, 0
                    
                    # Generate new tasks only for difficulties that need more
                    generated_tasks = await self.generate_additional_tasks_by_difficulty(
                        skill, examples, tasks_to_generate
                    )
                
                if generated_tasks:
                    # Tasks are already saved incrementally during generation
                    self.logger.info(f"✅ Completed generation for skill '{skill}' with {len(generated_tasks)} tasks (saved incrementally)")
                    return skill, len(generated_tasks)
                else:
                    return skill, 0
            except Exception as e:
                self.logger.error(f"Error processing skill '{skill}': {e}")
                return skill, 0
        
        # Process all skills in the batch concurrently
        tasks = [process_single_skill(skill, examples) for skill, examples in skill_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results

    async def process_all_skills(self, input_csv: str, tasks_per_difficulty: int = 100, 
                               test_mode: bool = False, test_tasks: int = 10, 
                               only_remaining: bool = True, batch_size: int = None):
        """Process all skills from the input CSV concurrently using difficulty-based generation"""
        self.logger.info("=== Concurrent Math Task Generator ===")
        self.logger.info(f"Input file: {input_csv}")
        self.logger.info(f"Tasks per difficulty level: {tasks_per_difficulty}")
        self.logger.info(f"Total tasks per skill: {tasks_per_difficulty * 3} (assuming Easy, Medium, Hard)")
        if test_mode:
            self.logger.info(f"Test mode: {test_mode} ({test_tasks} total tasks per skill)")
        else:
            self.logger.info(f"Test mode: {test_mode}")
        self.logger.info(f"Only remaining skills: {only_remaining}")
        self.logger.info(f"Max concurrent workers: {self.max_workers}")
        
        # Read tasks by skill
        tasks_by_skill = self.read_tasks_by_skill(input_csv)
        
        if not tasks_by_skill:
            self.logger.error("No tasks found in input file")
            return
        
        # Get already generated skills if requested
        already_generated = set()
        if only_remaining:
            already_generated = self.get_already_generated_skills()
            self.logger.info(f"Found {len(already_generated)} already generated skills")
            if already_generated:
                self.logger.info("Already generated skills:")
                for skill in sorted(already_generated):
                    self.logger.info(f"  ✓ {skill}")
        
        # Filter skills to process
        skills_to_process = []
        for skill, examples in tasks_by_skill.items():
            if only_remaining and skill in already_generated:
                self.logger.info(f"Skipping {skill}: already generated")
                continue
            
            if len(examples) < 5:  # Need minimum examples
                self.logger.warning(f"Skipping {skill}: only {len(examples)} examples (need at least 5)")
                continue
            
            skills_to_process.append((skill, examples))
        
        if not skills_to_process:
            self.logger.info("No remaining skills to generate! All skills are already done.")
            return
        
        self.total_skills = len(skills_to_process)
        self.completed_skills = 0
        
        self.logger.info(f"Will generate tasks for {len(skills_to_process)} skill(s) using {self.max_workers} concurrent workers:")
        for skill, _ in skills_to_process:
            self.logger.info(f"  🔄 {skill}")
        
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
            
            batch_info = f"Processing batch {batch_num}/{total_batches} ({len(batch)} skills): {', '.join([skill for skill, _ in batch])}"
            self.logger.info(batch_info)
            
            batch_start = time.time()
            results = await self.process_skill_batch(batch, tasks_per_difficulty, test_mode, test_tasks)
            batch_time = time.time() - batch_start
            
            # Count successful generations in this batch
            batch_generated = sum(count for skill, count in results if isinstance(count, int))
            total_generated += batch_generated
            
            batch_result = f"Batch {batch_num} completed in {batch_time:.1f}s, generated {batch_generated} tasks"
            self.logger.info(batch_result)
        
        total_time = time.time() - start_time
        
        # Final summary
        self.logger.info("All skills processed!")
        self.logger.info(f"Total time: {total_time:.1f}s")
        self.logger.info(f"Total tasks generated: {total_generated}")
        self.logger.info(f"Average time per skill: {total_time / len(skills_to_process):.1f}s")
        self.logger.info(f"Skills processed: {self.completed_skills}/{self.total_skills}")
    
    def process_generated_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean generated tasks before saving"""
        self.logger.debug(f"Processing {len(tasks)} generated tasks")
        processed_tasks = []
        
        for i, task in enumerate(tasks):
            # Skip non-dictionary items (sometimes Gemini returns malformed JSON with integers, strings, etc.)
            if not isinstance(task, dict):
                warning_msg = f"Skipping non-dictionary item #{i+1}: {type(task).__name__}: {task}"
                self.logger.warning(warning_msg)
                
                # Log the full problematic item for debugging
                self.logger.error(f"=== PROBLEMATIC TASK ITEM #{i+1} ===")
                self.logger.error(f"Type: {type(task).__name__}")
                self.logger.error(f"Content: {repr(task)}")
                self.logger.error(f"Full item data: {task}")
                self.logger.error("-" * 50)
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
            required_fields = ["skill", "difficulty", "question_text_latex", "correct_answer"]
            missing_fields = [field for field in required_fields if not cleaned_task.get(field)]
            
            if missing_fields:
                warning_msg = f"Task #{i+1} missing required fields: {', '.join(missing_fields)}. Adding defaults."
                self.logger.warning(warning_msg)
                
                # Log the full problematic task for debugging
                self.logger.error(f"=== TASK WITH MISSING FIELDS #{i+1} ===")
                self.logger.error(f"Missing fields: {missing_fields}")
                self.logger.error(f"Available fields: {list(cleaned_task.keys())}")
                self.logger.error(f"Full task data:")
                self.logger.error("-" * 50)
                for key, value in cleaned_task.items():
                    # Truncate very long values for readability
                    if isinstance(value, str) and len(value) > 200:
                        truncated_value = value[:200] + "... [TRUNCATED]"
                    else:
                        truncated_value = value
                    self.logger.error(f"  {key}: {repr(truncated_value)}")
                self.logger.error("-" * 50)
                
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
            
            # Convert format issues in LaTeX fields
            for key in cleaned_task:
                if key.endswith('_latex') and isinstance(cleaned_task[key], str):
                    # Fix common LaTeX formatting issues
                    cleaned_task[key] = cleaned_task[key].replace('[Math: ', '$').replace(']', '$')
                    # Ensure proper MathJax delimiters
                    cleaned_task[key] = cleaned_task[key].replace('$$$', '$').replace('$$', '$')
            
            processed_tasks.append(cleaned_task)
        
        self.logger.info(f"Successfully processed {len(processed_tasks)} tasks from {len(tasks)} raw tasks")
        return processed_tasks

    def check_existing_tasks_by_difficulty(self, skill: str) -> Dict[str, int]:
        """
        Check existing generated tasks for a skill and return how many more tasks to generate per difficulty.
        Returns a dictionary with difficulty as key and number of tasks to generate as value.
        """
        import glob
        
        # Create a filename pattern based on the skill name
        safe_skill_name = "".join(c for c in skill if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_skill_name = safe_skill_name.replace(' ', '_')
        filename_pattern = f"generated_{safe_skill_name}.csv"
        
        # Look for existing files
        matching_files = glob.glob(filename_pattern)
        if not matching_files:
            # No existing file, need to generate all tasks
            return {"Easy": 100, "Medium": 100, "Hard": 100}
        
        # Count existing tasks by difficulty
        tasks_by_difficulty = defaultdict(int)
        try:
            with open(matching_files[0], 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    difficulty = row.get('difficulty', '').strip()
                    if difficulty:
                        tasks_by_difficulty[difficulty] += 1
        except Exception as e:
            self.logger.warning(f"Error reading {matching_files[0]}: {e}")
            return {"Easy": 100, "Medium": 100, "Hard": 100}
        
        # Calculate how many more tasks needed per difficulty
        tasks_to_generate = {}
        for difficulty in ["Easy", "Medium", "Hard"]:
            existing_count = tasks_by_difficulty.get(difficulty, 0)
            if existing_count < 90:  # If less than 90, generate more to reach 100
                tasks_to_generate[difficulty] = 100 - existing_count
            else:
                tasks_to_generate[difficulty] = 0
                
        self.logger.info(f"[{skill}] Existing tasks by difficulty: {dict(tasks_by_difficulty)}")
        self.logger.info(f"[{skill}] Additional tasks to generate: {tasks_to_generate}")
        
        return tasks_to_generate
    
async def main():
    # Configuration
    INPUT_CSV = "SAT Questions - SAT_Math_No_Graph.csv"  # Your input CSV file
    TASKS_PER_DIFFICULTY = 100    # Number of tasks to generate per difficulty level (100 each for Easy, Medium, Hard = 300 total per skill)
    TEST_MODE = False              # Set to True to generate only 10 tasks for testing
    TEST_TASKS = 10               # Number of tasks in test mode
    PROJECT_ID = "studyhall-dev-383420"  # Your Google Cloud project ID
    MAX_WORKERS = 2               # Number of concurrent workers (adjust based on API limits)
    
    # Initialize generator
    try:
        generator = ConcurrentMathTaskGenerator(project_id=PROJECT_ID, max_workers=MAX_WORKERS)
        generator.logger.info("Successfully initialized Concurrent Gemini API with Vertex AI")
    except Exception as e:
        # Create a temporary logger for initialization errors
        temp_logger = setup_logging()
        temp_logger.error(f"Error initializing Gemini API: {e}")
        temp_logger.error("Make sure you have run 'gcloud auth application-default login'")
        temp_logger.error("And ensure your project has Vertex AI API enabled")
        return
    
    # Process all skills concurrently (only remaining ones by default)
    await generator.process_all_skills(
        input_csv=INPUT_CSV,
        tasks_per_difficulty=TASKS_PER_DIFFICULTY,  # 100 per difficulty level
        test_mode=TEST_MODE,
        test_tasks=TEST_TASKS,
        only_remaining=True,  # Set to False to regenerate all skills
        batch_size=MAX_WORKERS  # Process this many skills concurrently
    )
    
    generator.logger.info("=== Concurrent Generation Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
