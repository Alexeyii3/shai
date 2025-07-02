import csv
import os
import json
import random
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
from google import genai
from google.genai import types


class ConcurrentMathTaskGenerator:
    def __init__(self, project_id: str = "studyhall-dev-383420", max_workers: int = 1):
        """Initialize the Gemini API client with Vertex AI and concurrency settings"""
        self.project_id = project_id
        self.location = "global"
        self.model = "gemini-2.5-pro"
        self.max_workers = max_workers
        
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
                    print(f"⚠️ Warning: Missing expected columns: {sorted(missing_columns)}")
                
                if unused_columns:
                    print(f"📊 Filtering out unused columns: {sorted(unused_columns)}")
                
                for row in reader:
                    skill = row.get('skill', '').strip()
                    if skill:
                        # Create a filtered row with only the needed columns
                        filtered_row = {col: row.get(col, '') for col in needed_columns if col in available_columns}
                        tasks_by_skill[skill].append(filtered_row)
            
            print(f"Successfully read tasks from {csv_file}")
            for skill, tasks in tasks_by_skill.items():
                print(f"  - {skill}: {len(tasks)} tasks")
                
        except FileNotFoundError:
            print(f"CSV file {csv_file} not found.")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
        
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
            content = match.group(1)
            for pattern, replacement in critical_fixes:
                content = re.sub(pattern, replacement, content)
            return f'"{content}"'
        
        # Apply fixes to content within JSON strings
        json_text = re.sub(r'"([^"]*(?:\\.[^"]*)*)"', fix_in_string, json_text)
        
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
            (r'\\\\\\\\', r'\\\\\\\\\\\\\\\\'),  # Fix double backslashes in tables
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
        json_text = re.sub(r'"([^"]*(?:\\.[^"]*)*)"', fix_latex_in_string, json_text)
        
        return json_text
    
    def parse_json_with_fallback(self, json_text: str) -> List[Dict[str, Any]]:
        """Parse JSON with multiple fallback strategies, handling text before JSON"""
        
        # Try 1: Direct parsing
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"  Direct JSON parsing failed: {str(e)[:100]}")
            print(f"  JSON text: {json_text}")
        
        # Try 2: Extract JSON array from mixed content (improved regex)
        try:
            import re
            
            # Look for JSON array that starts with [ and ends with ]
            # This handles cases where there's explanatory text before/after JSON
            patterns = [
                # Pattern 1: Find complete JSON array (greedy match)
                r'\[\s*\{.*?\}\s*\]',
                # Pattern 2: Find JSON array with proper nesting
                r'\[(?:[^[\]{}]*\{[^{}]*\}[^[\]{}]*)*\]',
                # Pattern 3: Simple array pattern (fallback)
                r'\[.*?\]'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, json_text, re.DOTALL)
                for match in matches:
                    try:
                        # Clean up the match
                        cleaned_match = match.strip()
                        
                        # Try to parse this potential JSON
                        result = json.loads(cleaned_match)
                        if isinstance(result, list) and len(result) > 0:
                            print(f"  Successfully extracted JSON using pattern: {pattern}")
                            return result
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"  Regex extraction failed: {str(e)[:100]}")
        
        # Try 3: Additional escaping fixes on the full text
        try:
            fixed_text = json_text
            latex_fixes = {
                '\\$': '\\\\$',
                '\\%': '\\\\%',
                '\\&': '\\\\&',
                '\\#': '\\\\#',
                '\\^': '\\\\^',
                '\\_': '\\\\_',
                '\\{': '\\\\{',
                '\\}': '\\\\}',
            }
            
            for old, new in latex_fixes.items():
                fixed_text = fixed_text.replace(old, new)
            
            return json.loads(fixed_text)
        except json.JSONDecodeError as e:
            print(f"  Fallback JSON parsing failed: {str(e)[:100]}")
        
        # Try 4: Extract JSON array and apply fixes
        try:
            import re
            
            # More aggressive JSON extraction
            # Look for content between first [ and last ]
            start_idx = json_text.find('[')
            end_idx = json_text.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                potential_json = json_text[start_idx:end_idx + 1]
                
                # Apply LaTeX fixes
                for old, new in latex_fixes.items():
                    potential_json = potential_json.replace(old, new)
                
                result = json.loads(potential_json)
                if isinstance(result, list):
                    print(f"  Successfully extracted JSON using index-based extraction")
                    return result
                    
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Index-based extraction failed: {str(e)[:100]}")
        
        # Try 5: Line-by-line search for JSON
        try:
            lines = json_text.split('\n')
            json_lines = []
            in_json = False
            brace_count = 0
            
            for line in lines:
                stripped = line.strip()
                if not in_json and stripped.startswith('['):
                    in_json = True
                    json_lines = [line]
                    brace_count = stripped.count('[') - stripped.count(']')
                elif in_json:
                    json_lines.append(line)
                    brace_count += stripped.count('[') - stripped.count(']')
                    if brace_count <= 0 and stripped.endswith(']'):
                        break
            
            if json_lines:
                potential_json = '\n'.join(json_lines)
                result = json.loads(potential_json)
                if isinstance(result, list):
                    print(f"  Successfully extracted JSON using line-by-line parsing")
                    return result
                    
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Line-by-line extraction failed: {str(e)[:100]}")
        
        raise json.JSONDecodeError("All JSON parsing methods failed", json_text, 0)
    
    def prepare_examples_for_prompt(self, tasks: List[Dict[str, Any]], num_examples: int = 25) -> List[Dict[str, Any]]:
        """Prepare example tasks for the prompt, cleaning math content"""
        examples = random.sample(tasks, min(num_examples, len(tasks)))
        
        cleaned_examples = []
        for task in examples:
            cleaned_task = {}
            for key, value in task.items():
                if isinstance(value, str):
                    cleaned_task[key] = self.clean_math_content(value)
                else:
                    cleaned_task[key] = value
            cleaned_examples.append(cleaned_task)
        
        return cleaned_examples
    
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
- The JSON must contain exactly {num_to_generate} complete problem objects"""
        
        return prompt
    
    async def generate_batch_with_backoff(self, prompt: str, skill: str, max_retries: int = 5, 
                                        base_delay: float = 1.0) -> Tuple[Optional[List[Dict[str, Any]]], int, int, int]:
        """Generate a batch of tasks with exponential backoff and rate limiting"""
        
        async with self.rate_limiter:
            # Add delay between requests
            await asyncio.sleep(self.request_delay)
            
            # Track errors for summary
            error_counts = {}
            
            for attempt in range(max_retries):
                try:
                    print(f"  [{skill}] Generating batch (attempt {attempt + 1}/{max_retries})...")
                    
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
                            print(f"  API Stream Error: {type(stream_error).__name__}: {stream_error}")
                            
                            # Detailed analysis of streaming errors
                            if "RetryError" in str(type(stream_error).__name__):
                                print(f"  🔍 RetryError Details:")
                                if hasattr(stream_error, 'last_attempt') and stream_error.last_attempt:
                                    if hasattr(stream_error.last_attempt, 'exception'):
                                        underlying = stream_error.last_attempt.exception
                                        print(f"    - Stream underlying error: {type(underlying).__name__}: {underlying}")
                                        if hasattr(underlying, 'code'):
                                            print(f"    - Stream error code: {underlying.code}")
                                        if hasattr(underlying, 'message'):
                                            print(f"    - Stream error message: {underlying.message}")
                            
                            # Re-raise to be caught by the outer exception handler
                            raise
                    
                    # Execute the API call asynchronously
                    await loop.run_in_executor(None, generate_content)
                    
                    response_text = response_text.strip()
                    
                    # Print token usage information
                    if total_tokens > 0:
                        print(f"  [{skill}] Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
                    else:
                        print(f"  [{skill}] Token usage information not available")
                    
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
                    
                    print(f"  [{skill}] Successfully generated {len(processed_tasks)} tasks")
                    return processed_tasks, input_tokens, output_tokens, total_tokens
                    
                except json.JSONDecodeError as e:
                    print(f"  [{skill}] JSON parsing error (attempt {attempt + 1}): {e}")
                    print(f"  [{skill}] Raw response (first 500 chars): {response_text[:500]}...")
                    if len(response_text) > 500:
                        print(f"  [{skill}] Response length: {len(response_text)} characters")
                except Exception as e:
                    error_name = type(e).__name__
                    error_message = str(e)
                    
                    print(f"  [{skill}] API execution error: {error_name}: {error_message}")
                    
                    # Detailed error analysis for RetryError and other common API errors
                    if "RetryError" in error_name:
                        print(f"  [{skill}] 🔍 RetryError Details:")
                        
                        # Try to extract the underlying error from RetryError
                        try:
                            print(f"    - RetryError type: {type(e)}")
                            print(f"    - RetryError attributes: {[attr for attr in dir(e) if not attr.startswith('_')]}")
                            
                            if hasattr(e, 'last_attempt') and e.last_attempt:
                                print(f"    - Last attempt type: {type(e.last_attempt)}")
                                print(f"    - Last attempt attributes: {[attr for attr in dir(e.last_attempt) if not attr.startswith('_')]}")
                                
                                if hasattr(e.last_attempt, 'exception'):
                                    underlying_error = e.last_attempt.exception
                                    print(f"    - Underlying error type: {type(underlying_error).__name__}")
                                    print(f"    - Underlying error: {underlying_error}")
                                    
                                    # If it's a callable (like Future.exception), try to call it
                                    if callable(underlying_error):
                                        try:
                                            actual_exception = underlying_error()
                                            print(f"    - Actual exception: {type(actual_exception).__name__}: {actual_exception}")
                                            
                                            # Now check for specific error attributes
                                            if hasattr(actual_exception, 'code'):
                                                print(f"    - Error code: {actual_exception.code}")
                                            if hasattr(actual_exception, 'message'):
                                                print(f"    - Error message: {actual_exception.message}")
                                            if hasattr(actual_exception, 'details'):
                                                print(f"    - Error details: {actual_exception.details}")
                                            if hasattr(actual_exception, 'status_code'):
                                                print(f"    - Status code: {actual_exception.status_code}")
                                            if hasattr(actual_exception, 'response'):
                                                print(f"    - Response: {actual_exception.response}")
                                                
                                            # Show all attributes of the actual exception
                                            exception_attrs = [attr for attr in dir(actual_exception) if not attr.startswith('_') and not callable(getattr(actual_exception, attr))]
                                            print(f"    - Exception attributes: {exception_attrs}")
                                            
                                        except Exception as call_e:
                                            print(f"    - Could not call exception method: {call_e}")
                                    else:
                                        # Check for specific error types on the underlying error
                                        if hasattr(underlying_error, 'code'):
                                            print(f"    - Error code: {underlying_error.code}")
                                        if hasattr(underlying_error, 'message'):
                                            print(f"    - Error message: {underlying_error.message}")
                                        if hasattr(underlying_error, 'details'):
                                            print(f"    - Error details: {underlying_error.details}")
                                        
                                        # Show all attributes
                                        underlying_attrs = [attr for attr in dir(underlying_error) if not attr.startswith('_') and not callable(getattr(underlying_error, attr))]
                                        print(f"    - Underlying error attributes: {underlying_attrs}")
                                        
                                if hasattr(e.last_attempt, 'outcome'):
                                    print(f"    - Outcome: {e.last_attempt.outcome}")
                                    
                            # Try to extract request details
                            if hasattr(e, 'request') and e.request:
                                print(f"    - Request method: {getattr(e.request, 'method', 'N/A')}")
                                print(f"    - Request URL: {getattr(e.request, 'url', 'N/A')}")
                            
                            # Try to access the Future directly if it's in the error message
                            error_str = str(e)
                            if 'Future at' in error_str:
                                print(f"    - Future detected in error string: {error_str}")
                                # Try to extract more info about what went wrong
                                import re
                                future_match = re.search(r'Future at (0x[0-9a-f]+)', error_str)
                                if future_match:
                                    print(f"    - Future address: {future_match.group(1)}")
                                
                        except Exception as inner_e:
                            print(f"    - Could not extract detailed error info: {inner_e}")
                            import traceback
                            print(f"    - Traceback: {traceback.format_exc()}")
                    
                    elif "ClientError" in error_name:
                        print(f"  [{skill}] 🔍 ClientError Details:")
                        if hasattr(e, 'response') and e.response:
                            print(f"    - Status code: {getattr(e.response, 'status_code', 'N/A')}")
                            print(f"    - Response headers: {getattr(e.response, 'headers', 'N/A')}")
                            try:
                                response_text = e.response.text if hasattr(e.response, 'text') else 'N/A'
                                print(f"    - Response text: {response_text[:200]}...")
                            except:
                                print(f"    - Response text: Could not extract")
                        
                        if hasattr(e, 'request') and e.request:
                            print(f"    - Request URL: {getattr(e.request, 'url', 'N/A')}")
                            print(f"    - Request method: {getattr(e.request, 'method', 'N/A')}")
                    
                    elif "TimeoutError" in error_name or "Timeout" in error_name:
                        print(f"  [{skill}] 🔍 Timeout Details:")
                        print(f"    - This might be due to network latency or server overload")
                        print(f"    - Consider increasing request timeout or reducing batch size")
                        
                    elif "PermissionDenied" in error_name or "Forbidden" in error_name:
                        print(f"  [{skill}] 🔍 Permission Details:")
                        print(f"    - Check if your API credentials are valid")
                        print(f"    - Verify project permissions for Vertex AI")
                        print(f"    - Ensure the service account has proper roles")
                        
                    elif "ResourceExhausted" in error_name or "QuotaExceeded" in error_name:
                        print(f"  [{skill}] 🔍 Quota Details:")
                        print(f"    - API quota may be exceeded")
                        print(f"    - Consider reducing concurrent requests or batch size")
                        print(f"    - Check quota limits in Google Cloud Console")
                    
                    # Try to extract additional error attributes
                    error_attrs = [attr for attr in dir(e) if not attr.startswith('_') and attr not in ['args', 'with_traceback']]
                    if error_attrs:
                        print(f"  [{skill}] 🔍 Additional error attributes: {error_attrs}")
                        for attr in error_attrs[:5]:  # Limit to first 5 attributes
                            try:
                                value = getattr(e, attr)
                                if not callable(value):
                                    print(f"    - {attr}: {value}")
                            except:
                                pass
                    
                    print(f"  [{skill}] Generation error (attempt {attempt + 1}): {error_name}: {error_message}")
                    
                    # Track error for summary
                    if error_name not in error_counts:
                        error_counts[error_name] = 0
                    error_counts[error_name] += 1
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"  [{skill}] Waiting {delay:.1f} seconds before retry...")
                    await asyncio.sleep(delay)
            
            print(f"  [{skill}] Error summary after {max_retries} attempts:")
            for error_type, count in error_counts.items():
                print(f"    - {error_type}: {count} occurrences")
            
            return None, 0, 0, 0
    
    async def generate_tasks_for_skill(self, skill: str, examples: List[Dict[str, Any]], 
                                     total_tasks: int = 100, batch_size: int = 10) -> List[Dict[str, Any]]:
        """Generate tasks for a specific skill in batches"""
        all_generated = []
        examples_for_prompt = self.prepare_examples_for_prompt(examples, 15)
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens_used = 0
        
        print(f"\n[{skill}] Starting generation of {total_tasks} tasks")
        print(f"[{skill}] Available examples: {len(examples)}, Using: {len(examples_for_prompt)} examples")
        
        num_batches = (total_tasks + batch_size - 1) // batch_size
        
        for batch_num in range(num_batches):
            remaining_tasks = total_tasks - len(all_generated)
            current_batch_size = min(batch_size, remaining_tasks)
            
            print(f"[{skill}] Batch {batch_num + 1}/{num_batches}: Generating {current_batch_size} tasks...")
            
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
                print(f"  [{skill}] Total generated so far: {len(all_generated)}/{total_tasks}")
            else:
                print(f"  [{skill}] Failed to generate batch {batch_num + 1}")
                break
            
            # Small delay between batches
            if batch_num < num_batches - 1:
                await asyncio.sleep(2)
        
        print(f"\n[{skill}] Completed generation: {len(all_generated)} tasks")
        print(f"[{skill}] Total token usage: Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_tokens_used}")
        
        # Update progress
        with self.lock:
            self.completed_skills += 1
            print(f"\n📊 Progress: {self.completed_skills}/{self.total_skills} skills completed")
        
        return all_generated
    
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
        
        print(f"\n[{skill}] Starting generation of {tasks_per_difficulty} tasks per difficulty level")
        print(f"[{skill}] Available difficulties: {list(examples_by_difficulty.keys())}")
        for difficulty, examples_list in examples_by_difficulty.items():
            print(f"[{skill}] {difficulty}: {len(examples_list)} examples")
        
        all_generated = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens_used = 0
        
        # Generate tasks for each difficulty level
        for difficulty in sorted(examples_by_difficulty.keys()):
            difficulty_examples = examples_by_difficulty[difficulty]
            
            if len(difficulty_examples) < 3:  # Need minimum examples for this difficulty
                print(f"⚠️  [{skill}] Skipping {difficulty}: only {len(difficulty_examples)} examples (need at least 3)")
                continue
            
            print(f"\n[{skill}] Generating {tasks_per_difficulty} tasks for difficulty: {difficulty}")
            
            # Prepare examples for this difficulty
            examples_for_prompt = self.prepare_examples_for_prompt(difficulty_examples, min(15, len(difficulty_examples)))
            
            # Generate tasks in batches for this difficulty
            difficulty_generated = []
            num_batches = (tasks_per_difficulty + batch_size - 1) // batch_size
            
            for batch_num in range(num_batches):
                remaining_tasks = tasks_per_difficulty - len(difficulty_generated)
                current_batch_size = min(batch_size, remaining_tasks)
                
                print(f"[{skill}] {difficulty} - Batch {batch_num + 1}/{num_batches}: Generating {current_batch_size} tasks...")
                
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
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    total_tokens_used += total_tokens
                    print(f"  [{skill}] {difficulty} - Generated so far: {len(difficulty_generated)}/{tasks_per_difficulty}")
                else:
                    print(f"  [{skill}] {difficulty} - Failed to generate batch {batch_num + 1}")
                    break
                
                # Small delay between batches
                if batch_num < num_batches - 1:
                    await asyncio.sleep(2)
            
            all_generated.extend(difficulty_generated)
            print(f"[{skill}] {difficulty} - Completed: {len(difficulty_generated)} tasks")
        
        print(f"\n[{skill}] TOTAL COMPLETED: {len(all_generated)} tasks across all difficulties")
        print(f"[{skill}] Total token usage: Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_tokens_used}")
        
        # Update progress
        with self.lock:
            self.completed_skills += 1
            print(f"\n📊 Progress: {self.completed_skills}/{self.total_skills} skills completed")
        
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
        
        # Generate tasks only for difficulties that need more tasks
        for difficulty, target_count in tasks_to_generate.items():
            if target_count <= 0:  # Skip if no additional tasks needed
                print(f"[{skill}] No additional {difficulty} tasks needed")
                continue
                
            if difficulty not in examples_by_difficulty:
                print(f"⚠️ [{skill}] No examples for {difficulty} difficulty, skipping")
                continue
                
            if len(examples_by_difficulty[difficulty]) < 3:  # Need minimum examples
                print(f"⚠️ [{skill}] Not enough examples for {difficulty} difficulty, skipping")
                continue
            
            print(f"\n[{skill}] Generating {target_count} additional tasks for difficulty: {difficulty}")
            
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
                
                print(f"[{skill}] {difficulty} - Batch {batch_num + 1}/{num_batches}: Generating {current_batch_size} tasks...")
                
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
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    total_tokens_used += total_tokens
                    print(f"  [{skill}] {difficulty} - Generated so far: {len(difficulty_generated)}/{target_count}")
                else:
                    print(f"  [{skill}] {difficulty} - Failed to generate batch {batch_num + 1}")
                    break
                
                # Small delay between batches
                if batch_num < num_batches - 1:
                    await asyncio.sleep(2)
            
            all_generated.extend(difficulty_generated)
            print(f"[{skill}] {difficulty} - Completed: {len(difficulty_generated)} additional tasks")
        
        if all_generated:
            print(f"\n[{skill}] TOTAL ADDITIONAL TASKS: {len(all_generated)}")
            print(f"[{skill}] Total token usage: Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_tokens_used}")
        else:
            print(f"\n[{skill}] No additional tasks were needed")
        
        return all_generated

    def save_tasks_to_csv(self, tasks: List[Dict[str, Any]], filename: str):
        """Save tasks to CSV file"""
        if not tasks:
            print(f"No tasks to save for {filename}")
            return
        
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
            print(f"✅ Saved {len(tasks)} tasks to {filename}")
        except ValueError as e:
            if "fields not in fieldnames" in str(e):
                print(f"❌ CSV Writing Error: Extra fields detected in tasks for {filename}")
                print(f"Error message: {e}")
                print("\n🔍 DEBUGGING: Full content of problematic tasks:")
                print("=" * 80)
                
                # Find and print the complete content of tasks with extra fields
                problematic_count = 0
                for i, task in enumerate(tasks):
                    extra_fields = set(task.keys()) - set(columns)
                    if extra_fields:
                        problematic_count += 1
                        print(f"\n📋 PROBLEMATIC TASK #{i + 1}:")
                        print(f"   Extra fields detected: {sorted(extra_fields)}")
                        print(f"   FULL TASK CONTENT:")
                        print("-" * 60)
                        
                        # Print all fields in the task for complete debugging info
                        for key, value in task.items():
                            # Truncate very long values for readability but keep essential info
                            if isinstance(value, str) and len(value) > 200:
                                truncated_value = value[:200] + "... [TRUNCATED]"
                            else:
                                truncated_value = value
                            
                            # Highlight extra fields
                            marker = "🚨 EXTRA → " if key in extra_fields else "         "
                            print(f"   {marker}{key}: {repr(truncated_value)}")
                        
                        print("-" * 60)
                
                print(f"\n📊 Summary: Found {problematic_count} tasks with extra fields out of {len(tasks)} total tasks")
                print("❌ This error should not occur after validation - please check the validation logic")
            else:
                print(f"❌ ValueError saving to {filename}: {e}")
        except Exception as e:
            print(f"❌ General error saving to {filename}: {e}")
    
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
                        print(f"  ✅ Complete: {skill_name} ({dict(tasks_by_difficulty)})")
                    else:
                        incomplete_skills.add(skill_name)
                        print(f"  ⚠️ Incomplete: {skill_name} ({dict(tasks_by_difficulty)})")
            except Exception as e:
                print(f"Warning: Could not read {file}: {e}")
        
        # For backwards compatibility, return just the complete skills
        # The incomplete skills will be handled by check_existing_tasks_by_difficulty
        return generated_skills
        
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
                        print(f"[{skill}] All difficulties have sufficient tasks (≥90), skipping generation")
                        return skill, 0
                    
                    # Generate new tasks only for difficulties that need more
                    generated_tasks = await self.generate_additional_tasks_by_difficulty(
                        skill, examples, tasks_to_generate
                    )
                
                if generated_tasks:
                    # Create filename
                    safe_skill_name = "".join(c for c in skill if c.isalnum() or c in (' ', '-', '_')).strip()
                    safe_skill_name = safe_skill_name.replace(' ', '_')
                    filename = f"generated_{safe_skill_name}.csv"
                    
                    # Check if the file already exists
                    existing_tasks = []
                    if os.path.exists(filename) and not test_mode:
                        try:
                            with open(filename, 'r', encoding='utf-8') as f:
                                reader = csv.DictReader(f)
                                existing_tasks = list(reader)
                            print(f"[{skill}] Read {len(existing_tasks)} existing tasks from {filename}")
                        except Exception as e:
                            print(f"⚠️ Error reading existing file {filename}: {e}")
                    
                    # Combine existing and new tasks
                    combined_tasks = existing_tasks + generated_tasks
                    
                    # Save combined tasks to CSV
                    self.save_tasks_to_csv(combined_tasks, filename)
                    return skill, len(generated_tasks)
                else:
                    return skill, 0
            except Exception as e:
                print(f"❌ Error processing skill '{skill}': {e}")
                return skill, 0
        
        # Process all skills in the batch concurrently
        tasks = [process_single_skill(skill, examples) for skill, examples in skill_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results

    async def process_all_skills(self, input_csv: str, tasks_per_difficulty: int = 100, 
                               test_mode: bool = False, test_tasks: int = 10, 
                               only_remaining: bool = True, batch_size: int = None):
        """Process all skills from the input CSV concurrently using difficulty-based generation"""
        print("=== Concurrent Math Task Generator ===")
        print(f"Input file: {input_csv}")
        print(f"Tasks per difficulty level: {tasks_per_difficulty}")
        print(f"Total tasks per skill: {tasks_per_difficulty * 3} (assuming Easy, Medium, Hard)")
        print(f"Test mode: {test_mode} ({test_tasks} total tasks per skill in test mode)" if test_mode else "Test mode: {test_mode}")
        print(f"Test mode: {test_mode}")
        print(f"Only remaining skills: {only_remaining}")
        print(f"Max concurrent workers: {self.max_workers}")
        print()
        
        # Read tasks by skill
        tasks_by_skill = self.read_tasks_by_skill(input_csv)
        
        if not tasks_by_skill:
            print("No tasks found in input file")
            return
        
        # Get already generated skills if requested
        already_generated = set()
        if only_remaining:
            already_generated = self.get_already_generated_skills()
            print(f"📊 Found {len(already_generated)} already generated skills")
            if already_generated:
                print("✅ Already generated:")
                for skill in sorted(already_generated):
                    print(f"  ✓ {skill}")
                print()
        
        # Filter skills to process
        skills_to_process = []
        for skill, examples in tasks_by_skill.items():
            if only_remaining and skill in already_generated:
                print(f"⏭️  Skipping {skill}: already generated")
                continue
            
            if len(examples) < 5:  # Need minimum examples
                print(f"⚠️  Skipping {skill}: only {len(examples)} examples (need at least 5)")
                continue
            
            skills_to_process.append((skill, examples))
        
        if not skills_to_process:
            print("🎉 No remaining skills to generate! All skills are already done.")
            return
        
        self.total_skills = len(skills_to_process)
        self.completed_skills = 0
        
        print(f"\n⏳ Will generate tasks for {len(skills_to_process)} skill(s) using {self.max_workers} concurrent workers:")
        for skill, _ in skills_to_process:
            print(f"  🔄 {skill}")
        print()
        
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
            
            print(f"\n🚀 Processing batch {batch_num}/{total_batches} ({len(batch)} skills)")
            print(f"   Skills: {', '.join([skill for skill, _ in batch])}")
            
            batch_start = time.time()
            results = await self.process_skill_batch(batch, tasks_per_difficulty, test_mode, test_tasks)
            batch_time = time.time() - batch_start
            
            # Count successful generations in this batch
            batch_generated = sum(count for skill, count in results if isinstance(count, int))
            total_generated += batch_generated
            
            print(f"✅ Batch {batch_num} completed in {batch_time:.1f}s, generated {batch_generated} tasks")
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 All skills processed!")
        print(f"📊 Total time: {total_time:.1f}s")
        print(f"📊 Total tasks generated: {total_generated}")
        print(f"📊 Average time per skill: {total_time / len(skills_to_process):.1f}s")
        print(f"📊 Skills processed: {self.completed_skills}/{self.total_skills}")
    
    def process_generated_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean generated tasks before saving"""
        processed_tasks = []
        
        for task in tasks:
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
                print(f"  Warning: Task missing required fields: {', '.join(missing_fields)}. Adding defaults.")
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
            print(f"Warning: Error reading {matching_files[0]}: {e}")
            return {"Easy": 100, "Medium": 100, "Hard": 100}
        
        # Calculate how many more tasks needed per difficulty
        tasks_to_generate = {}
        for difficulty in ["Easy", "Medium", "Hard"]:
            existing_count = tasks_by_difficulty.get(difficulty, 0)
            if existing_count < 90:  # If less than 90, generate more to reach 100
                tasks_to_generate[difficulty] = 100 - existing_count
            else:
                tasks_to_generate[difficulty] = 0
                
        print(f"[{skill}] Existing tasks by difficulty: {dict(tasks_by_difficulty)}")
        print(f"[{skill}] Additional tasks to generate: {tasks_to_generate}")
        
        return tasks_to_generate
    
async def main():
    # Configuration
    INPUT_CSV = "SAT Questions - SAT_Math_No_Graph.csv"  # Your input CSV file
    TASKS_PER_DIFFICULTY = 100    # Number of tasks to generate per difficulty level (100 each for Easy, Medium, Hard = 300 total per skill)
    TEST_MODE = False              # Set to True to generate only 10 tasks for testing
    TEST_TASKS = 10               # Number of tasks in test mode
    PROJECT_ID = "studyhall-dev-383420"  # Your Google Cloud project ID
    MAX_WORKERS = 1               # Number of concurrent workers (adjust based on API limits)
    
    # Initialize generator
    try:
        generator = ConcurrentMathTaskGenerator(project_id=PROJECT_ID, max_workers=MAX_WORKERS)
        print("Successfully initialized Concurrent Gemini API with Vertex AI")
    except Exception as e:
        print(f"Error initializing Gemini API: {e}")
        print("Make sure you have run 'gcloud auth application-default login'")
        print("And ensure your project has Vertex AI API enabled")
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
    
    print("\n=== Concurrent Generation Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
