import csv
import os
import json
import random
import time
import asyncio
import re
import sys
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading

# Ensure proper error handling for imports
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("\n" + "="*80)
    print("❌ Error: Required Google AI packages not found.")
    print("\nPlease install the required packages using:")
    print("\n    pip install google-generativeai")
    print("\nIf you continue to see this error after installation, ensure you have:")
    print("1. Properly set up Google Cloud credentials")
    print("2. Run 'gcloud auth application-default login'")
    print("3. Enabled the Vertex AI API in your Google Cloud project")
    print("\nFor more information, visit: https://cloud.google.com/vertex-ai/docs/start/client-libraries")
    print("="*80 + "\n")
    sys.exit(1)

class ConcurrentMathTaskGenerator:
    def __init__(self, project_id: str = "studyhall-dev-383420", max_workers: int = 3):
        """Initialize the Gemini API client with Vertex AI and concurrency settings"""
        self.project_id = project_id
        self.location = "us-central1"
        self.model = "gemini-2.5-pro"
        self.max_workers = max_workers  # Default reduced to 3 to avoid rate limits
        
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
            print(f"  [Thread {thread_id % 1000}] Created new Gemini API client")
        
        # Increment counter
        self._local.request_counter += 1
        
        # Check if we should reset the client after a number of requests
        if self._local.request_counter >= self._max_requests_per_session:
            print(f"  [Thread {thread_id % 1000}] Resetting API client after {self._local.request_counter} requests")
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
    
    def prepare_examples_for_prompt(self, tasks: List[Dict[str, Any]], num_examples: int = 15) -> List[Dict[str, Any]]:
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
    
    def create_generation_prompt(self, examples: List[Dict[str, Any]], skill: str, 
                                existing_generated: List[Dict[str, Any]] = None, num_to_generate: int = 10) -> str:
        """Create a detailed prompt for generating math tasks"""
        
        # Analyze examples to understand patterns
        difficulties = set(task.get('difficulty', '').strip() for task in examples)
        test_name = examples[0].get('test', 'Math') if examples else 'Math'
        domain = examples[0].get('domain', 'Algebra') if examples else 'Algebra'
        
        prompt = f"""You are an expert math educator creating high-quality {skill} problems with visual elements. 

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
   - figure_description: DETAILED description of graphs, diagrams, tables, or visual elements
   - option_A_latex, option_B_latex, option_C_latex, option_D_latex: Multiple choice options (if applicable, include tables in MathJax syntax if needed)
   - correct_answer: The correct answer (A, B, C, D, or direct answer)
   - correct_answer_spr_latex: The correct answer in LaTeX format
   - step_1, step_2, step_3, step_4, step_5, step_6: Individual explanation steps (UP TO 6 STEPS MAXIMUM)

6. **Figure Descriptions - CRITICAL - EXTREMELY DETAILED**: 
   - Create ULTRA-DETAILED, COMPREHENSIVE figure descriptions that are SO SPECIFIC that an artist could draw the exact figure without ANY ambiguity
   - Include EVERY visual detail: coordinate systems, exact axis ranges, precise labels, gridline spacing, tick marks, scales, fonts, positioning
   - Describe graphs with COMPLETE precision: exact line types (solid, dashed, dotted), curve shapes, intersection coordinates, slope values, point markers, arrow directions
   - Describe tables with FULL structural details: exact dimensions, border styles, cell alignment, header formatting, data positioning, spacing
   - Describe geometric figures with ABSOLUTE precision: exact shapes, all measurements, angle values, vertex labels, orientation, positioning, line styles
   - Use MAXIMUM mathematical precision and drawing-level detail
   - MUCH MORE DETAILED than basic examples - include positioning, styling, exact coordinates, visual formatting
   - Examples of REQUIRED DETAIL LEVEL:
     * "Coordinate plane graph with origin O labeled at bottom-left intersection. X-axis extends from -6 to 8 with tick marks at every integer, labeled at intervals of 2 (-6, -4, -2, 0, 2, 4, 6, 8). Y-axis extends from -4 to 10 with tick marks at every integer, labeled at intervals of 2 (-4, -2, 0, 2, 4, 6, 8, 10). Light gray gridlines at every unit. Parabola drawn as solid black curve, opening upward, vertex marked with solid black dot at exact coordinates (2, -3), passes through points (0, 1), (1, -2), (3, -2), (4, 1) all marked with solid black dots. Axis labels 'x' and 'y' positioned at positive ends of axes."
     * "Table with black border, 5 rows including header, 4 columns. Header row with gray background, bold black text, centered alignment. Column headers: 'Time (h)', 'Position (m)', 'Velocity (m/s)', 'Acceleration (m/s²)'. Data rows with white background, regular black text, right-aligned numbers. Exact data: Row 1: 0, 0, 5, 2; Row 2: 1, 7, 9, 4; Row 3: 2, 20, 17, 8; Row 4: 3, 45, 33, 16. Cell padding 5px, border thickness 1px."
     * "Right triangle ABC oriented with right angle C at bottom-left. Vertex A at top (acute angle 37°), vertex B at bottom-right (acute angle 53°). Side AC (left edge) exactly vertical, 8 units long, labeled '8' in blue text positioned left of midpoint. Side BC (bottom edge) exactly horizontal, 6 units long, labeled '6' in blue text positioned below midpoint. Hypotenuse AB at angle, 10 units long, labeled '10' in blue text positioned above midpoint. Right angle symbol (small square) at vertex C. All vertices labeled with capital letters in black text positioned 3mm outside triangle."

7. **Variety**: Create diverse problems within the skill area
8. **Avoid Common Issues**: 
   - Don't use [Math: ...] notation
   - Use proper LaTeX delimiters ($...$)
   - Ensure fractions use \\frac{{numerator}}{{denominator}}
   - Use \\left( and \\right) for large parentheses when needed

9. **Visual Elements**: 
   - If problems involve tables or data, include them in answer options using MathJax syntax
   - Format tables using \\begin{{array}} and \\end{{array}} with proper alignment
   - Example: $\\begin{{array}}{{|c|c|}} \\hline x & y \\\\ \\hline 1 & 2 \\\\ 3 & 4 \\\\ \\hline \\end{{array}}$
   - Always describe the visual element in figure_description field

10. **Step Formatting**:
   - Each step should be a complete, standalone explanation
   - Use clear mathematical notation in each step
   - Each step should logically lead to the next
   - Fill only the necessary steps (can be fewer than 6)

11. **Problem Types**: Mix different types of problems within the skill:
   - Graph-based problems (coordinate planes, functions, systems)
   - Table-based problems (data analysis, linear relationships)
   - Geometric figure problems (triangles, circles, polygons)
   - Some with multiple choice options (A, B, C, D)
   - Some with direct numerical answers
   - Include word problems with visual components
   - Vary complexity within difficulty levels"""

        if existing_generated:
            prompt += f"""

AVOID REPETITION: You have already generated the following problems. DO NOT create similar problems:
{json.dumps(existing_generated, indent=2)} 
"""

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

CRITICAL JSON REQUIREMENTS: 
- **figure_description is MANDATORY and must be ULTRA-DETAILED**: Every problem must have an extremely detailed figure description that provides complete drawing specifications
- **DRAWING-LEVEL PRECISION**: Include enough detail that a technical illustrator could recreate the exact figure without any questions or ambiguity
- **SPECIFY EVERYTHING**: Exact coordinates, line styles, colors, fonts, positioning, spacing, borders, alignments, tick marks, labels, orientations, scales, and visual formatting
- Use separate step_1, step_2, etc. fields for each explanation step
- Each step should be complete and standalone
- Fill UP TO 6 STEPS MAXIMUM (can be fewer, leave unused steps empty or null)
- Include tables in answer options using MathJax array syntax when needed
- The JSON array must be syntactically perfect and parseable

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

FIGURE DESCRIPTION EXAMPLES - MINIMUM DETAIL LEVEL REQUIRED:
- For graphs: "Coordinate plane with origin O at intersection of axes. X-axis: horizontal line extending from -8 to 8, tick marks at every integer, numerical labels at even integers (-8, -6, -4, -2, 0, 2, 4, 6, 8) positioned 3mm below axis, axis label 'x' at right end. Y-axis: vertical line extending from -6 to 10, tick marks at every integer, numerical labels at even integers positioned 3mm left of axis, axis label 'y' at top end. Light gray gridlines at every unit forming 1×1 squares. Function graphed as solid blue curve representing f(x) = 2x + 1, passing through exactly these points with 2mm diameter filled blue circles: (-4, -7), (-2, -3), (0, 1), (2, 5), (4, 9). Line has constant positive slope, no curve, perfectly straight segments between points."

- For tables: "Rectangular table with solid black border (2px width). Dimensions: 6 columns × 5 rows including header. Header row: light gray background (#F0F0F0), bold black text (12pt Arial), centered alignment, row height 25px. Column headers with 2px spacing: 'Time (min)', 'Distance (km)', 'Speed (km/h)', 'Fuel (L)', 'Cost ($)', 'Total'. Data rows: white background, regular black text, right-aligned numbers except first column (left-aligned), row height 20px. Exact data values in grid format with consistent decimal places. Cell padding: 5px all sides. Alternating row highlighting: every odd data row has light blue background (#F8F8FF)."

- For geometric figures: "Equilateral triangle ABC positioned with base BC horizontal at bottom. Vertex A at apex (top), vertices B at bottom-left, C at bottom-right. Triangle orientation: A directly above midpoint of BC. All sides equal length 6 cm. Side AB: left edge from A to B, labeled '6 cm' in black text positioned 2mm outside triangle, parallel to side. Side BC: bottom edge from B to C, labeled '6 cm' in black text positioned 2mm below side, horizontal orientation. Side AC: right edge from A to C, labeled '6 cm' in black text positioned 2mm outside triangle, parallel to side. All three interior angles are 60°, marked with small arc symbols (3mm radius) and '60°' labels positioned inside triangle near each vertex. Vertices labeled with capital letters A, B, C in bold black text positioned 3mm outside triangle vertices."
"""
        
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
                    print(f"  [{skill}] Generating batch (attempt {retry_count + 1}/{max_total_retries})...")
                    
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
                            print(f"  API Stream Error: {error_name}: {str(e)}")
                            
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
                        print(f"  [{skill}] API request timed out after 240 seconds")
                        api_errors["TimeoutError"] += 1
                        should_reset_client = True
                        raise
                    except Exception as e:
                        print(f"  [{skill}] API execution error: {type(e).__name__}: {str(e)}")
                        raise
                    
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
                    
                    if not new_tasks:  # Empty list
                        raise ValueError("Response contains empty list of tasks")
                    
                    # Process and clean generated tasks
                    processed_tasks = self.process_generated_tasks(new_tasks)
                    
                    # Check if we got a reasonable number of tasks
                    if len(processed_tasks) < 1:
                        raise ValueError(f"Too few tasks generated: {len(processed_tasks)}")
                    
                    print(f"  [{skill}] Successfully generated {len(processed_tasks)} tasks")
                    
                    # Reset error counts on success
                    with self.lock:
                        self.error_counters["success"] = self.error_counters.get("success", 0) + 1
                        
                    return processed_tasks, input_tokens, output_tokens, total_tokens
                    
                except json.JSONDecodeError as e:
                    print(f"  [{skill}] JSON parsing error (attempt {retry_count + 1}): {e}")
                    print(f"  [{skill}] Raw response (first 500 chars): {response_text[:500]}...")
                    if len(response_text) > 500:
                        print(f"  [{skill}] Response length: {len(response_text)} characters")
                    api_errors["Other"] += 1
                
                except asyncio.TimeoutError:
                    print(f"  [{skill}] Request timed out (attempt {retry_count + 1}) - API took too long to respond")
                    api_errors["TimeoutError"] += 1
                    should_reset_client = True
                    
                except Exception as e:
                    error_name = type(e).__name__
                    error_msg = str(e)
                    print(f"  [{skill}] Generation error (attempt {retry_count + 1}): {error_name}: {error_msg}")
                    
                    # Track specific API errors
                    if "RetryError" in error_name:
                        api_errors["RetryError"] += 1
                        should_reset_client = True
                        print(f"  [{skill}] 🚨 RetryError detected - will reset client and retry with increased delay")
                        # Allow extra retries for RetryError
                        if retry_count >= max_retries and api_errors["RetryError"] <= 2:
                            max_total_retries += 1
                            
                    elif "ClientError" in error_name:
                        api_errors["ClientError"] += 1
                        should_reset_client = True
                        print(f"  [{skill}] 🚨 ClientError detected - will reset client and retry with increased delay")
                        
                    elif "ResourceExhausted" in error_msg:
                        api_errors["ResourceExhausted"] += 1
                        should_reset_client = True
                        print(f"  [{skill}] 🚨 ResourceExhausted error - API quota exceeded. Adding longer delay before retry.")
                        # Add extra delay for quota issues
                        await asyncio.sleep(15 + random.uniform(5, 15))
                    else:
                        api_errors["Other"] += 1
                
                retry_count += 1
                
                # Reset client if needed (but limit the number of resets)
                if should_reset_client and client_reset_count < max_client_resets:
                    client_reset_count += 1
                    print(f"  [{skill}] 🔄 Resetting API client (reset #{client_reset_count})...")
                    self.get_client(force_reset=True)
                
                if retry_count < max_total_retries:
                    # Calculate delay with more aggressive exponential backoff and jitter
                    delay = base_delay * (3 ** min(retry_count, 3)) + random.uniform(2, 10)
                    
                    # Add extra delay for specific error types
                    if api_errors["RetryError"] > 0 or api_errors["ClientError"] > 0:
                        delay += 10 + (api_errors["RetryError"] + api_errors["ClientError"]) * 5
                    
                    # Cap the max delay at 2 minutes
                    delay = min(delay, 120)
                    
                    print(f"  [{skill}] Waiting {delay:.1f} seconds before retry {retry_count + 1}/{max_total_retries}...")
                    
                    # Track errors in the global counter
                    with self.lock:
                        for err_type, count in api_errors.items():
                            if count > 0:
                                self.error_counters[err_type] = self.error_counters.get(err_type, 0) + 1
                    
                    await asyncio.sleep(delay)
            
            # Log error summary
            print(f"  [{skill}] ❌ All retries failed. Error counts: {api_errors}")
            return None, 0, 0, 0
    
    async def generate_tasks_for_skill(self, skill: str, examples: List[Dict[str, Any]], 
                                     total_tasks: int = 100, batch_size: int = 25) -> List[Dict[str, Any]]:
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
        print(f"[{skill}] Total token usage: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_tokens_used}")
        
        # Update progress
        with self.lock:
            self.completed_skills += 1
            print(f"\n📊 Progress: {self.completed_skills}/{self.total_skills} skills completed")
        
        return all_generated
    
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
                
                # Add retry for the entire batch with increasing delays
                max_batch_retries = 2
                for batch_retry in range(max_batch_retries + 1):
                    try:
                        if batch_retry > 0:
                            retry_delay = 30 * batch_retry
                            print(f"  [{skill}] {difficulty} - Batch retry {batch_retry}/{max_batch_retries} after {retry_delay}s delay")
                            await asyncio.sleep(retry_delay)
                            
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
                            break  # Success, exit retry loop
                        else:
                            print(f"  [{skill}] {difficulty} - Failed to generate batch {batch_num + 1} (attempt {batch_retry + 1})")
                            if batch_retry == max_batch_retries:
                                # All retries failed
                                print(f"  ❌ [{skill}] {difficulty} - All batch attempts failed, moving to next batch")
                                # Don't break the outer batch loop, try the next batch
                    except Exception as e:
                        print(f"  ❌ [{skill}] {difficulty} - Error in batch {batch_num + 1} (attempt {batch_retry + 1}): {e}")
                        if batch_retry == max_batch_retries:
                            print(f"  ❌ [{skill}] {difficulty} - All batch attempts failed with errors")
                            # Continue to next batch
                
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

    def get_already_generated_skills(self) -> set:
        """Get list of skills that have already been generated"""
        generated_skills = set()
        
        # Look for generated CSV files in current directory
        for filename in os.listdir('.'):
            if filename.startswith('generated_') and filename.endswith('.csv'):
                # Extract skill name from filename
                skill_name = filename[10:-4]  # Remove 'generated_' prefix and '.csv' suffix
                skill_name = skill_name.replace('_', ' ')  # Convert underscores back to spaces
                generated_skills.add(skill_name)
        
        return generated_skills
    
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
                    # Create filename
                    safe_skill_name = "".join(c for c in skill if c.isalnum() or c in (' ', '-', '_')).strip()
                    safe_skill_name = safe_skill_name.replace(' ', '_')
                    filename = f"generated_{safe_skill_name}.csv"
                    
                    # Save to CSV
                    self.save_tasks_to_csv(generated_tasks, filename)
                    return skill, len(generated_tasks)
                else:
                    print(f"⚠️ No tasks were generated for skill '{skill}'")
                    return skill, 0
            except asyncio.TimeoutError as e:
                print(f"❌ Timeout error processing skill '{skill}': {e}")
                return skill, f"TIMEOUT: {str(e)}"
            except Exception as e:
                error_type = type(e).__name__
                print(f"❌ Error processing skill '{skill}': {error_type}: {e}")
                
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
                print(f"❌ Unhandled exception for skill '{skill}': {error_type}: {error_msg}")
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
        print("=== Concurrent Math Task Generator (Difficulty-Based) ===")
        print(f"Input file: {input_csv}")
        print(f"Tasks per difficulty level: {tasks_per_difficulty}")
        print(f"Total tasks per skill: {tasks_per_difficulty * 3} (assuming Easy, Medium, Hard)")
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
            
            # If this isn't the first batch, add a cooling period between batches to avoid rate limits
            if i > 0:
                cooling_period = 30  # 30 seconds between batches to avoid hitting API limits
                print(f"⏱️ Adding cooling period of {cooling_period}s between batches to avoid API rate limits...")
                await asyncio.sleep(cooling_period)
            
            batch_start = time.time()
            results = await self.process_skill_batch(batch, tasks_per_difficulty, test_mode, test_tasks)
            batch_time = time.time() - batch_start
            
            # Count successful generations in this batch
            batch_generated = sum(count for skill, count in results if isinstance(count, int) and count > 0)
            total_generated += batch_generated
            
            # Track errors
            batch_errors = sum(1 for skill, count in results if not isinstance(count, int) or count == 0)
            
            print(f"✅ Batch {batch_num} completed in {batch_time:.1f}s, generated {batch_generated} tasks")
            if batch_errors > 0:
                print(f"⚠️ {batch_errors} skills had errors in this batch")
        
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
    
    def save_tasks_to_csv(self, tasks: List[Dict[str, Any]], filename: str):
        """Save tasks to CSV file with detailed error diagnostics"""
        if not tasks:
            print(f"No tasks to save for {filename}")
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

async def main():
    # Configuration
    INPUT_CSV = "SAT_questions_with_figures.csv"  # Your input CSV file
    TASKS_PER_DIFFICULTY = 100    # Number of tasks to generate per difficulty level (100 each for Easy, Medium, Hard = 300 total per skill)
    TEST_MODE = False             # Set to True to generate only 10 tasks for testing
    TEST_TASKS = 10              # Number of tasks in test mode
    PROJECT_ID = "studyhall-dev-383420"  # Your Google Cloud project ID
    MAX_WORKERS = 1              # Reduced concurrent workers to avoid API rate limits
    BATCH_SIZE = 5               # Number of skills to process concurrently (reduced to avoid rate limits)
    
    print("=== SAT Math Task Generator with Figures (Difficulty-Based) ===")
    print(f"Input file: {INPUT_CSV}")
    print(f"Tasks per difficulty: {TASKS_PER_DIFFICULTY} × 3 difficulties = {TASKS_PER_DIFFICULTY * 3} total per skill")
    print(f"Test mode: {'Enabled - ' + str(TEST_TASKS) + ' tasks per skill' if TEST_MODE else 'Disabled'}")
    print(f"Project ID: {PROJECT_ID}")
    print(f"Concurrent workers: {MAX_WORKERS}")
    print(f"Skills batch size: {BATCH_SIZE}")
    
    # Initialize generator with robust error handling
    try:
        generator = ConcurrentMathTaskGenerator(project_id=PROJECT_ID, max_workers=MAX_WORKERS)
        print("✅ Successfully initialized Concurrent Gemini API with Vertex AI")
    except ImportError as e:
        print("\n" + "="*80)
        print("❌ Error importing required modules: {e}")
        print("\nPlease install the required packages using:")
        print("\n    pip install google-generativeai")
        print("\nIf you continue to see this error after installation, ensure you have:")
        print("1. Properly set up Google Cloud credentials")
        print("2. Run 'gcloud auth application-default login'")
        print("3. Enabled the Vertex AI API in your Google Cloud project")
        print("\nFor more information, visit: https://cloud.google.com/vertex-ai/docs/start/client-libraries")
        print("="*80 + "\n")
        return
    except Exception as e:
        print(f"❌ Error initializing Gemini API: {e}")
        print("Make sure you have run 'gcloud auth application-default login'")
        print("And ensure your project has Vertex AI API enabled")
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
        print("\n✅ Generation completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user. Partial results may have been saved.")
    except Exception as e:
        print(f"\n❌ An error occurred during generation: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Print error statistics
        total_time = time.time() - start_time
        print("\n=== Gemini API Error Statistics ===")
        if hasattr(generator, 'error_counters') and generator.error_counters:
            total_errors = sum(count for err_type, count in generator.error_counters.items() if err_type != "success")
            total_success = generator.error_counters.get("success", 0)
            
            print(f"📊 Total API calls: {total_errors + total_success}")
            print(f"✅ Successful calls: {total_success}")
            print(f"❌ Failed calls: {total_errors}")
            
            if total_errors > 0:
                print("\nError breakdown:")
                for err_type, count in sorted(generator.error_counters.items(), key=lambda x: x[1], reverse=True):
                    if err_type != "success":
                        print(f"  • {err_type}: {count} occurrences")
                        
            print(f"\nRuntime: {total_time:.1f}s")
            
            if "RetryError" in generator.error_counters or "ClientError" in generator.error_counters:
                print("\n⚠️ Notable API issues detected ⚠️")
                print("Recommendations:")
                print("1. Reduce MAX_WORKERS and BATCH_SIZE even further")
                print("2. Increase request delays (self.request_delay in the generator) to 5-10 seconds")
                print("3. Add longer pauses between batches")
                print("4. Check your Vertex AI API quotas in the Google Cloud Console")
                print("5. Consider running at non-peak hours or spreading generation across multiple days")
        
        print("\n=== Concurrent Generation Process Finished ===")


if __name__ == "__main__":
    asyncio.run(main())