import csv
import os
import json
import random
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict
from google import genai
from google.genai import types

class MathTaskGenerator:
    def __init__(self, project_id: str = "studyhall-dev-383420"):
        """Initialize the Gemini API client with Vertex AI"""
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location="us-central1",
        )
        self.model = "gemini-2.5-pro"  # Using the latest model
        self.generated_tasks = []  # Keep track of generated tasks to avoid repetition
    
    def read_tasks_by_skill(self, csv_file: str) -> Dict[str, List[Dict[str, Any]]]:
        """Read tasks from CSV and group them by skill"""
        tasks_by_skill = defaultdict(list)
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    skill = row.get('skill', '').strip()
                    if skill:
                        tasks_by_skill[skill].append(dict(row))
            
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
    
    def fix_json_escaping(self, json_text: str) -> str:
        """Fix common JSON escaping issues with LaTeX content"""
        import re
        
        # Fix unescaped backslashes that should be escaped in JSON strings
        # This is a careful approach to avoid breaking valid escapes
        
        # First, protect already properly escaped sequences
        protected_sequences = {}
        protection_counter = 0
        
        # Protect sequences that are already properly escaped
        for seq in ['\\\\', '\\"', '\\/', '\\b', '\\f', '\\n', '\\r', '\\t']:
            while seq in json_text:
                placeholder = f"__PROTECTED_{protection_counter}__"
                json_text = json_text.replace(seq, placeholder, 1)
                protected_sequences[placeholder] = seq
                protection_counter += 1
        
        # Now fix common LaTeX escaping issues within JSON string values
        # Look for patterns like "content with \something" and fix to "content with \\something"
        
        # Pattern to match JSON string values (between quotes, but not breaking on escaped quotes)
        def fix_backslashes_in_match(match):
            content = match.group(1)
            
            # Fix single backslashes that should be doubled for JSON
            # But be careful not to break already-escaped sequences
            content = re.sub(r'(?<!\\)\\(?![\\"/bfnrt])', r'\\\\', content)
            
            return f'"{content}"'
        
        # Apply the fix to string values in JSON
        json_text = re.sub(r'"([^"]*(?:\\.[^"]*)*)"', fix_backslashes_in_match, json_text)
        
        # Restore protected sequences
        for placeholder, original in protected_sequences.items():
            json_text = json_text.replace(placeholder, original)
        
        return json_text
    
    def parse_json_with_fallback(self, json_text: str) -> List[Dict[str, Any]]:
        """Parse JSON with multiple fallback strategies"""
        
        # Try 1: Direct parsing
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"  Direct JSON parsing failed: {str(e)[:100]}")
        
        # Try 2: Additional escaping fixes
        try:
            # More aggressive backslash fixing
            fixed_text = json_text
            # Fix common LaTeX sequences that cause issues
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
                # Only replace if not already escaped
                fixed_text = fixed_text.replace(old, new)
            
            return json.loads(fixed_text)
        except json.JSONDecodeError as e:
            print(f"  Fallback JSON parsing failed: {str(e)[:100]}")
        
        # Try 3: Extract JSON array using regex
        try:
            import re
            # Look for array pattern in the text
            array_pattern = r'\[.*?\]'
            match = re.search(array_pattern, json_text, re.DOTALL)
            if match:
                array_text = match.group(0)
                return json.loads(array_text)
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"  Regex extraction failed: {str(e)[:100]}")
        
        # If all fails, raise the original error
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
   - Vary complexity within difficulty levels"""

        if existing_generated:
            prompt += f"""

AVOID REPETITION: You have already generated the following problems. DO NOT create similar problems:
{json.dumps(existing_generated, indent=2)} 
"""

        prompt += f"""

OUTPUT FORMAT: Return ONLY a valid JSON array of {num_to_generate} problems, no additional text:

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

CRITICAL: 
- Use separate step_1, step_2, etc. fields for each explanation step
- Each step should be complete and standalone
- Use UP TO 6 STEPS MAXIMUM (can be fewer, leave unused steps empty or null)
- Include tables in answer options using MathJax array syntax when needed
- IMPORTANT: In JSON, escape backslashes properly (use \\\\ for LaTeX \\, \\$ for \$, etc.)
- Ensure all LaTeX expressions are properly escaped for valid JSON format"""
        
        return prompt
    
    def generate_batch_with_backoff(self, prompt: str, max_retries: int = 3, base_delay: float = 1.0) -> Optional[List[Dict[str, Any]]]:
        """Generate a batch of tasks with exponential backoff"""
        for attempt in range(max_retries):
            try:
                print(f"  Generating batch (attempt {attempt + 1}/{max_retries})...")
                
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
                
                for chunk in self.client.models.generate_content_stream(
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
                
                response_text = response_text.strip()
                
                # Print token usage information
                if total_tokens > 0:
                    print(f"  Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
                else:
                    print(f"  Token usage information not available")
                
                # Clean up response
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                response_text = response_text.strip()
                
                # Try to fix common JSON escaping issues before parsing
                response_text = self.fix_json_escaping(response_text)
                
                # Parse JSON with fallback methods
                new_tasks = self.parse_json_with_fallback(response_text)
                
                # Validate structure
                if not isinstance(new_tasks, list):
                    raise ValueError("Response is not a list")
                
                # Clean math content in generated tasks
                for task in new_tasks:
                    for key, value in task.items():
                        if isinstance(value, str):
                            task[key] = self.clean_math_content(value)
                
                print(f"  Successfully generated {len(new_tasks)} tasks")
                return new_tasks, input_tokens, output_tokens, total_tokens
                
            except json.JSONDecodeError as e:
                print(f"  JSON parsing error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    print(f"  Raw response: {response_text[:500]}...")
            except Exception as e:
                print(f"  Generation error (attempt {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"  Waiting {delay:.1f} seconds before retry...")
                time.sleep(delay)
        
        return None, 0, 0, 0
    
    def generate_tasks_for_skill(self, skill: str, examples: List[Dict[str, Any]], 
                                total_tasks: int = 100, batch_size: int = 25) -> List[Dict[str, Any]]:
        """Generate tasks for a specific skill in batches"""
        all_generated = []
        examples_for_prompt = self.prepare_examples_for_prompt(examples, 15)
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens_used = 0
        
        print(f"\nGenerating {total_tasks} tasks for skill: {skill}")
        print(f"Available examples: {len(examples)}, Using: {len(examples_for_prompt)} examples, generating in batches of {batch_size}")
        
        num_batches = (total_tasks + batch_size - 1) // batch_size
        
        for batch_num in range(num_batches):
            remaining_tasks = total_tasks - len(all_generated)
            current_batch_size = min(batch_size, remaining_tasks)
            
            print(f"\nBatch {batch_num + 1}/{num_batches}: Generating {current_batch_size} tasks...")
            
            prompt = self.create_generation_prompt(
                examples_for_prompt, 
                skill, 
                all_generated,  # Pass existing generated tasks to avoid repetition
                current_batch_size
            )
            
            batch_result = self.generate_batch_with_backoff(prompt)
            
            if batch_result[0]:  # batch_tasks is not None
                batch_tasks, input_tokens, output_tokens, total_tokens = batch_result
                all_generated.extend(batch_tasks)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_tokens_used += total_tokens
                print(f"  Total generated so far: {len(all_generated)}/{total_tasks}")
            else:
                print(f"  Failed to generate batch {batch_num + 1}")
                break
            
            # Small delay between batches
            if batch_num < num_batches - 1:
                time.sleep(2)
        
        print(f"\nCompleted generation for {skill}: {len(all_generated)} tasks")
        print(f"Total token usage for {skill}: Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_tokens_used}")
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
            with open(filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(tasks)
            print(f"Saved {len(tasks)} tasks to {filename}")
        except Exception as e:
            print(f"Error saving to {filename}: {e}")
    
    def get_already_generated_skills(self) -> set:
        """Get list of skills that have already been generated"""
        import glob
        
        generated_skills = set()
        generated_files = glob.glob("generated_*.csv")
        
        for file in generated_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    first_row = next(reader, None)
                    if first_row and 'skill' in first_row:
                        actual_skill = first_row['skill'].strip()
                        generated_skills.add(actual_skill)
            except Exception as e:
                print(f"Warning: Could not read {file}: {e}")
        
        return generated_skills

    def process_all_skills(self, input_csv: str, tasks_per_skill: int = 100, 
                          test_mode: bool = False, test_tasks: int = 10, only_remaining: bool = True):
        """Process all skills from the input CSV, optionally skipping already generated ones"""
        print("=== Math Task Generator ===")
        print(f"Input file: {input_csv}")
        print(f"Tasks per skill: {tasks_per_skill if not test_mode else test_tasks}")
        print(f"Test mode: {test_mode}")
        print(f"Only remaining skills: {only_remaining}")
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
        skills_to_process = {}
        for skill, examples in tasks_by_skill.items():
            if only_remaining and skill in already_generated:
                print(f"⏭️  Skipping {skill}: already generated")
                continue
            
            if len(examples) < 5:  # Need minimum examples
                print(f"⚠️  Skipping {skill}: only {len(examples)} examples (need at least 5)")
                continue
            
            skills_to_process[skill] = examples
        
        if not skills_to_process:
            print("🎉 No remaining skills to generate! All skills are already done.")
            return
        
        print(f"\n⏳ Will generate tasks for {len(skills_to_process)} skill(s):")
        for skill in sorted(skills_to_process.keys()):
            print(f"  🔄 {skill}")
        print()
        
        # Process each remaining skill
        for skill, examples in skills_to_process.items():
            # Generate tasks
            target_tasks = test_tasks if test_mode else tasks_per_skill
            generated_tasks = self.generate_tasks_for_skill(skill, examples, target_tasks)
            
            if generated_tasks:
                # Create filename
                safe_skill_name = "".join(c for c in skill if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_skill_name = safe_skill_name.replace(' ', '_')
                filename = f"generated_{safe_skill_name}.csv"
                
                # Save to CSV
                self.save_tasks_to_csv(generated_tasks, filename)
            
            print("-" * 80)


def main():
    # Configuration
    INPUT_CSV = "SAT Questions - SAT_Math_No_Graph.csv"  # Your input CSV file
    TASKS_PER_SKILL = 100         # Number of tasks to generate per skill
    TEST_MODE = False              # Set to True to generate only 10 tasks for testing
    TEST_TASKS = 10               # Number of tasks in test mode
    PROJECT_ID = "studyhall-dev-383420"  # Your Google Cloud project ID
    
    # Initialize generator
    try:
        generator = MathTaskGenerator(project_id=PROJECT_ID)
        print("Successfully initialized Gemini API with Vertex AI")
    except Exception as e:
        print(f"Error initializing Gemini API: {e}")
        print("Make sure you have run 'gcloud auth application-default login'")
        print("And ensure your project has Vertex AI API enabled")
        return
    
    # Process all skills (only remaining ones by default)
    generator.process_all_skills(
        input_csv=INPUT_CSV,
        tasks_per_skill=TASKS_PER_SKILL,
        test_mode=TEST_MODE,
        test_tasks=TEST_TASKS,
        only_remaining=True  # Set to False to regenerate all skills
    )
    
    print("\n=== Generation Complete ===")


if __name__ == "__main__":
    main()
