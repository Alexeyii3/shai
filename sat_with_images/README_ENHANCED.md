# Concurrent Math Task Generator with Figure Descriptions

Enhanced version of the math task generator that includes **detailed figure descriptions** for visual mathematical content.

## 🎯 **Key Features**

### **Enhanced Visual Content Generation**
- **Detailed Figure Descriptions**: Generates comprehensive descriptions of graphs, tables, diagrams, and visual elements
- **Mathematical Precision**: Includes coordinate systems, axis ranges, key points, and visual details
- **Educational Clarity**: Descriptions help understand mathematical content without seeing the actual figure

### **Figure Description Examples**
The generator creates detailed descriptions like:

```
"Graph of two intersecting lines in the xy-plane. Line 1 passes through points (-2, 4) and (3, -1) with negative slope. Line 2 passes through points (0, 2) and (4, 6) with positive slope. Intersection point at (1, 3). X-axis ranges from -5 to 5, Y-axis ranges from -3 to 7, with unit grid."
```

```
"Table with 5 rows and 3 columns. Headers: 'Time (hours)', 'Distance (miles)', 'Speed (mph)'. Data shows increasing time from 1 to 4 hours, corresponding distances from 45 to 180 miles, and constant speed of 45 mph."
```

## 📁 **Files Structure**

```
sat_with_images/
├── generate_concurrent.py     # Enhanced concurrent generator
├── test_generator.py         # Test script for verification
├── SAT_questions_with_figures.csv  # Input data with figure descriptions
└── generated_*.csv           # Output files with figure descriptions
```

## 🚀 **Usage**

### **Basic Generation**
```bash
cd sat_with_images
python generate_concurrent.py
```

### **Test the Generator**
```bash
python test_generator.py
```

## 📊 **Enhanced CSV Structure**

The generator now produces CSV files with complete structure matching the input:

| Field | Description |
|-------|-------------|
| `test` | Test name (e.g., "Math") |
| `domain` | Mathematical domain (e.g., "Algebra") |
| `skill` | Specific skill area |
| `difficulty` | Easy, Medium, or Hard |
| `question_text_plain` | Question without LaTeX |
| `question_text_latex` | Question with LaTeX formatting |
| **`figure_description`** | **DETAILED visual element description** |
| `option_A/B/C/D` | Multiple choice options (plain text) |
| `option_A/B/C/D_latex` | Multiple choice options (LaTeX) |
| `correct_answer` | Correct answer identifier |
| `correct_answer_spr_latex` | Answer in LaTeX format |
| `explanation_plain` | Combined explanation (plain text) |
| `explanation_latex` | Combined explanation (LaTeX) |
| `step_1` through `step_6` | Individual solution steps |

## 🎨 **Figure Description Types**

### **1. Graph-Based Problems**
```
"Graph of a quadratic function f(x) = x² - 4x + 3 in the xy-plane. Parabola opens upward with vertex at (2, -1). X-intercepts at (1, 0) and (3, 0). Y-intercept at (0, 3). X-axis ranges from -1 to 5, Y-axis ranges from -3 to 5, with gridlines at unit intervals."
```

### **2. Table-Based Problems**
```
"Table with 4 rows and 2 columns showing linear relationship. Column headers: 'x' and 'y'. Data points: (0, 5), (2, 11), (4, 17), (6, 23). Each y-value increases by 6 when x increases by 2, showing constant rate of change of 3."
```

### **3. Geometric Figures**
```
"Right triangle DEF with right angle at vertex E. Hypotenuse DF = 15 units, leg DE = 9 units, leg EF = 12 units. Angle D measures 37°, angle F measures 53°. Triangle positioned with DE horizontal and EF vertical."
```

### **4. System of Equations**
```
"Graph showing system of two linear equations in xy-plane. Line 1: y = 2x + 1 (slope = 2, y-intercept = 1). Line 2: y = -x + 4 (slope = -1, y-intercept = 4). Lines intersect at point (1, 3). Coordinate plane with x-axis from -2 to 6, y-axis from -1 to 7."
```

## ⚙️ **Configuration**

In `generate_concurrent.py`, adjust these settings:

```python
# API Configuration
PROJECT_ID = "studyhall-dev-383420"
MAX_WORKERS = 4               # Concurrent workers
REQUEST_DELAY = 1.0           # Delay between requests

# Generation Settings
TASKS_PER_SKILL = 100         # Tasks per skill
TEST_MODE = False             # Quick testing mode
TEST_TASKS = 10               # Tasks in test mode
```

## 🔧 **Advanced Features**

### **Intelligent Field Processing**
- **Automatic Plain Text Generation**: Converts LaTeX to plain text
- **Step Consolidation**: Combines individual steps into explanation fields
- **Field Validation**: Ensures all required CSV fields are present
- **Content Cleaning**: Standardizes mathematical notation

### **Enhanced Error Handling**
- **Per-skill Error Recovery**: Individual skill failures don't stop other workers
- **JSON Parsing Fallbacks**: Multiple strategies for parsing AI responses
- **Content Validation**: Verifies generated content structure

### **Quality Assurance**
- **Figure Description Validation**: Ensures every task has a figure description
- **Mathematical Accuracy**: Validates LaTeX formatting and mathematical content
- **Consistency Checking**: Maintains format consistency across generated tasks

## 📈 **Performance Metrics**

### **Generation Speed**
- **Concurrent Processing**: 3-4x faster than sequential generation
- **Batch Optimization**: Intelligent batching for API efficiency
- **Rate Limiting**: Respects API limits while maximizing throughput

### **Content Quality**
- **Detailed Descriptions**: Comprehensive figure descriptions for all visual elements
- **Mathematical Precision**: Accurate LaTeX formatting and mathematical notation
- **Educational Value**: Clear, instructive content suitable for learning

## 🧪 **Testing**

### **Quick Test**
```bash
python test_generator.py
```

### **Verification Steps**
1. **API Connection**: Verifies Gemini API connectivity
2. **Data Loading**: Confirms input CSV reading
3. **Generation**: Tests task generation with figure descriptions
4. **Output Validation**: Checks CSV structure and content quality
5. **Figure Description**: Validates presence and quality of visual descriptions

## 📋 **Example Output**

Generated tasks include comprehensive figure descriptions:

```json
{
  "test": "Math",
  "domain": "Algebra", 
  "skill": "Linear equations in two variables",
  "difficulty": "Medium",
  "question_text_latex": "What is the slope of the line shown in the graph?",
  "figure_description": "Graph of a linear function in the xy-plane passing through points (-2, 1) and (4, 7). Line has positive slope of 1. X-axis ranges from -5 to 6 with unit markings. Y-axis ranges from -2 to 8 with unit markings. Grid lines visible at each unit interval.",
  "correct_answer": "1",
  "step_1": "Identify two points on the line: (-2, 1) and (4, 7)",
  "step_2": "Apply slope formula: m = (y₂ - y₁)/(x₂ - x₁)",
  "step_3": "Calculate: m = (7 - 1)/(4 - (-2)) = 6/6 = 1"
}
```

## 🛠️ **Troubleshooting**

### **Common Issues**

1. **Missing Figure Descriptions**
   - Check prompt requirements
   - Verify AI model response parsing
   - Ensure field validation is working

2. **LaTeX Formatting Issues**
   - Review JSON escaping in responses
   - Check mathematical notation cleaning
   - Validate LaTeX syntax

3. **CSV Structure Problems**
   - Verify column order matches input
   - Check field processing function
   - Ensure all required fields are populated

### **Debug Mode**
Enable detailed logging by setting:
```python
TEST_MODE = True
TEST_TASKS = 5
```

This generates a small sample for verification before full production runs.

## 📞 **Support**

For issues with:
- **API Configuration**: Check Google Cloud setup and authentication
- **Generation Quality**: Review prompt engineering and examples
- **Performance**: Adjust worker count and rate limiting
- **Output Format**: Verify CSV structure and field processing

The enhanced generator creates comprehensive mathematical content with detailed visual descriptions, making it ideal for educational applications requiring rich visual context.
