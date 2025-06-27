import csv
import os
from collections import Counter
from typing import Dict, List
import glob

def analyze_csv_difficulty(csv_file: str) -> Dict[str, float]:
    """Analyze difficulty distribution in a single CSV file"""
    difficulties = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                difficulty = row.get('difficulty', '').strip()
                if difficulty:
                    difficulties.append(difficulty)
        
        if not difficulties:
            return {}
        
        # Count occurrences
        difficulty_counts = Counter(difficulties)
        total_tasks = len(difficulties)
        
        # Calculate percentages
        difficulty_percentages = {}
        for difficulty, count in difficulty_counts.items():
            percentage = (count / total_tasks) * 100
            difficulty_percentages[difficulty] = {
                'count': count,
                'percentage': percentage
            }
        
        return difficulty_percentages, total_tasks
        
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return {}, 0
    except Exception as e:
        print(f"Error reading '{csv_file}': {e}")
        return {}, 0

def print_difficulty_analysis(csv_file: str, analysis: Dict, total_tasks: int):
    """Print formatted difficulty analysis for a CSV file"""
    print(f"\n📊 {csv_file}")
    print("=" * (len(csv_file) + 4))
    print(f"Total tasks: {total_tasks}")
    print()
    
    if not analysis:
        print("❌ No difficulty data found")
        return
    
    # Sort by count (descending)
    sorted_difficulties = sorted(analysis.items(), key=lambda x: x[1]['count'], reverse=True)
    
    print("Difficulty Distribution:")
    print("-" * 40)
    
    for difficulty, data in sorted_difficulties:
        count = data['count']
        percentage = data['percentage']
        bar_length = int(percentage / 2)  # Scale for display
        bar = "█" * bar_length
        
        print(f"{difficulty:<15} {count:>4} ({percentage:>5.1f}%) {bar}")
    
    print("-" * 40)

def analyze_multiple_csvs(csv_patterns: List[str]):
    """Analyze multiple CSV files matching the given patterns"""
    print("🔍 CSV Difficulty Level Analysis")
    print("=" * 50)
    
    all_files = []
    for pattern in csv_patterns:
        matching_files = glob.glob(pattern)
        all_files.extend(matching_files)
    
    if not all_files:
        print("❌ No CSV files found matching the patterns:")
        for pattern in csv_patterns:
            print(f"   - {pattern}")
        return
    
    # Remove duplicates and sort
    all_files = sorted(list(set(all_files)))
    
    print(f"Found {len(all_files)} CSV file(s) to analyze:\n")
    
    summary_data = {}
    
    for csv_file in all_files:
        analysis, total_tasks = analyze_csv_difficulty(csv_file)
        print_difficulty_analysis(csv_file, analysis, total_tasks)
        
        # Store for summary
        summary_data[csv_file] = {
            'analysis': analysis,
            'total_tasks': total_tasks
        }
    
    # Print overall summary
    print_overall_summary(summary_data)

def print_overall_summary(summary_data: Dict):
    """Print an overall summary across all CSV files"""
    if len(summary_data) <= 1:
        return
    
    print("\n" + "=" * 60)
    print("📈 OVERALL SUMMARY ACROSS ALL FILES")
    print("=" * 60)
    
    # Aggregate all difficulties
    overall_difficulties = Counter()
    total_all_tasks = 0
    
    for file_data in summary_data.values():
        total_all_tasks += file_data['total_tasks']
        for difficulty, data in file_data['analysis'].items():
            overall_difficulties[difficulty] += data['count']
    
    if total_all_tasks == 0:
        print("❌ No tasks found across all files")
        return
    
    print(f"Total tasks across all files: {total_all_tasks}")
    print()
    
    # Sort by count (descending)
    sorted_overall = sorted(overall_difficulties.items(), key=lambda x: x[1], reverse=True)
    
    print("Overall Difficulty Distribution:")
    print("-" * 50)
    
    for difficulty, count in sorted_overall:
        percentage = (count / total_all_tasks) * 100
        bar_length = int(percentage / 2)  # Scale for display
        bar = "█" * bar_length
        
        print(f"{difficulty:<15} {count:>5} ({percentage:>5.1f}%) {bar}")
    
    print("-" * 50)
    
    # File comparison table
    print("\n📋 File Comparison Table:")
    print("-" * 80)
    
    # Get all unique difficulties
    all_difficulties = set()
    for file_data in summary_data.values():
        all_difficulties.update(file_data['analysis'].keys())
    all_difficulties = sorted(all_difficulties)
    
    # Header
    header = f"{'File':<30}"
    for diff in all_difficulties:
        header += f"{diff:<12}"
    header += f"{'Total':<8}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for filename, file_data in summary_data.items():
        # Shorten filename for display
        short_filename = os.path.basename(filename)
        if len(short_filename) > 28:
            short_filename = short_filename[:25] + "..."
        
        row = f"{short_filename:<30}"
        
        for diff in all_difficulties:
            if diff in file_data['analysis']:
                count = file_data['analysis'][diff]['count']
                percentage = file_data['analysis'][diff]['percentage']
                row += f"{count}({percentage:.0f}%)"[:11].ljust(12)
            else:
                row += f"{'0(0%)':<12}"
        
        row += f"{file_data['total_tasks']:<8}"
        print(row)

def main():
    """Main function to run the analysis"""
    
    # Default patterns to look for CSV files
    csv_patterns = [
        "*.csv",  # All CSV files in current directory
        "generated_*.csv",  # Generated files
        "SAT*.csv",  # SAT files
    ]
    
    # You can also specify specific files:
    specific_files = [
        "SAT Questions - SAT Math No Graph.csv",
        # Add more specific files here if needed
    ]
    
    print("Choose analysis mode:")
    print("1. Analyze all CSV files in current directory")
    print("2. Analyze specific files")
    print("3. Analyze generated_*.csv files only")
    
    try:
        choice = input("\nEnter choice (1-3) or just press Enter for option 1: ").strip()
        
        if choice == "2":
            patterns = specific_files
        elif choice == "3":
            patterns = ["generated_*.csv"]
        else:
            patterns = ["*.csv"]
        
        analyze_multiple_csvs(patterns)
        
    except KeyboardInterrupt:
        print("\n\n👋 Analysis cancelled by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")

if __name__ == "__main__":
    main() 