import csv
import glob
import os
from collections import Counter
from typing import Dict, List, Tuple

def analyze_answer_distribution(csv_file: str) -> Tuple[Dict[str, int], int]:
    """Analyze answer option distribution in a single CSV file"""
    answers = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                correct_answer = row.get('correct_answer', '').strip().upper()
                if correct_answer:
                    answers.append(correct_answer)
        
        answer_counts = Counter(answers)
        total_tasks = len(answers)
        
        return answer_counts, total_tasks
        
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return Counter(), 0
    except Exception as e:
        print(f"Error reading '{csv_file}': {e}")
        return Counter(), 0

def print_answer_analysis(csv_file: str, answer_counts: Counter, total_tasks: int):
    """Print formatted answer distribution analysis for a CSV file"""
    print(f"\n📊 {os.path.basename(csv_file)}")
    print("=" * (len(os.path.basename(csv_file)) + 4))
    print(f"Total tasks: {total_tasks}")
    
    if total_tasks == 0:
        print("❌ No answer data found")
        return
    
    print("\nAnswer Distribution:")
    print("-" * 50)
    
    # Standard options
    standard_options = ['A', 'B', 'C', 'D']
    
    # Print standard options first
    for option in standard_options:
        count = answer_counts.get(option, 0)
        percentage = (count / total_tasks) * 100 if total_tasks > 0 else 0
        bar_length = int(percentage / 2)  # Scale for display
        bar = "█" * bar_length
        
        print(f"Option {option:<3} {count:>4} ({percentage:>5.1f}%) {bar}")
    
    # Print any non-standard answers
    non_standard = {k: v for k, v in answer_counts.items() if k not in standard_options}
    if non_standard:
        print("\nNon-standard answers:")
        for answer, count in sorted(non_standard.items()):
            percentage = (count / total_tasks) * 100
            print(f"'{answer}':{' ' * (6-len(answer))}{count:>4} ({percentage:>5.1f}%)")
    
    print("-" * 50)
    
    # Check for balance
    if len([count for count in [answer_counts.get(opt, 0) for opt in standard_options] if count > 0]) >= 2:
        std_counts = [answer_counts.get(opt, 0) for opt in standard_options]
        max_count = max(std_counts)
        min_count = min([c for c in std_counts if c > 0]) if any(std_counts) else 0
        
        if max_count > 0 and min_count > 0:
            balance_ratio = max_count / min_count if min_count > 0 else float('inf')
            if balance_ratio > 2.0:
                print(f"⚠️  Imbalanced distribution detected (ratio: {balance_ratio:.1f}:1)")
            else:
                print(f"✅ Well-balanced distribution (ratio: {balance_ratio:.1f}:1)")

def analyze_multiple_csvs(csv_patterns: List[str]):
    """Analyze multiple CSV files matching the given patterns"""
    print("🔍 Answer Option Distribution Analysis")
    print("=" * 55)
    
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
    
    # Store data for overall summary
    overall_counts = Counter()
    total_all_tasks = 0
    file_summaries = {}
    
    # Analyze each file
    for csv_file in all_files:
        answer_counts, total_tasks = analyze_answer_distribution(csv_file)
        print_answer_analysis(csv_file, answer_counts, total_tasks)
        
        # Accumulate for overall summary
        overall_counts.update(answer_counts)
        total_all_tasks += total_tasks
        file_summaries[csv_file] = {
            'counts': answer_counts,
            'total': total_tasks
        }
    
    # Print overall summary
    if len(all_files) > 1:
        print_overall_summary(overall_counts, total_all_tasks, file_summaries)

def print_overall_summary(overall_counts: Counter, total_all_tasks: int, file_summaries: Dict):
    """Print an overall summary across all CSV files"""
    print("\n" + "=" * 70)
    print("📈 OVERALL SUMMARY ACROSS ALL FILES")
    print("=" * 70)
    
    if total_all_tasks == 0:
        print("❌ No tasks found across all files")
        return
    
    print(f"Total tasks across all files: {total_all_tasks}")
    print()
    
    # Overall distribution
    print("Overall Answer Distribution:")
    print("-" * 60)
    
    standard_options = ['A', 'B', 'C', 'D']
    
    for option in standard_options:
        count = overall_counts.get(option, 0)
        percentage = (count / total_all_tasks) * 100 if total_all_tasks > 0 else 0
        bar_length = int(percentage / 2)
        bar = "█" * bar_length
        
        print(f"Option {option:<3} {count:>5} ({percentage:>5.1f}%) {bar}")
    
    # Non-standard answers
    non_standard = {k: v for k, v in overall_counts.items() if k not in standard_options}
    if non_standard:
        print("\nNon-standard answers overall:")
        for answer, count in sorted(non_standard.items()):
            percentage = (count / total_all_tasks) * 100
            print(f"'{answer}':{' ' * (6-len(answer))}{count:>5} ({percentage:>5.1f}%)")
    
    print("-" * 60)
    
    # Balance analysis
    std_counts = [overall_counts.get(opt, 0) for opt in standard_options]
    if any(std_counts):
        max_count = max(std_counts)
        min_count = min([c for c in std_counts if c > 0])
        
        if min_count > 0:
            balance_ratio = max_count / min_count
            print(f"\n📊 Balance Analysis:")
            print(f"   Max/Min ratio: {balance_ratio:.2f}:1")
            
            if balance_ratio <= 1.5:
                print("   ✅ Excellent balance")
            elif balance_ratio <= 2.0:
                print("   ✅ Good balance")
            elif balance_ratio <= 3.0:
                print("   ⚠️  Moderate imbalance")
            else:
                print("   ❌ Significant imbalance")
    
    # File comparison table
    print(f"\n📋 File Comparison Table:")
    print("-" * 90)
    
    # Header
    header = f"{'File':<35}{'A':>8}{'B':>8}{'C':>8}{'D':>8}{'Other':>8}{'Total':>8}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for filename, data in file_summaries.items():
        short_filename = os.path.basename(filename)
        if len(short_filename) > 33:
            short_filename = short_filename[:30] + "..."
        
        counts = data['counts']
        total = data['total']
        
        a_count = counts.get('A', 0)
        b_count = counts.get('B', 0)
        c_count = counts.get('C', 0)
        d_count = counts.get('D', 0)
        other_count = total - (a_count + b_count + c_count + d_count)
        
        row = f"{short_filename:<35}{a_count:>8}{b_count:>8}{c_count:>8}{d_count:>8}{other_count:>8}{total:>8}"
        print(row)

def main():
    """Main function to run the analysis"""
    
    print("Choose analysis mode:")
    print("1. Analyze all CSV files in current directory")
    print("2. Analyze generated_*.csv files only")
    print("3. Analyze specific file")
    print("4. Analyze original SAT file only")
    
    try:
        choice = input("\nEnter choice (1-4) or just press Enter for option 2: ").strip()
        
        if choice == "1":
            patterns = ["*.csv"]
        elif choice == "3":
            filename = input("Enter CSV filename: ").strip()
            if not filename.endswith('.csv'):
                filename += '.csv'
            patterns = [filename]
        elif choice == "4":
            patterns = ["SAT*.csv"]
        else:  # Default to option 2
            patterns = ["generated_*.csv"]
        
        analyze_multiple_csvs(patterns)
        
    except KeyboardInterrupt:
        print("\n\n👋 Analysis cancelled by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")

if __name__ == "__main__":
    main() 