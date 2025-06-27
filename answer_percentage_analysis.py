import csv
import glob
import os
from collections import Counter
from typing import Dict, List, Tuple

def analyze_answer_percentages(csv_file: str) -> Dict[str, float]:
    """Analyze answer option percentages in a single CSV file"""
    answers = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                correct_answer = row.get('correct_answer', '').strip().upper()
                if correct_answer in ['A', 'B', 'C', 'D']:  # Only count standard options
                    answers.append(correct_answer)
        
        if not answers:
            return {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
        
        answer_counts = Counter(answers)
        total_standard = len(answers)
        
        percentages = {}
        for option in ['A', 'B', 'C', 'D']:
            count = answer_counts.get(option, 0)
            percentages[option] = (count / total_standard * 100) if total_standard > 0 else 0.0
            
        return percentages
        
    except Exception as e:
        print(f"Error reading '{csv_file}': {e}")
        return {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}

def get_standard_answer_count(csv_file: str) -> int:
    """Get count of tasks with standard A/B/C/D answers"""
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            count = 0
            for row in reader:
                correct_answer = row.get('correct_answer', '').strip().upper()
                if correct_answer in ['A', 'B', 'C', 'D']:
                    count += 1
            return count
    except Exception as e:
        return 0

def print_percentage_table(files_data: List[Tuple[str, Dict[str, float], int]]):
    """Print a clean table showing percentage distributions"""
    
    print("\n📊 ANSWER OPTION PERCENTAGE DISTRIBUTION")
    print("=" * 85)
    print("(Only includes tasks with standard A/B/C/D answers)")
    print()
    
    # Header
    header = f"{'Skill':<45}{'MC Tasks':>8}{'A %':>8}{'B %':>8}{'C %':>8}{'D %':>8}"
    print(header)
    print("-" * len(header))
    
    total_tasks = 0
    overall_percentages = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
    
    # Print each file's data
    for filename, percentages, mc_count in files_data:
        # Extract skill name from filename
        skill_name = filename.replace('generated_', '').replace('.csv', '')
        skill_name = skill_name.replace('_', ' ')
        
        # Truncate long names
        if len(skill_name) > 43:
            skill_name = skill_name[:40] + "..."
        
        # Format percentages
        a_pct = f"{percentages['A']:.1f}%"
        b_pct = f"{percentages['B']:.1f}%"
        c_pct = f"{percentages['C']:.1f}%"
        d_pct = f"{percentages['D']:.1f}%"
        
        row = f"{skill_name:<45}{mc_count:>8}{a_pct:>8}{b_pct:>8}{c_pct:>8}{d_pct:>8}"
        print(row)
        
        # Accumulate for overall calculation
        if mc_count > 0:
            total_tasks += mc_count
            for option in ['A', 'B', 'C', 'D']:
                overall_percentages[option] += (percentages[option] * mc_count / 100)
    
    # Print overall summary
    print("-" * len(header))
    
    if total_tasks > 0:
        for option in ['A', 'B', 'C', 'D']:
            overall_percentages[option] = (overall_percentages[option] / total_tasks) * 100
        
        overall_row = f"{'OVERALL AVERAGE':<45}{total_tasks:>8}"
        overall_row += f"{overall_percentages['A']:>7.1f}%"
        overall_row += f"{overall_percentages['B']:>7.1f}%"
        overall_row += f"{overall_percentages['C']:>7.1f}%"
        overall_row += f"{overall_percentages['D']:>7.1f}%"
        print(overall_row)
    
    print()
    
    # Balance analysis
    if total_tasks > 0:
        max_pct = max(overall_percentages.values())
        min_pct = min(overall_percentages.values())
        
        print("📈 BALANCE ANALYSIS:")
        print(f"   Highest: {max_pct:.1f}% | Lowest: {min_pct:.1f}%")
        
        if min_pct > 0:
            ratio = max_pct / min_pct
            print(f"   Ratio: {ratio:.1f}:1")
            
            if ratio <= 1.3:
                status = "✅ Excellent balance"
            elif ratio <= 1.6:
                status = "✅ Good balance"  
            elif ratio <= 2.0:
                status = "⚠️  Moderate imbalance"
            else:
                status = "❌ Significant imbalance"
            
            print(f"   Status: {status}")
        
        print(f"\n💡 IDEAL DISTRIBUTION: 25.0% each (A, B, C, D)")
        
        # Show deviation from ideal
        deviations = []
        for option in ['A', 'B', 'C', 'D']:
            deviation = abs(overall_percentages[option] - 25.0)
            deviations.append(deviation)
        
        avg_deviation = sum(deviations) / len(deviations)
        print(f"   Average deviation from ideal: {avg_deviation:.1f} percentage points")

def print_simple_summary(files_data: List[Tuple[str, Dict[str, float], int]]):
    """Print a simple summary of just the percentages"""
    
    print("\n🎯 QUICK SUMMARY - PERCENTAGE ONLY")
    print("=" * 50)
    
    total_tasks = 0
    overall_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    
    # Calculate overall totals
    for filename, percentages, mc_count in files_data:
        if mc_count > 0:
            total_tasks += mc_count
            for option in ['A', 'B', 'C', 'D']:
                overall_counts[option] += int(percentages[option] * mc_count / 100)
    
    # Print overall percentages
    if total_tasks > 0:
        print(f"Total multiple choice tasks: {total_tasks}")
        print()
        for option in ['A', 'B', 'C', 'D']:
            count = overall_counts[option]
            percentage = (count / total_tasks) * 100
            bar = "█" * int(percentage // 2)
            print(f"Option {option}: {percentage:5.1f}% {bar}")
        
        print()
        balance_ratio = max(overall_counts.values()) / min([v for v in overall_counts.values() if v > 0])
        print(f"Balance ratio: {balance_ratio:.1f}:1")

def main():
    """Main function to run the percentage analysis"""
    
    print("🔍 Answer Option Percentage Analysis")
    print("=" * 45)
    
    # Find all generated CSV files
    csv_files = glob.glob("generated_*.csv")
    
    if not csv_files:
        print("❌ No generated CSV files found")
        return
    
    csv_files.sort()
    print(f"Found {len(csv_files)} generated CSV files\n")
    
    # Analyze each file
    files_data = []
    
    for csv_file in csv_files:
        percentages = analyze_answer_percentages(csv_file)
        mc_count = get_standard_answer_count(csv_file)
        files_data.append((csv_file, percentages, mc_count))
    
    # Choose display format
    print("Choose display format:")
    print("1. Detailed table with all skills")
    print("2. Simple summary only")
    
    try:
        choice = input("\nEnter choice (1-2) or press Enter for detailed table: ").strip()
        
        if choice == "2":
            print_simple_summary(files_data)
        else:
            print_percentage_table(files_data)
            
    except KeyboardInterrupt:
        print("\n\n👋 Analysis cancelled by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")

if __name__ == "__main__":
    main() 