import csv
import glob
from collections import Counter

def quick_difficulty_analysis(csv_file):
    """Quick analysis of difficulty distribution"""
    try:
        difficulties = []
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                diff = row.get('difficulty', '').strip()
                if diff:
                    difficulties.append(diff)
        
        if not difficulties:
            return None, 0
        
        counts = Counter(difficulties)
        total = len(difficulties)
        
        # Calculate percentages
        percentages = {diff: (count/total)*100 for diff, count in counts.items()}
        
        return percentages, total
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None, 0

def main():
    # Find all CSV files
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("No CSV files found in current directory")
        return
    
    print("📊 Difficulty Distribution Analysis")
    print("=" * 50)
    
    for csv_file in sorted(csv_files):
        percentages, total = quick_difficulty_analysis(csv_file)
        
        if percentages is None:
            continue
            
        print(f"\n📁 {csv_file}")
        print(f"Total tasks: {total}")
        
        # Sort by percentage (descending)
        sorted_diff = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        
        for difficulty, percentage in sorted_diff:
            count = int((percentage/100) * total)
            print(f"  {difficulty}: {count} tasks ({percentage:.1f}%)")

if __name__ == "__main__":
    main() 