#!/usr/bin/env python3
"""
Count the number of rows (tasks) in all generated CSV files.
This script helps track the total number of generated math tasks across all skills.
"""

import csv
import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def count_rows_in_csv(filepath: str) -> Tuple[int, str]:
    """
    Count the number of rows in a CSV file (excluding header).
    Returns (row_count, skill_name)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            row_count = len(rows)
            
            # Extract skill name from first row
            skill_name = rows[0].get('skill', 'Unknown') if rows else 'Unknown'
            
            return row_count, skill_name
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0, 'Error'


def get_file_size_mb(filepath: str) -> float:
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    except:
        return 0.0


def analyze_generated_files():
    """Analyze all generated CSV files and provide detailed statistics"""
    
    # Find all generated CSV files
    pattern = "sat/generated_*.csv"  # Default to SAT directory
    csv_files = glob.glob(pattern)
    
    # If no files found in SAT directory, try current directory
    if not csv_files:
        pattern = "generated_*.csv"
        csv_files = glob.glob(pattern)
    
    if not csv_files:
        print("❌ No generated CSV files found!")
        print(f"   Looking for files matching pattern: {pattern}")
        return
    
    print("📊 Generated CSV Files Analysis")
    print("=" * 60)
    
    total_tasks = 0
    file_stats = []
    skill_stats = defaultdict(int)
    
    # Analyze each file
    for filepath in sorted(csv_files):
        row_count, skill_name = count_rows_in_csv(filepath)
        file_size = get_file_size_mb(filepath)
        
        file_stats.append({
            'filename': os.path.basename(filepath),
            'skill': skill_name,
            'tasks': row_count,
            'size_mb': file_size
        })
        
        total_tasks += row_count
        skill_stats[skill_name] += row_count
    
    # Display detailed file statistics
    print("\n📁 Individual File Statistics:")
    print("-" * 60)
    print(f"{'Filename':<35} {'Skill':<25} {'Tasks':<8} {'Size (MB)'}")
    print("-" * 60)
    
    for stat in file_stats:
        print(f"{stat['filename']:<35} {stat['skill']:<25} {stat['tasks']:<8} {stat['size_mb']:.2f}")
    
    # Display summary statistics
    print("\n📈 Summary Statistics:")
    print("-" * 40)
    print(f"Total CSV files: {len(csv_files)}")
    print(f"Total tasks generated: {total_tasks:,}")
    print(f"Average tasks per file: {total_tasks / len(csv_files):.1f}")
    print(f"Total file size: {sum(stat['size_mb'] for stat in file_stats):.2f} MB")
    
    # Display top skills by task count
    if skill_stats:
        print("\n🏆 Skills by Task Count:")
        print("-" * 30)
        sorted_skills = sorted(skill_stats.items(), key=lambda x: x[1], reverse=True)
        for skill, count in sorted_skills[:10]:  # Show top 10
            print(f"{skill:<25}: {count:>6} tasks")
        
        if len(sorted_skills) > 10:
            print(f"... and {len(sorted_skills) - 10} more skills")
    
    # Check for potential issues
    print("\n🔍 Quality Check:")
    print("-" * 20)
    
    small_files = [stat for stat in file_stats if stat['tasks'] < 100]
    if small_files:
        print(f"⚠️  Files with < 100 tasks: {len(small_files)}")
        for stat in small_files:
            print(f"   - {stat['filename']}: {stat['tasks']} tasks")
    else:
        print("✅ All files have adequate task counts (≥100)")
    
    large_files = [stat for stat in file_stats if stat['size_mb'] > 5.0]
    if large_files:
        print(f"📦 Large files (>5MB): {len(large_files)}")
        for stat in large_files:
            print(f"   - {stat['filename']}: {stat['size_mb']:.2f} MB")
    
    # Estimate completion if target is 100 tasks per skill
    target_per_skill = 100
    complete_skills = len([stat for stat in file_stats if stat['tasks'] >= target_per_skill])
    incomplete_skills = len(file_stats) - complete_skills
    
    print(f"\n🎯 Progress Towards 100 Tasks per Skill:")
    print("-" * 40)
    print(f"Complete skills (≥100 tasks): {complete_skills}")
    print(f"Incomplete skills (<100 tasks): {incomplete_skills}")
    if incomplete_skills > 0:
        remaining_tasks = sum(max(0, target_per_skill - stat['tasks']) for stat in file_stats)
        print(f"Estimated remaining tasks needed: {remaining_tasks:,}")


def count_specific_skill(skill_pattern: str):
    """Count tasks for files matching a specific skill pattern"""
    matching_files = glob.glob(f"generated_*{skill_pattern}*.csv")
    
    if not matching_files:
        print(f"❌ No files found matching pattern: *{skill_pattern}*")
        return
    
    print(f"📊 Files matching pattern '*{skill_pattern}*':")
    print("-" * 50)
    
    total_tasks = 0
    for filepath in sorted(matching_files):
        row_count, skill_name = count_rows_in_csv(filepath)
        total_tasks += row_count
        print(f"{os.path.basename(filepath):<35}: {row_count:>6} tasks ({skill_name})")
    
    print("-" * 50)
    print(f"Total tasks for pattern '{skill_pattern}': {total_tasks}")


def export_summary_csv():
    """Export summary statistics to a CSV file"""
    csv_files = glob.glob("generated_*.csv")
    
    if not csv_files:
        print("❌ No generated CSV files found!")
        return
    
    summary_data = []
    
    for filepath in sorted(csv_files):
        row_count, skill_name = count_rows_in_csv(filepath)
        file_size = get_file_size_mb(filepath)
        
        summary_data.append({
            'filename': os.path.basename(filepath),
            'skill': skill_name,
            'task_count': row_count,
            'file_size_mb': round(file_size, 2),
            'status': 'Complete' if row_count >= 100 else 'Incomplete'
        })
    
    # Write summary CSV
    summary_filename = "generated_files_summary.csv"
    with open(summary_filename, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['filename', 'skill', 'task_count', 'file_size_mb', 'status']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)
    
    print(f"✅ Summary exported to: {summary_filename}")
    print(f"📊 Total files analyzed: {len(summary_data)}")
    print(f"📊 Total tasks: {sum(row['task_count'] for row in summary_data):,}")


def main():
    """Main function with command-line interface"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "summary" or command == "export":
            export_summary_csv()
        elif command.startswith("skill:"):
            skill_pattern = command[6:]  # Remove "skill:" prefix
            count_specific_skill(skill_pattern)
        elif command == "help":
            print("Usage:")
            print("  python count_generated_tasks.py              # Full analysis")
            print("  python count_generated_tasks.py summary      # Export summary CSV")
            print("  python count_generated_tasks.py skill:algebra # Count specific skill")
            print("  python count_generated_tasks.py help         # Show this help")
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' for usage information")
    else:
        # Default: full analysis
        analyze_generated_files()


if __name__ == "__main__":
    main()
