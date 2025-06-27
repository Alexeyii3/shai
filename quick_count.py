#!/usr/bin/env python3
"""
Quick row counter for generated CSV files.
Simple script to get total task count across all generated files.
"""

import csv
import glob
import os


def quick_count():
    """Quick count of all rows in generated CSV files"""
    csv_files = glob.glob("generated_*.csv")
    
    if not csv_files:
        print("No generated CSV files found!")
        return
    
    total_tasks = 0
    file_count = 0
    
    print("Counting tasks in generated CSV files...")
    print("-" * 40)
    
    for filepath in sorted(csv_files):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                # Count rows excluding header
                row_count = sum(1 for row in csv.DictReader(file))
                total_tasks += row_count
                file_count += 1
                
                # Show filename and count
                filename = os.path.basename(filepath)
                print(f"{filename:<35}: {row_count:>6} tasks")
                
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    print("-" * 40)
    print(f"Total files: {file_count}")
    print(f"Total tasks: {total_tasks:,}")
    
    if file_count > 0:
        avg_tasks = total_tasks / file_count
        print(f"Average per file: {avg_tasks:.1f} tasks")


if __name__ == "__main__":
    quick_count()
