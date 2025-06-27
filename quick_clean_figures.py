#!/usr/bin/env python3
"""
Simple one-liner script to remove rows with empty figure_description
"""

import csv
import sys

def quick_clean(input_file):
    """Quick clean - removes rows with empty figure_description"""
    
    # Create backup
    backup_file = input_file + '.backup'
    import shutil
    shutil.copy2(input_file, backup_file)
    print(f"Created backup: {backup_file}")
    
    # Read and filter
    rows_kept = 0
    rows_removed = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        filtered_rows = []
        for row in reader:
            if row.get('figure_description', '').strip():
                filtered_rows.append(row)
                rows_kept += 1
            else:
                rows_removed += 1
    
    # Write filtered data
    with open(input_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames) 
        writer.writeheader()
        writer.writerows(filtered_rows)
    
    print(f"Rows kept: {rows_kept}")
    print(f"Rows removed: {rows_removed}")
    print("Done!")

if __name__ == "__main__":
    file_path = "/Users/alexey/Desktop/SAT_Gemini/SAT_questions_with_figures.csv"
    quick_clean(file_path)
