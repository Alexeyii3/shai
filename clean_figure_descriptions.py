#!/usr/bin/env python3
"""
Remove rows with empty figure_description from SAT_questions_with_figures.csv

This script reads the CSV file, identifies rows where the figure_description column is empty,
and creates a new CSV file without those rows.
"""

import csv
import os
import shutil
from typing import List, Dict


def clean_csv_file(input_file: str, output_file: str = None) -> Dict[str, int]:
    """
    Remove rows with empty figure_description from CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (if None, overwrites input file)
    
    Returns:
        Dictionary with statistics about the cleaning operation
    """
    
    if output_file is None:
        output_file = input_file
        # Create backup
        backup_file = input_file + '.backup'
        shutil.copy2(input_file, backup_file)
        print(f"📋 Created backup: {backup_file}")
    
    stats = {
        'total_rows': 0,
        'empty_description_rows': 0,
        'kept_rows': 0,
        'header_row': 1
    }
    
    cleaned_rows = []
    
    try:
        # Read the CSV file
        with open(input_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            
            if 'figure_description' not in fieldnames:
                print("❌ Error: 'figure_description' column not found in CSV file")
                print(f"Available columns: {', '.join(fieldnames)}")
                return stats
            
            print(f"📊 Processing CSV file: {input_file}")
            print(f"📊 Found columns: {', '.join(fieldnames)}")
            print()
            
            # Process each row
            for row_num, row in enumerate(reader, start=2):  # Start at 2 because header is row 1
                stats['total_rows'] += 1
                
                # Check if figure_description is empty
                figure_desc = row.get('figure_description', '').strip()
                
                if not figure_desc:
                    stats['empty_description_rows'] += 1
                    print(f"🗑️  Removing row {row_num}: Empty figure_description")
                    # Show some context about the removed row
                    skill = row.get('skill', 'Unknown')[:50]
                    difficulty = row.get('difficulty', 'Unknown')
                    print(f"    Skill: {skill}")
                    print(f"    Difficulty: {difficulty}")
                    print()
                else:
                    stats['kept_rows'] += 1
                    cleaned_rows.append(row)
        
        # Write the cleaned CSV file
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            if cleaned_rows:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(cleaned_rows)
            else:
                # If no rows to keep, still write header
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
        
        print("✅ File cleaning completed!")
        print(f"📊 Results:")
        print(f"   Total rows processed: {stats['total_rows']}")
        print(f"   Rows with empty figure_description: {stats['empty_description_rows']}")
        print(f"   Rows kept: {stats['kept_rows']}")
        print(f"   Output file: {output_file}")
        
        if stats['empty_description_rows'] > 0:
            percentage_removed = (stats['empty_description_rows'] / stats['total_rows']) * 100
            print(f"   Percentage removed: {percentage_removed:.1f}%")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found: {input_file}")
    except Exception as e:
        print(f"❌ Error processing file: {e}")
    
    return stats


def preview_empty_rows(input_file: str, max_preview: int = 10):
    """Preview rows that would be removed (for verification before cleaning)"""
    
    print(f"🔍 Previewing rows with empty figure_description in: {input_file}")
    print("=" * 80)
    
    empty_rows_found = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            if 'figure_description' not in reader.fieldnames:
                print("❌ Error: 'figure_description' column not found in CSV file")
                return
            
            for row_num, row in enumerate(reader, start=2):
                figure_desc = row.get('figure_description', '').strip()
                
                if not figure_desc:
                    empty_rows_found += 1
                    
                    if empty_rows_found <= max_preview:
                        print(f"\n📝 Row {row_num} (would be removed):")
                        print(f"   Skill: {row.get('skill', 'Unknown')}")
                        print(f"   Difficulty: {row.get('difficulty', 'Unknown')}")
                        print(f"   Question (first 100 chars): {row.get('question_text_plain', 'Unknown')[:100]}...")
                        print(f"   Figure description: '{figure_desc}' (empty)")
            
            if empty_rows_found == 0:
                print("✅ No rows found with empty figure_description!")
            else:
                print(f"\n📊 Summary:")
                print(f"   Total rows with empty figure_description: {empty_rows_found}")
                if empty_rows_found > max_preview:
                    print(f"   (Showing first {max_preview} rows only)")
    
    except FileNotFoundError:
        print(f"❌ Error: File not found: {input_file}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function with command-line interface"""
    import sys
    
    input_file = "/Users/alexey/Desktop/SAT_Gemini/SAT_questions_with_figures.csv"
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "preview":
            preview_empty_rows(input_file)
        elif command == "clean":
            # Perform the actual cleaning
            stats = clean_csv_file(input_file)
            
            if stats['empty_description_rows'] > 0:
                print(f"\n🎉 Successfully removed {stats['empty_description_rows']} rows with empty figure_description!")
            else:
                print("\n✅ No rows needed to be removed - all figure_description fields have content!")
        
        elif command == "clean-to-new":
            # Clean to a new file (keep original)
            output_file = input_file.replace('.csv', '_cleaned.csv')
            stats = clean_csv_file(input_file, output_file)
            
            if stats['empty_description_rows'] > 0:
                print(f"\n🎉 Successfully created cleaned file: {output_file}")
                print(f"📊 Removed {stats['empty_description_rows']} rows with empty figure_description!")
            else:
                print(f"\n✅ Created cleaned file: {output_file}")
                print("📊 No rows needed to be removed - all figure_description fields have content!")
        
        elif command == "help":
            print("Usage:")
            print("  python clean_figure_descriptions.py preview       # Preview rows that would be removed")
            print("  python clean_figure_descriptions.py clean         # Clean the file (creates backup)")
            print("  python clean_figure_descriptions.py clean-to-new  # Clean to new file (keep original)")
            print("  python clean_figure_descriptions.py help          # Show this help")
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' for usage information")
    else:
        # Default: preview mode
        print("Default mode: Preview")
        print("Use 'clean' argument to actually remove rows")
        print()
        preview_empty_rows(input_file)


if __name__ == "__main__":
    main()
