import csv
import os
import glob
from typing import Set

def get_all_skills_from_csv(csv_file: str) -> Set[str]:
    """Extract all unique skills from the original CSV"""
    skills = set()
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                skill = row.get('skill', '').strip()
                if skill:
                    skills.add(skill)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
    
    return skills

def get_generated_skills() -> Set[str]:
    """Extract skills from already generated CSV files"""
    generated_skills = set()
    
    # Find all generated CSV files
    generated_files = glob.glob("generated_*.csv")
    
    for file in generated_files:
        # Extract skill name from filename
        # Format: generated_Skill_Name.csv
        skill_part = file.replace("generated_", "").replace(".csv", "")
        # Convert underscores back to spaces and handle special cases
        skill_name = skill_part.replace("_", " ")
        
        # Handle some special cases that might have gotten mangled
        skill_name = skill_name.replace("One variable data Distributions", "One-variable data: Distributions")
        skill_name = skill_name.replace("Two variable data Models", "Two-variable data: Models")
        skill_name = skill_name.replace("Evaluating statistical claims Observational", "Evaluating statistical claims: Observational")
        
        generated_skills.add(skill_name)
    
    return generated_skills

def create_skill_mapping() -> dict:
    """Create mapping between original skill names and generated filenames"""
    mapping = {}
    generated_files = glob.glob("generated_*.csv")
    
    for file in generated_files:
        # Read the first few rows to get the actual skill name
        try:
            with open(file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                first_row = next(reader, None)
                if first_row and 'skill' in first_row:
                    actual_skill = first_row['skill'].strip()
                    mapping[actual_skill] = file
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    return mapping

def main():
    original_csv = "SAT Questions - SAT Math No Graph.csv"
    
    print("🔍 Checking Generated vs Remaining Skills")
    print("=" * 50)
    
    # Get all skills from original CSV
    all_skills = get_all_skills_from_csv(original_csv)
    print(f"📊 Total skills in original CSV: {len(all_skills)}")
    
    # Get mapping of generated skills
    skill_mapping = create_skill_mapping()
    generated_skill_names = set(skill_mapping.keys())
    
    print(f"✅ Already generated: {len(generated_skill_names)}")
    print(f"⏳ Remaining to generate: {len(all_skills - generated_skill_names)}")
    
    print("\n✅ ALREADY GENERATED SKILLS:")
    print("-" * 40)
    for skill in sorted(generated_skill_names):
        filename = skill_mapping.get(skill, "Unknown file")
        print(f"  ✓ {skill}")
        print(f"    File: {filename}")
    
    print("\n⏳ REMAINING SKILLS TO GENERATE:")
    print("-" * 40)
    remaining_skills = all_skills - generated_skill_names
    
    if remaining_skills:
        for skill in sorted(remaining_skills):
            print(f"  ⏳ {skill}")
    else:
        print("  🎉 All skills have been generated!")
    
    print(f"\n📈 SUMMARY:")
    print(f"  Total skills: {len(all_skills)}")
    print(f"  Generated: {len(generated_skill_names)} ({len(generated_skill_names)/len(all_skills)*100:.1f}%)")
    print(f"  Remaining: {len(remaining_skills)} ({len(remaining_skills)/len(all_skills)*100:.1f}%)")
    
    return remaining_skills

if __name__ == "__main__":
    main() 