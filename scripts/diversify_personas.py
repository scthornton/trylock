#!/usr/bin/env python3
"""
Diversify personas in generated trajectories.

Replaces repetitive personas (like "Dr. Sarah Chen") with varied alternatives
while preserving attack content and structure.
"""

import json
import random
import re
from pathlib import Path

# Diverse persona pool
FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Avery",
    "Cameron", "Dakota", "Reese", "Skyler", "Jamie", "Drew", "Blake", "Parker",
    "Maya", "Priya", "Wei", "Yuki", "Fatima", "Elena", "Kofi", "Dmitri",
    "Isabella", "Marcus", "Aisha", "Chen", "Raj", "Sofia", "Omar", "Kenji",
    "Amara", "Viktor", "Zara", "Lucas", "Nia", "Hassan", "Ingrid", "Miguel"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Chen", "Patel", "Kim", "Garcia", "Martinez",
    "Anderson", "Thompson", "Nakamura", "Singh", "Okonkwo", "Mueller", "Santos",
    "Ivanov", "Larsson", "Cohen", "Fitzgerald", "Yamamoto", "Ali", "Kowalski",
    "Bergström", "Rossi", "Andersen", "Johansson", "Hansen", "Nielsen", "Petrov"
]

TITLES = ["Dr.", "Prof.", "", "", "", ""]  # Empty more common (no title)

INSTITUTIONS = [
    "Stanford", "MIT", "Carnegie Mellon", "Berkeley", "Georgia Tech",
    "University of Washington", "Cornell", "Princeton", "Columbia", "UCLA",
    "University of Michigan", "ETH Zurich", "Oxford", "Cambridge", "Imperial College",
    "Tsinghua University", "University of Toronto", "NUS", "EPFL", "TU Munich",
    "a major research university", "a leading tech company", "a cybersecurity firm",
    "an AI safety lab", "a defense contractor", "a Fortune 500 company"
]

ROLES = [
    "cybersecurity researcher", "security analyst", "AI safety researcher",
    "ML engineer", "software developer", "IT professional", "security consultant",
    "penetration tester", "threat analyst", "data scientist", "systems architect",
    "DevSecOps engineer", "cloud security specialist", "incident responder",
    "security auditor", "privacy engineer", "compliance analyst"
]

STUDENT_ROLES = [
    "PhD student", "graduate student", "master's student", "doctoral candidate",
    "research assistant", "postdoc", "visiting researcher", "research fellow"
]

PROJECTS = [
    "AI safety mechanisms", "adversarial robustness", "prompt injection defenses",
    "LLM security", "machine learning security", "AI alignment", "red teaming AI",
    "automated vulnerability detection", "social engineering defense",
    "phishing detection", "malware analysis", "threat modeling",
    "security awareness training", "AI governance", "responsible AI"
]


def generate_persona():
    """Generate a random diverse persona."""
    title = random.choice(TITLES)
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)

    if title:
        name = f"{title} {first} {last}"
    else:
        name = f"{first} {last}"

    # Decide if student or professional
    if random.random() < 0.4:
        role = random.choice(STUDENT_ROLES)
        intro = f"I'm {name}, a {role}"
    else:
        role = random.choice(ROLES)
        intro = f"I'm {name}, a {role}"

    # Add institution sometimes
    if random.random() < 0.6:
        inst = random.choice(INSTITUTIONS)
        intro += f" at {inst}"

    return name, intro


def replace_persona_in_text(text):
    """Replace common persona patterns with diverse alternatives."""

    # Pattern for "Dr. Sarah Chen" style
    sarah_chen_pattern = r"(?:Dr\.\s*)?Sarah\s+Chen"

    # Pattern for generic researcher intros
    researcher_intro_pattern = r"I'm\s+(?:Dr\.\s*)?[A-Z][a-z]+\s+[A-Z][a-z]+,?\s*(?:a\s+)?(?:cybersecurity\s+)?researcher"

    # Pattern for student intros
    student_intro_pattern = r"I'm\s+(?:a\s+)?(?:PhD|graduate|master's)\s+student"

    name, new_intro = generate_persona()

    # Replace Sarah Chen specifically
    if re.search(sarah_chen_pattern, text, re.IGNORECASE):
        text = re.sub(sarah_chen_pattern, name, text, flags=re.IGNORECASE)
        # Also try to update the role description
        text = re.sub(
            r"cybersecurity researcher at (?:Stanford|MIT|Berkeley)",
            f"{random.choice(ROLES)} at {random.choice(INSTITUTIONS)}",
            text
        )

    # Replace generic researcher patterns
    if re.search(researcher_intro_pattern, text, re.IGNORECASE):
        text = re.sub(researcher_intro_pattern, new_intro, text, count=1)

    return text


def diversify_trajectory(traj):
    """Diversify personas in a single trajectory."""
    modified = False

    for turn in traj.get('conversation', []):
        if turn.get('role') == 'user':
            original = turn.get('content', '')

            # Check if this has a common persona pattern
            if any(p in original for p in ['Sarah Chen', 'Dr. Sarah', 'cybersecurity researcher at Stanford',
                                            'cybersecurity researcher at MIT']):
                turn['content'] = replace_persona_in_text(original)
                modified = True

    return traj, modified


def process_file(filepath):
    """Process a single file and diversify personas."""
    trajectories = []
    modified_count = 0

    with open(filepath) as f:
        for line in f:
            if line.strip():
                traj = json.loads(line)
                traj, modified = diversify_trajectory(traj)
                if modified:
                    modified_count += 1
                trajectories.append(traj)

    # Write back
    with open(filepath, 'w') as f:
        for traj in trajectories:
            f.write(json.dumps(traj) + '\n')

    return len(trajectories), modified_count


def main():
    data_dir = Path(__file__).parent.parent / "data" / "tier1_open" / "attacks"
    files = list(data_dir.glob("*_generated.jsonl"))

    print("=" * 60)
    print("PERSONA DIVERSIFICATION")
    print("=" * 60)

    total_modified = 0
    total_trajectories = 0

    for filepath in files:
        print(f"\nProcessing {filepath.name}...")
        count, modified = process_file(filepath)
        total_trajectories += count
        total_modified += modified
        print(f"  {modified}/{count} trajectories modified")

    print("\n" + "=" * 60)
    print(f"Total: {total_modified}/{total_trajectories} trajectories diversified")
    print("=" * 60)


if __name__ == "__main__":
    main()
