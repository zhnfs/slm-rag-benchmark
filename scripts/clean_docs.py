import re
import os

def clean_well_control_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    stack = []
    in_stack = False

    for line in lines:
        stripped = line.strip()
        
        # Detect single character lines (or small symbols) that look like a vertical stack
        # Excluding empty lines and section headers
        if len(stripped) == 1 or (len(stripped) <= 3 and stripped in ['and', 'for', 'the', 'off']):
            stack.append(stripped)
            in_stack = True
        elif in_stack:
            # Join the stack and add it
            if stack:
                combined = "".join(stack)
                # Basic cleanup of common math artifacts
                combined = combined.replace("∗", " * ").replace("=", " = ").replace("+", " + ")
                cleaned_lines.append(combined + "\n")
            stack = []
            in_stack = False
            cleaned_lines.append(line)
        else:
            # Remove LaTeX display tags if they exist
            if '{\\displaystyle' in line:
                # Keep only the content inside or a cleaned version
                match = re.search(r'\{\\displaystyle (.*?)\}', line)
                if match:
                    cleaned_lines.append(match.group(1) + "\n")
                else:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

if __name__ == "__main__":
    path = "/Users/hainingzheng/pythonCodes/slm-rag-benchmark/data/documents/Well_control.txt"
    if os.path.exists(path):
        clean_well_control_file(path)
        print(f"✅ Successfully cleaned {path}")
