import os
import random
import string

# Folder to store generated files
output_dir = "synthetic_anomalies"
os.makedirs(output_dir, exist_ok=True)

num_files = 5          # Number of files to generate
lines_per_file = 500   # Number of lines per file

def random_line(length=80):
    """Generate a random line of code-like text"""
    chars = string.ascii_letters + string.digits + "_"
    return "".join(random.choices(chars, k=length))

for i in range(num_files):
    filename = os.path.join(output_dir, f"anomaly_file_{i+1}.py")
    with open(filename, "w") as f:
        f.write(f"# Synthetic anomalous file {i+1}\n")
        for j in range(lines_per_file):
            if random.random() < 0.1:
                # Occasionally add a "suspicious" function call
                f.write(f'os.system("echo test {j}")\n')
            else:
                f.write(f'print("{random_line()}")\n')

print(f"✅ Generated {num_files} synthetic anomaly files in '{output_dir}'")
