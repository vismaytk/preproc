import codecs
import os

def fix_file_encoding(path):
    """Fix file encoding to UTF-8 without BOM."""
    try:
        # Read the file in binary mode
        with open(path, 'rb') as f:
            content = f.read()
        
        # Try to decode as UTF-8 first
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            # If that fails, try to decode as UTF-16
            try:
                text = content.decode('utf-16')
            except UnicodeDecodeError:
                # If that fails too, try to decode as ASCII
                text = content.decode('ascii', errors='ignore')
        
        # Write back as UTF-8 without BOM
        with codecs.open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Fixed encoding for {path}")
    except Exception as e:
        print(f"Error fixing {path}: {str(e)}")

# Fix all Python files in the project
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            fix_file_encoding(path) 