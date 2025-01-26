from PIL import Image
import subprocess
import os
import tempfile
import shutil
from PIL import ImageEnhance

def get_inkscape_path():
    """Get the Inkscape executable path"""
    possible_paths = [
        r"C:\Program Files\Inkscape\bin\inkscape.exe",
        r"C:\Program Files (x86)\Inkscape\bin\inkscape.exe",
        r"C:\Program Files\Inkscape\inkscape.exe",
        r"C:\Program Files (x86)\Inkscape\inkscape.exe"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    raise FileNotFoundError("Inkscape not found in standard installation paths")

def svg_to_png(svg_path, png_path, size):
    """Convert SVG to PNG using Inkscape with high quality settings"""
    try:
        inkscape_path = get_inkscape_path()
        
        # Calculate DPI based on size
        if size[0] <= 32:  # Higher DPI for small sizes
            dpi = 1200  # Much higher DPI for small icons
        elif size[0] <= 48:
            dpi = 600
        else:
            dpi = 300
        
        # Use higher DPI for better quality
        subprocess.run([
            inkscape_path,
            '--export-type=png',
            f'--export-filename={png_path}',
            f'--export-width={size[0]}',
            f'--export-height={size[1]}',
            f'--export-dpi={dpi}',
            '--export-background-opacity=0',  # Ensure transparency
            svg_path
        ], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting SVG to PNG: {e}")
        return False
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False

def svg_to_ico(svg_path, ico_path):
    """Convert SVG to ICO with multiple sizes for best quality"""
    try:
        # Windows standard icon sizes (in reverse order - largest first)
        sizes = [(256,256), (128,128), (96,96), (64,64), (48,48), 
                (40,40), (32,32), (24,24), (20,20), (16,16)]
        
        # Create temporary directory for PNG files
        with tempfile.TemporaryDirectory() as temp_dir:
            images = []
            for size in sizes:
                temp_png = os.path.join(temp_dir, f'temp_{size[0]}.png')
                
                # Convert SVG to PNG at current size
                if not svg_to_png(svg_path, temp_png, size):
                    continue
                
                # Open and convert to RGBA
                img = Image.open(temp_png)
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # Apply slight sharpening to smaller sizes
                if size[0] <= 32:
                    enhancer = ImageEnhance.Sharpness(img)
                    img = enhancer.enhance(2.0)  # Increased sharpening
                
                images.append(img)
            
            if not images:
                raise Exception("Failed to convert SVG to any size")
            
            # Save as ICO with all sizes
            images[0].save(
                ico_path,
                format='ICO',
                sizes=[(img.size) for img in images],
                append_images=images[1:],
                optimize=True
            )
            
            print(f"Successfully created {ico_path} with sizes:")
            for size in sizes:
                print(f"  - {size[0]}x{size[0]} pixels")
            return True
            
    except Exception as e:
        print(f"Error creating ICO: {e}")
        return False

if __name__ == '__main__':
    # Use the new square logo as source
    svg_path = 'LogoSquare3.svg'
    ico_path = 'LogoSquare3.ico'
    
    if not os.path.exists(svg_path):
        print(f"Source SVG file not found: {svg_path}")
    else:
        svg_to_ico(svg_path, ico_path)
