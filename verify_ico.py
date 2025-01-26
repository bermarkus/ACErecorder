from PIL import Image
import os

def verify_ico(ico_path):
    """Verify ICO file contents and show details of each size"""
    try:
        img = Image.open(ico_path)
        print(f"\nAnalyzing ICO file: {ico_path}")
        print(f"File size: {os.path.getsize(ico_path):,} bytes")
        
        # Get all available sizes
        sizes = img.info.get('sizes', [])
        print(f"\nFound {len(sizes)} image sizes:")
        
        # Check each size
        for size in sizes:
            img.size = size
            img.load()
            print(f"\nSize: {size[0]}x{size[1]}")
            print(f"Mode: {img.mode}")
            print(f"Bands: {img.getbands()}")
            print(f"Has transparency: {'A' in img.getbands()}")
            
        # List missing standard sizes
        standard_sizes = [(16,16), (32,32), (48,48), (256,256)]
        missing = [size for size in standard_sizes if size not in sizes]
        if missing:
            print("\nMissing standard sizes:", missing)
            
    except Exception as e:
        print(f"Error analyzing ICO: {e}")

if __name__ == '__main__':
    ico_path = 'LogoSquare1Jan2025.ico'
    if not os.path.exists(ico_path):
        print(f"Icon file not found: {ico_path}")
    else:
        verify_ico(ico_path)
