"""
Download the unemployment dataset from Kaggle.

This script downloads the dataset if it doesn't already exist locally.
You need to have kaggle CLI configured with your API credentials.

Setup:
1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account Settings > API > Create New Token
3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\.kaggle\ (Windows)

Alternative: Download manually from:
https://www.kaggle.com/datasets/pienik/unemployment-in-canada-by-province-1976-present
"""

import os
import sys
from pathlib import Path


def download_from_kaggle():
    """Download dataset using Kaggle API."""
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'pienik/unemployment-in-canada-by-province-1976-present',
            path='.',
            unzip=True
        )
        print("✓ Dataset downloaded successfully!")
        return True
    except ImportError:
        print("Kaggle package not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        return False


def download_fallback():
    """Provide instructions for manual download."""
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("""
1. Go to: https://www.kaggle.com/datasets/pienik/unemployment-in-canada-by-province-1976-present

2. Click "Download" (you may need to sign in)

3. Extract the ZIP file to this project directory

4. Ensure 'Unemployment_Canada_1976_present.csv' is in the root folder
    """)
    print("="*60)


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    data_file = project_root / "Unemployment_Canada_1976_present.csv"
    
    if data_file.exists():
        print(f"✓ Data file already exists: {data_file}")
        return 0
    
    print("Attempting to download dataset...")
    
    # Change to project root
    os.chdir(project_root)
    
    if not download_from_kaggle():
        download_fallback()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
