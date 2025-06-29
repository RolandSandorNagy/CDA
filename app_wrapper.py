# app_wrapper.py

import sys
from app import main  # This imports the main function from app.py

if len(sys.argv) < 3:
    print("Usage: python app_wrapper.py <asset_id> <manual_description>")
    sys.exit(1)

assetID = sys.argv[1]
manualDescription = sys.argv[2]

main(assetID, manualDescription)
