#!/bin/bash
# Download LEVIR-CD (binary building change detection)
# Total: ~2.9GB  |  Layout: {train,val,test}/{A,B,label}/

set -e
DEST=/root/autodl-tmp/LEVIR-CD
mkdir -p "$DEST"

echo "=== Method 1: OpenDataLab (recommended on CN servers) ==="
echo "Requires free account at https://openxlab.org.cn"
echo "Run: openxlab login  (then enter AK/SK from your profile)"
echo "Then re-run this script."
echo ""
if openxlab login --status 2>/dev/null | grep -q "Logged"; then
    echo "Already logged in. Downloading..."
    openxlab dataset download -r OpenDataLab/LEVIR-CD -s /raw/train.zip -t "$DEST"
    openxlab dataset download -r OpenDataLab/LEVIR-CD -s /raw/val.zip   -t "$DEST"
    openxlab dataset download -r OpenDataLab/LEVIR-CD -s /raw/test.zip  -t "$DEST"
    echo "Extracting..."
    mkdir -p "$DEST/images"
    cd "$DEST"
    unzip -q train.zip -d images/ && rm train.zip
    unzip -q val.zip   -d images/ && rm val.zip
    unzip -q test.zip  -d images/ && rm test.zip
    echo "Done: $DEST/images/{train,val,test}/{A,B,label}/"
    exit 0
fi

echo "=== Method 2: Baidu Netdisk ==="
echo "Download from LEVIR official page:"
echo "  URL:  https://pan.baidu.com/s/XXXXXX (find on http://chenhao.in/LEVIR/)"
echo "  Code: l7iv"
echo "Download all three zips, then:"
echo "  unzip train.zip -d $DEST/images/"
echo "  unzip val.zip   -d $DEST/images/"
echo "  unzip test.zip  -d $DEST/images/"
echo ""
echo "Exiting. Please login to OpenDataLab or use Baidu Netdisk."
exit 1