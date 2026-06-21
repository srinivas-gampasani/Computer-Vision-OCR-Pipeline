#!/bin/bash
# ============================================================
#  Computer Vision OCR Pipeline — Quick Start
# ============================================================
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'

echo ""
echo "============================================================"
echo "  Computer Vision OCR Pipeline"
echo "  Srinivas Gampasani · AI & ML Engineering"
echo "============================================================"
echo ""

# Check Tesseract
if ! command -v tesseract &>/dev/null; then
    echo -e "${RED}ERROR: Tesseract OCR engine not found.${NC}"
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr"
    echo "  macOS:         brew install tesseract"
    exit 1
fi
echo -e "${GREEN}Tesseract found: $(tesseract --version | head -1)${NC}"

if [ ! -f "backend/.env" ]; then
    cp backend/.env.example backend/.env
    echo -e "${YELLOW}Created backend/.env from template.${NC}"
fi

echo ""
echo -e "${BLUE}Installing Python dependencies...${NC}"
cd backend
pip install -r requirements.txt -q
cd ..
echo -e "${GREEN}Done.${NC}"

echo ""
echo -e "${BLUE}Running pipeline against sample documents...${NC}"
cd backend
python ../scripts/run_ocr.py --all-samples
cd ..

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Starting API server on http://localhost:8000${NC}"
echo -e "${GREEN}  Open frontend/index.html in your browser to use the UI${NC}"
echo -e "${GREEN}  API docs: http://localhost:8000/docs${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
