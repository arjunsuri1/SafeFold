#!/bin/bash

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Creating conda environment...${NC}"
conda env create -f environment.yml --solver classic --override-channels -c pytorch -c dglteam/label/cu113 -c defaults

echo -e "${BLUE}Activating environment...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SafeFold

echo -e "${BLUE}Installing pipx...${NC}"
python -m pip install --user pipx
python -m pipx ensurepath

echo -e "${BLUE}Installing amina-cli...${NC}"
pipx install amina-cli

echo -e "${BLUE}Installing gdown...${NC}"
pip install gdown

echo -e "${BLUE}Downloading DPFunc models...${NC}"
ARCHIVE="dpfunc_models.tar.gz"
gdown -O "$ARCHIVE" "https://drive.google.com/uc?id=1V0VTFTiB29ilbAIOZn0okBQWPlbOI3wN"

echo -e "${BLUE}Extracting models...${NC}"
tar -xzf "$ARCHIVE"

echo -e "${BLUE}Moving models to correct directory...${NC}"
mkdir -p SafeFold_architecture/DPFunc_fork/save_models
mv save_models/* SafeFold_architecture/DPFunc_fork/save_models/

echo -e "${BLUE}Cleaning up...${NC}"
rm -rf save_models
rm -f "$ARCHIVE"

echo -e "${GREEN}Setup complete.${NC}"
echo ""
echo -e "${YELLOW}Activate the environment with:${NC}"
echo "conda activate SafeFold"
echo ""
echo -e "${YELLOW}Then authenticate with Amina:${NC}"
echo 'amina auth set-key "YOUR_API_KEY"'
