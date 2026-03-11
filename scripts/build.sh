#!/bin/bash
# StochasticSwarm Build Script
# Automated build system for the particle simulation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      StochasticSwarm Build System                ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════╝${NC}"
echo ""

# Parse command line arguments
BUILD_TYPE="Release"
CLEAN_BUILD=false
RUN_AFTER=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -r|--run)
            RUN_AFTER=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./build.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -d, --debug     Build in debug mode (with symbols)"
            echo "  -c, --clean     Clean build directory before building"
            echo "  -r, --run       Run simulation after successful build"
            echo "  -v, --verbose   Verbose build output"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./build.sh                  # Standard release build"
            echo "  ./build.sh -d -v            # Debug build with verbose output"
            echo "  ./build.sh -c -r            # Clean build and run"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Display build configuration
echo -e "${YELLOW}Build Configuration:${NC}"
echo "  Build Type: $BUILD_TYPE"
echo "  Clean Build: $CLEAN_BUILD"
echo "  Run After Build: $RUN_AFTER"
echo "  Verbose: $VERBOSE"
echo ""

# Step 1: Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${YELLOW}[1/5] Cleaning build directory...${NC}"
    rm -rf build
    echo -e "${GREEN}✓ Build directory cleaned${NC}"
else
    echo -e "${YELLOW}[1/5] Skipping clean (use -c to clean)${NC}"
fi

# Step 2: Create output directory
echo -e "${YELLOW}[2/5] Setting up output directory...${NC}"
mkdir -p output
echo -e "${GREEN}✓ Output directory ready${NC}"

# Step 3: Create build directory
echo -e "${YELLOW}[3/5] Creating build directory...${NC}"
mkdir -p build
cd build
echo -e "${GREEN}✓ Build directory created${NC}"

# Step 4: Configure with CMake
echo -e "${YELLOW}[4/5] Configuring with CMake...${NC}"
if [ "$VERBOSE" = true ]; then
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
else
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE .. > /dev/null 2>&1
fi
echo -e "${GREEN}✓ CMake configuration complete${NC}"

# Step 5: Build with make
echo -e "${YELLOW}[5/5] Compiling...${NC}"
if [ "$VERBOSE" = true ]; then
    make
else
    make > /dev/null 2>&1
fi
echo -e "${GREEN}✓ Compilation successful${NC}"

cd ..

# Display build summary
echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Build Complete!                         ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Executable:${NC} ./build/StochasticSwarm"
echo -e "${GREEN}Output Dir:${NC} ./output/"
echo ""

# Run if requested
if [ "$RUN_AFTER" = true ]; then
    echo -e "${YELLOW}Running simulation...${NC}"
    echo ""
    ./build/StochasticSwarm
fi

# Display next steps
if [ "$RUN_AFTER" = false ]; then
    echo "Next steps:"
    echo "  1. Run simulation:  ./build/StochasticSwarm"
    echo "  2. Visualize data:  python visualize_particles.py --mode animate"
    echo "  3. Run experiments: ./run_temperature_experiments.sh"
fi

echo ""
echo -e "${GREEN}Build completed successfully! 🎉${NC}"
