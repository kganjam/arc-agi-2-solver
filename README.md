# ARC AGI Challenge Solver

An interactive web application for solving ARC (Abstraction and Reasoning Corpus) AGI challenges with an enhanced grid editor, AI assistance, and comprehensive puzzle navigation.

## Features

### 🎨 Grid Editor
- **Interactive color palette** - 10 ARC colors (0-9) with visual selection
- **Click-to-paint** grid cells with selected color
- **Adjustable grid size** - Resize answer grid from 1x1 to 30x30
- **Copy from input** - Start with the test input as template
- **Clear grid** - Reset all cells to black (color 0)

### 🧩 Puzzle Viewer
- **Full dataset support** - Load entire ARC AGI dataset
- **Navigation controls** - Previous/Next buttons and direct puzzle number input
- **Training examples** - View input/output pairs for pattern learning
- **Test cases** - Solve the test inputs with the grid editor

### 🤖 AI Assistant
- **Context-aware chat** - AI has access to current puzzle data
- **Pattern analysis** - Get insights about grid transformations
- **Grid summaries** - Automatic analysis of grid sizes and structure

### 📊 Answer Validation
- **Submit answers** - Check your solutions
- **Visual feedback** - Success/error messages for submissions

## Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ArcAGI
```

2. Run the application:
```bash
python run.py
```

This will:
- Create/activate a virtual environment
- Install all required dependencies
- Start the web server on port 8050
- Launch Microsoft Edge browser (or default browser)

### Manual Setup (Optional)

If you prefer manual setup:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python arc_app_enhanced.py
```

Then open http://localhost:8050 in your browser.

## Usage Guide

### Solving Puzzles

1. **Navigate to a puzzle** using the Previous/Next buttons or enter a puzzle number
2. **Study the training examples** to identify the transformation pattern
3. **Use the grid editor** to create your answer:
   - Select a color from the palette
   - Click grid cells to paint them
   - Resize the grid if needed
   - Use "Copy Input" to start with the test input
4. **Submit your answer** to check if it's correct
5. **Ask the AI Assistant** for help analyzing patterns

### Keyboard Shortcuts
- **Enter** in chat input - Send message
- **Enter** in puzzle number input - Go to puzzle

## Project Structure

```
ArcAGI/
├── run.py                  # Main launcher with venv management
├── arc_app_enhanced.py     # Enhanced FastAPI application
├── requirements.txt        # Python dependencies
├── data/
│   └── arc_agi/           # ARC puzzle datasets
├── logs/                   # Application logs
└── venv/                   # Virtual environment (auto-created)
```

## Configuration

### Change Port
Edit `arc_app_enhanced.py` and modify the port number:
```python
uvicorn.run(app, host="0.0.0.0", port=8050)  # Change 8050 to desired port
```

### Dataset Location
Puzzles are loaded from `data/arc_agi/` directory. Place your ARC dataset JSON files there.

## Development

### Adding Custom Puzzles
Edit the `load_arc_dataset()` function in `arc_app_enhanced.py` to add custom puzzles:

```python
puzzles.append({
    "id": "custom_001",
    "dataset": "custom",
    "train": [...],
    "test": [...]
})
```

### Extending the AI Assistant
Modify the `/api/chat` endpoint to add more sophisticated pattern analysis or integrate with external AI services.

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 8050
# On Windows:
netstat -ano | findstr :8050
# On Linux/Mac:
lsof -i :8050

# Kill the process or use a different port
```

### Dependencies Not Installing
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Clear pip cache
pip cache purge

# Reinstall requirements
pip install -r requirements.txt --no-cache-dir
```

### Application Not Starting
Check the logs in the `logs/` directory for detailed error messages:
```bash
tail -f logs/backend_*.log
```

## Testing

Run the test suite:
```bash
python test_app.py
```

## System Requirements

- **OS**: Windows, Linux, macOS, WSL
- **Python**: 3.8 or higher
- **Browser**: Modern web browser (Chrome, Firefox, Edge, Safari)
- **RAM**: 2GB minimum
- **Disk Space**: 1GB for virtual environment and datasets

## Technologies Used

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Data Format**: JSON (ARC dataset format)
- **Logging**: Python logging module

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes. The ARC dataset is created by François Chollet.

## Acknowledgments

- [ARC Prize](https://arcprize.org/) for the challenge dataset
- François Chollet for creating the ARC benchmark
- The ARC community for insights and strategies

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the logs in `logs/` directory
3. Open an issue on GitHub

---

**Current Version**: 1.0.0  
**Last Updated**: 2025