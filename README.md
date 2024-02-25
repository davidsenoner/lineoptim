# lineoptim

A simple example of line parameter optimization using Adam optimizer from torch library.
The goal is to find conductor resistance by giving target voltage [%] drop at line end and line structure.

The line structure is defined in resources/line_structure.json file and consists of loads and line segments.
Each load has a load characteristics and distance from source it is connected to.
Each line segment has starting parameters for optimization, source voltage and connected components.

The example can operate single or multiple phases.
See resources/line_structure.json for a network structure example.

## Usage
```python
python main.py
```

# License
MIT

