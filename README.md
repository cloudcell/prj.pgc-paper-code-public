# Code for Paper: "Polimorphic Graph Classifier"

### http://dx.doi.org/10.13140/RG.2.2.15744.55041

| Author | Date | License |
| ------ | ---- | ------- |
| Alexander Bikeyev | 2025-04-20 | AGPL v3 |

### Running the Example
```
# Create a virtual environment
python3 -m venv ./.venv

# Activate the virtual environment
source ./.venv/bin/activate

# Generate the sample Dataset
# (Warning - you need ~20GB of RAM free for this)
python3 text_to_binary_dataset.py

# Run the training
python 273_addressing_txt.py
```