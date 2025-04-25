# Code for Paper: "Polymorphic Graph Classifier"
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


# Podcast

https://youtu.be/axJjPfKU-U4


# Here the PGC Is Learning Arithmetic

The screenshot below demonstrates a dashboard with the learning progress of the algorithm learning arithmetic operations:
- green -- addition
- red -- multiplication

![Screenshot 2025-04-24 at 16-21-52 TensorBoard](https://github.com/user-attachments/assets/a5a9a97f-abe1-4a8d-b97b-472d73cc634a)


Here's the same dashboard with a couple of tiles zoomed in:

![Screenshot 2025-04-24 at 15-57-56 TensorBoard](https://github.com/user-attachments/assets/6a22bee8-fceb-4274-be07-79d5889b2de7)


And here are the top-5 paths learned during initial 573 steps of learning multiplication:

![test](https://github.com/user-attachments/assets/3df2a71b-46d4-4e9b-9d4c-a33e4d009b29)


# Here the PGC Is Learning to Generate Texts

And here is the dynamics observed during NLP tasks:

![Screenshot 2025-04-24 at 17-12-13 TensorBoard](https://github.com/user-attachments/assets/383715ef-e86d-492a-bef6-6a0bc6202c6a)

![test_nlp](https://github.com/user-attachments/assets/a592aa7a-73d1-4f07-9084-8178e8d73a67)

