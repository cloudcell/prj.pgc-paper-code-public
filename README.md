# Code for Paper: "Polymorphic Graph Classifier"
### http://dx.doi.org/10.13140/RG.2.2.15744.55041

| Design | Implementation | IDE | Dates | License |
| ------ | -------------- | ---- | ---- | ------- |
| Alexander Bikeyev | Inception Labs + Sonnet 3.7 | Windsurf |2025-04-17/present | AGPL v3 |

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

[https://youtu.be/axJjPfKU-U4](https://www.youtube.com/watch?v=tjqQ6yWCxCM)


# Signatures

AMU (early version)
![Figure_1](https://github.com/user-attachments/assets/af0b186d-a119-40e8-afea-581e51e456d7)

AMU (still does not compute 1*1), it appears that AMU cannot perform and will not perform multiplication by design.
![Figure_2](https://github.com/user-attachments/assets/225c5dcc-445c-4ab9-bc44-f385ec78a4bc)

AMU Training progress (as of right now... to be stopped.):
![image](https://github.com/user-attachments/assets/4b1d6366-8ef4-45a8-b341-2b8e77fa96b7)

AIU ("arithmetic integration unit" seems to be the only unit that 'behaves' relatively well mathematically).
...hopefully I will demonstrate this soon.

(I am somewhat worried that we're putting so much trust in LLMs that are nowhere near the precision in computation to this...).
(As for multiplication, I'm afraid it's not even possible to implement it reliably at all :( ).
Let's see...

# Here the PGC Is Learning Arithmetic

The screenshot below demonstrates a dashboard with the learning progress of the algorithm learning arithmetic operations:
- green -- addition
- red -- multiplication

![image](https://github.com/user-attachments/assets/4eb31716-8b99-4340-8bc9-317f28023fc4)

And here are the top-5 paths learned during initial 573 steps of learning multiplication:

![test](https://github.com/user-attachments/assets/3df2a71b-46d4-4e9b-9d4c-a33e4d009b29)


# Here the PGC Is Learning to Generate Texts

And here is the dynamics observed during NLP tasks:

![Screenshot 2025-04-24 at 17-12-13 TensorBoard](https://github.com/user-attachments/assets/383715ef-e86d-492a-bef6-6a0bc6202c6a)

![test_nlp](https://github.com/user-attachments/assets/a592aa7a-73d1-4f07-9084-8178e8d73a67)


# Statistics Visualisation Tool:

![image](https://github.com/user-attachments/assets/8016ee33-5c04-42f3-ba25-5f2b60406759)

# Dataset Viewer Tool:

![image](https://github.com/user-attachments/assets/9eb9fc32-2629-4733-b5a6-5182aa50f544)

# Algorithm Capabilities:

![image](https://github.com/user-attachments/assets/90f090ce-2b81-4067-8956-3ab53d53c903)

# Research Log:

2025-05-02 06:34

AIU Training:

Cleaner spectrum now:

![Figure_1,0,add,permute](https://github.com/user-attachments/assets/422ed70a-c8a5-4fac-9837-f88b6fde31b2)


2025-05-02 10:11

![Screenshot 2025-05-02 at 10-03-37 TensorBoard](https://github.com/user-attachments/assets/40768652-3052-4e08-8260-684f083c2402)

2025-05-02 10:44

![Figure_update](https://github.com/user-attachments/assets/ad311fea-bd77-475e-8d43-a5a4d82e194a)

2025-05-02 15:29

Running Compression Over the Book "Go Rin No Sho" by Miyamoto Musashi

![image_2025-05-02_15-28-17](https://github.com/user-attachments/assets/d3422514-765c-45db-a63f-61e729b2925f)

```
Enter your text (will use last 98 chars if longer): Go Rin No Sho

Generating text...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [00:07<00:00, 131.93it/s]

Generated text:
--------------------------------------------------
<|sot|>Go Rin No Sholl". Wheneming men attained o her unemy mov muan and wastraile with the 
spinking it is and 
an oy to and man of his sime for the Was of strategy", hi schools cade. in unceusl a 
the The shate it is de weaple evere and end the chowlere is a 
gith reccine. The Was the rantem ins in, strone shenethet within the thone with a pis of ry 
aroptent for siel, and cutting 
po sword sword more. 

The wreaties one than sted thohene. Whth the differ, sunveris abchog 
pen the came 
the 
mne. 

In you can oen we cutten oithodici as a straction sorrd. In sword with the "oaytorn 



"The Lo" Dife on the bacomo, comes:  woidt lown a fighevely it i greas. The enenet 
dace a 
af he him. The way of kno the exply ders wonf. Thit you offerytes are came and with the enemy as it is coured his interny. 

rothtan, 



''' 

Thelere: The man wo kear ownt man sing Kally in diming and atters pupecting and 
thin ulace, 



' Tenfires to hil pochtition curalegy it the eign coord of the fare iout can with the handit selers the enemy. 

Thit<|eot|>
--------------------------------------------------

Generation time: 7.76s
Model calls: 1024
Model call speed: 131.92 calls/second
```

Not perfect because a portion of the text was used for validation ONLY! (Need that piece for the compression as well!)
To be continued, hopefully...

---
If you are having trouble training your model, you can reach me at alex@cloudcell.nz .


