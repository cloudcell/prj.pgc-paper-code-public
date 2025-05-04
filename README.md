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

### For More Information, Visit This Repo:
https://github.com/cloudcell/pgc/


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

2025-05-02 19:03

Training on Go Rin No Sho -- VERBATIM:
![Screenshot 2025-05-02 at 19-01-23 TensorBoard](https://github.com/user-attachments/assets/541d8ffd-0600-4d64-b7c8-b7db04178717)

2025-05-02 20:22

Future Work:
Implement https://en.wikipedia.org/wiki/Brainfuck within a PGC Unit.

2025-05-02 21:03

Stopping wrongly formed AIU training operation (training set with 'holes' due to a bug):
![Screenshot 2025-05-02 at 21-01-48 TensorBoard](https://github.com/user-attachments/assets/e1dc50cb-f20a-42d0-8ce6-1ada71c5d3ea)

2025-05-02 21:22

Started AIU training operation on a corrected 'addition' training dataset:
![Screenshot 2025-05-02 at 21-20-10 TensorBoard](https://github.com/user-attachments/assets/6472a5e2-512d-4987-9ec7-19ad200b4f62)

2025-05-03 01:07

Success!!!
The algorithm can now pack and recover data based on an input string:
```
Select folder number (or 'c' to cancel): 15
Loading checkpoint: model_20250502_153922_epoch_37.pt
Model loaded successfully!

Text Generation Menu
=========================
1. Select checkpoint folder
2. Generate text from input
3. Generate text from empty input
4. Set number of tokens to generate
5. Generate test response
6. Exit
=========================
Current checkpoint folder: checkpoints_20250502_153917
Loaded checkpoint: model_20250502_153922_epoch_37.pt
Current tokens to generate: 100
 
Enter your choice (1-6): 3

Generating text from empty input...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 114.51it/s]

Generated text:
--------------------------------------------------
<|sot|>                                                                                                  . . . . . . he che ". c c Sho" is "cced" is a macy" an a che sch on. Grchi" lis boch". Se men in "hi<|eot|>
--------------------------------------------------

Generation time: 0.87s
Model calls: 100
Model call speed: 114.33 calls/second

Text Generation Menu
=========================
1. Select checkpoint folder
2. Generate text from input
3. Generate text from empty input
4. Set number of tokens to generate
5. Generate test response
6. Exit
=========================
Current checkpoint folder: checkpoints_20250502_153917
Loaded checkpoint: model_20250502_153922_epoch_37.pt
Current tokens to generate: 100

Generation Statistics:
Total model calls: 100
Total generation time: 0.87s
Average time per model call: 8.75ms
Model call speed: 114.33 calls/second
 
Enter your choice (1-6): 2
 
Enter your text (will use last 98 chars if longer): Miyamoto Musashi

Generating text...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 127.88it/s]

Generated text:
--------------------------------------------------
<|sot|>Miyamoto Musashi at his or cpreneming. I lisurshed with 
anemy. 

Shere of the Fictoci"iustn tepagoly rear to use wh<|eot|>
--------------------------------------------------

Generation time: 0.78s
Model calls: 100
Model call speed: 127.82 calls/second

Text Generation Menu
=========================
1. Select checkpoint folder
2. Generate text from input
3. Generate text from empty input
4. Set number of tokens to generate
5. Generate test response
6. Exit
=========================
Current checkpoint folder: checkpoints_20250502_153917
Loaded checkpoint: model_20250502_153922_epoch_37.pt
Current tokens to generate: 100

Generation Statistics:
Total model calls: 200
Total generation time: 1.66s
Average time per model call: 8.28ms
Model call speed: 120.70 calls/second
 
Enter your choice (1-6): 2
 
Enter your text (will use last 98 chars if longer): Miyamoto Musashi was born in 1584, in a Japan struggling to recover from more than four

Generating text...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 128.78it/s]

Generated text:
--------------------------------------------------
<|sot|>Miyamoto Musashi was born in 1584, in a Japan struggling to recover from more than four 
ceptins as a some sall or rit 
comet a spirten hears afteriss of the swerd. And strategy is expech<|eot|>
--------------------------------------------------

Generation time: 0.78s
Model calls: 100
Model call speed: 128.70 calls/second

Text Generation Menu
=========================
1. Select checkpoint folder
2. Generate text from input
3. Generate text from empty input
4. Set number of tokens to generate
5. Generate test response
6. Exit
=========================
Current checkpoint folder: checkpoints_20250502_153917
Loaded checkpoint: model_20250502_153922_epoch_37.pt
Current tokens to generate: 100

Generation Statistics:
Total model calls: 300
Total generation time: 2.43s
Average time per model call: 8.11ms
Model call speed: 123.26 calls/second
 
Enter your choice (1-6): 2
 
Enter your text (will use last 98 chars if longer): place in fencing halls and before shrines, in the streets and within castle walls. Duels were

Generating text...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 125.93it/s]

Generated text:
--------------------------------------------------
<|sot|>place in fencing halls and before shrines, in the streets and within castle walls. Duels were 
fought to the death or until one of the contestants was disabled, but a few generations 
after Mus<|eot|>
--------------------------------------------------

Generation time: 0.79s
Model calls: 100
Model call speed: 125.86 calls/second

Text Generation Menu
=========================
1. Select checkpoint folder
2. Generate text from input
3. Generate text from empty input
4. Set number of tokens to generate
5. Generate test response
6. Exit
=========================
Current checkpoint folder: checkpoints_20250502_153917
Loaded checkpoint: model_20250502_153922_epoch_37.pt
Current tokens to generate: 100

Generation Statistics:
Total model calls: 400
Total generation time: 3.23s
Average time per model call: 8.07ms
Model call speed: 123.90 calls/second
 
Enter your choice (1-6): 4
 
Enter number of tokens to generate (1-1024*16): 1024

Set number of tokens to: 1024

Text Generation Menu
=========================
1. Select checkpoint folder
2. Generate text from input
3. Generate text from empty input
4. Set number of tokens to generate
5. Generate test response
6. Exit
=========================
Current checkpoint folder: checkpoints_20250502_153917
Loaded checkpoint: model_20250502_153922_epoch_37.pt
Current tokens to generate: 1024

Generation Statistics:
Total model calls: 400
Total generation time: 3.23s
Average time per model call: 8.07ms
Model call speed: 123.90 calls/second
 
Enter your choice (1-6): 2
 
Enter your text (will use last 98 chars if longer): place in fencing halls and before shrines, in the streets and within castle walls. Duels were

Generating text...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [00:07<00:00, 131.29it/s]

Generated text:
--------------------------------------------------
<|sot|>place in fencing halls and before shrines, in the streets and within castle walls. Duels were 
fought to the death or until one of the contestants was disabled, but a few generations 
after Musashi's time the "shinai", a pliable bamboo sword, and later padded fencing 
armour, came to be widely used, so the chances of injury were greatly reduced. The 
samurai studied with all kinds of weapons: halberds, sticks, swords, chain and sickle, and 
others. Many schools using such weapons survive in traditional form in Japan today. 

To train in Kendo one must subjugate the self, bear the pain of gruelling practise, 
and cultivate a level mind in the face of peril. But the Way of the sword means not only 
fencing training but also living by the code of honour of the samurai elite. Warfare was the 
spirit of the samurai's everyday life, and he could face death as if it were a domestic 
routine. The meaning of life and death by the sword was mirrored in the everyday conduct 
of the feudal Japanese, and he who realised the resolute acceptance of death at any 
moment in his everyday life was a master of the sword.<|eot|>
--------------------------------------------------

Generation time: 7.80s
Model calls: 1024
Model call speed: 131.29 calls/second
```

Here are the training parameters:

![image](https://github.com/user-attachments/assets/3a81936c-4dfe-4919-8881-be460e0f510e)

---
2025-05-03 01:59

In a similar way, one could train lambda calculus (or at least addition / multiplication tables) to PGC.
Hopefully, I will soon release the results on how PGC has learned 1-3-digit addition function (in a few hours)...

For now, here's the progress as of 02:08

red line -- old multiplication table (defective); green line -- old addition table (incomplete) ; blue line -- new addition table (complete)... 

![Screenshot 2025-05-03 at 02-07-27 TensorBoard](https://github.com/user-attachments/assets/94ac1876-7d41-4718-a1d5-2ca1c4dc1d78)

seems to train well so far...

2025-05-03 10:06

AIU training is going well so far. Here's the frequency response to calculations and some alphanumeric input (details to be provided later):

![de-novo-digits-alphanumeric-freq-response](https://github.com/user-attachments/assets/3338012a-4425-4c54-88d1-d7f9e8b4aa11)


2025-05-04 00:07

AIU model_epoch_2 frequency response:

Linear Scale:
![image](https://github.com/user-attachments/assets/16a6249c-4741-4027-8120-6885fa5bdfa5)

Log Scale:
![image](https://github.com/user-attachments/assets/d1b64eaf-c152-4cea-9242-4aa079d18f30)


Selected Snippet:
Linear Scale:
![image](https://github.com/user-attachments/assets/fcccd61d-5c45-4d79-bee7-34b8dcf9598b)
Log Scale:
![image](https://github.com/user-attachments/assets/e4b5781c-a5e6-4dc3-9e52-5ea29eb47cd6)

2025-05-04 03:33
AIU training is going well so far... (see blue lines):
![Screenshot 2025-05-04 at 03-32-27 TensorBoard](https://github.com/user-attachments/assets/db1f3ef5-56f5-495f-89d7-d749e1a8ddc7)

2025-05-05 05:30
AIU training is still going well... (see blue lines):
![Screenshot 2025-05-05 at 05-28-46 TensorBoard](https://github.com/user-attachments/assets/1859b190-7409-43f4-bbd3-e2d31d28fcc4)


---
If you are having trouble training your model, you can reach me at alex@cloudcell.nz .


