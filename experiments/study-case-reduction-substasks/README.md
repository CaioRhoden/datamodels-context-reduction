# Study Case: Reduction Substasks

This section has the goal of exploring in a visual comparative example the behavior of three different subtasks when restricting the context only to the task or for all possible candidates. The selection of the tasks was based on the difference in performance from the experiment "impact-context-reduction"

The subtasks are:
-  Pass sentences starting with "professor" from active to passive
-  Identify the word starting with the letter "d" in the sentence
-  Words input starting with "y" and then passed to plural just add the "s"

## Structure

-  `\data`: Where the data inputs and outputs will be presented. The entire pool of candidates was in a file too big to be committed but it can be downloaded [here](https://drive.google.com/drive/folders/1GMrGvCYQ0-oGuq88W7e5pLhQWEAsMkOP?usp=drive_link)
-  `generate_embedding_distances.py`: calculate the embedding cosine distances between the test inputs and the candidates, the model used to generate the embeddings of the test is the 'sentence-transformers/stsb-roberta-large'

## Input:

![img1](https://i.imgur.com/Xw9haRJ.png)

## Results Comparison:

![img2](https://i.imgur.com/4QR2YfK.png)
