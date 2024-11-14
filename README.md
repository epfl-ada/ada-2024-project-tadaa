# What can wikispeedia teach us about making decisions in life?

## Abstract

Our project explores principles for thoughtful choices, using insights from the Wikispeedia dataset. First, we examine how much effort players put in their games and the effect it has on their success. We find that some users may not have examined the full content of each page or made a thorough effort. This may reflect a tendency to stick to surface-level information, suggesting that better decisions often come from fully exploring available information rather than stopping at the first glance. In addition to that, we emphasize the importance to not rely solely on large language models (LLMs) for guidance, as they don't provide the best course of actions and are not always consistent. Finally, we shade the light on the importance of society. Consulting others (leveraging crowd knowledge) can also enrich decision-making, though it’s important to understand that crowds don’t always succeed; examining specific paths reveals how groupthink and misjudgments can lead to failure. Analyzing failed attempts to reach a target page in Wikispeedia, 

When we talk about performance here, we consider the shortness of the paths as the criterion.
Also, It's important to clarify that extrapolating the advice found on Wikispeedia to decision-making in life in general is an exaggeration and should be taken with a grain of salt.

## Research questions

- Does people lazyness affect their performance? Is there a noticeable difference in the page position of clicked links between successful and unsuccessful players?
- Can we always trust LLMs? Is it safe to take their answers as facts without second checking?
- Will a crowd of players do better than the same players individually? In other words, does Condorcet's jury theorem apply? If it applies, what are the recurring patterns for the cases in which the crowd fails to beat the individuals?

## Additional data

- **Generated Paths by Qwen 3B**
  We genrated paths for the 10 most player source-target pairs in order to compare LLM performance to human players performance. We discuss the generation strategy in the following section.

## Project plans & Methods

### Task 1: Analyse page position of clicked links

We obtain the x and y coordinates of the links by using Selenium. It enables us to open the html file of the articles in a simulated browser and then select the links we are interested in. We set the window size of the browser to 1920x1080 because it is the most common. When the next article in path is accessible via several links, we select the coordinates of all these links.
We run this algorithm on each path of the 50 most popular source-target peers and compare the results between the optimal paths, those of people who didn't finish the game and those of people who did.

### Task 2: Analyse LLM performance in Wikispeedia

We use Qwen3b-4-bit-quantized as it fits in our GPU resources and it gives good results. We start by doing prompt engineering in order to get the model to understand the task.
If the word returned is not in the list we keep the conversation context with the model and send this new request asking it to correct itself. <br>
By this, we make the model rethink its answer and give it another chance to correct its choice. If the model persists in choosing a word not in the list, we consider the path failed. Otherwise, we recompute the list to chose from by getting the links in page that the model chose and removing from it all the articles that we went through. This allows to avoid looping in a circular fashion indefinitely. If the target word is in the list, then we consider the path a success and we stop the algorithm. Otherwise, we reset the model context in order to make the prompt short enough to fit into our GPU resources and we redo the same steps.
We only allow the model to run for 50 steps after which we consider the path as a failure.<br>

For each source-target pair, we make the model play 100 games in order to get statistically relevant paths that will allow us to make meaningful comparison to the human players paths.

### Task 3: Analyse crowd performance vs average performance

Our idea is to start a game at a certain page `src` and target `dst` with all the data of the previous games. To choose the second page we click on, we aggregate all the paths that have source `src` and destination `dst`, including those that go through `src` but still target `dst`. This way, we have the next page each player went to for all these paths from `src`, we choose our next page as the one most players chose. We repeat this operation until we reach `dst` (we call this procedure the crowd algorithm). For Condorcet's jury theorem to apply, we need to maximise the number of voters at each step. To do this, we chose paths with maximum voter scores. We call the voter score of a path the minimum number of voters encountered by the crowd algorithm.

We run this algorithm on each (`src`, `dst`) tuple with a voter score > 50 and compare the results with the real players obtained on average for the same (`src`, `dst`) tuple.


## Proposed timeline

22.11.2023: Data handling, preprocessing <br>
29.11.2023: Implement tasks <br>
06.12.2023: Compile final data analysis <br>
13.12.2023: Finalize visualizations and start writing the data story <br>
20.12.2023: Clean the repository and finalize the data story webpage <br>

## Team organization

Yasmine Chaker: task X <br>
Hassen Aissa: task 3 <br>
Reza Machraoui: task 4 <br>
Matisse Van Schalkwijk: task X <br>
Lysandre Costes: tasks 1 and 3 <br>
