# What can Wikispeedia teach us about making decisions in life?
<div align="center">
  <img src="data/tadaa_meme.jpg">
</div>

NB: You need to clone the project and access the results.ipynb notebook in order to display the plots as they are made with plotly.

🌐🧞‍♂️ Link to data story: [click here](https://how-to-make-decisions-according-to-wikispeedia.b12sites.com/index)

## Abstract

<!-- 164 words -->

In Wikispeedia, players move from a source to a target article with minimal clicks—similar to how we pursue life goals, seeking the most efficient path. Our project draws three key lessons from these games for making better decisions.

First, we observe that players who skimmed through pages often missed critical information, impacting their success. This reflects a tendency to rely on surface-level data, highlithing that thorough research and examining all available information often lead to better outcomes.

Second, while LLMs can aid decision-making, over-relying on them is risky; they don’t always offer the best or most consistent advice.

Lastly, seeking advice and gathering as much information as possible from others can provide valuable insights. However, in the end, it's crucial to make your own decision, staying critical and thoughtful instead of blindly following others' opinions.

These insights, while inspired by a game, underline the value of thoughtful, well-informed decisions in real life—where complexities go beyond simply choosing the next link.

NB: When we talk about performance in this project, we consider the shortness of the paths as the criterion.


## Research questions

- Does people lazyness affect their performance? Is there a noticeable difference in the page position of clicked links between successful and unsuccessful players?
- Can we always trust LLMs? Is it safe to take their answers as facts without second checking?
- Will a crowd of players do better than the same players individually? In other words, does Condorcet's jury theorem apply? If it applies, what are the recurring patterns for the cases in which the crowd fails to beat the individuals?

## Additional data

- **Generated Paths by Qwen-3B and LLama-3B**
  We generated paths for the 10 most played source-target pairs in order to compare LLM performance to human players performance. We discuss the generation strategies in the following section.


## Project Plans & Methods

### Task-1: Analyse page position of clicked links

We obtain the x and y coordinates of the links by using Selenium. It enables us to open the html file of the articles in a simulated browser and then select the links we are interested in. We set the window size of the browser to 1920x1080 because it is the most common. When the next article in path is accessible via several links, we select the coordinates of all these links.
We run this algorithm on each path of the 50 most popular source-target pairs and compare the results between the optimal paths, those of people who didn't finish the game and those of people who did. To take things a step further, we analyzed how exploration changes over time by plotting the variance in horizontal and vertical link positions as a function of the player's path position. Higher variance indicates more thorough exploration, with links selected from diverse areas.

### Task-2: Analyse LLM performance in Wikispeedia

We examine the top 10 most frequent source-target pairs in Wikispeedia's finished paths dataset. Two small LLMs, Qwen-3B and Llama-3B (4-bit quantized), were tested using two prompting strategies: a "simple prompt" (one-sentence instruction) and a "detailed prompt" (step-by-step guidance). Each model performed 100 runs with the simple prompt and 30 runs with the detailed prompt due to time constraints. Prompt engineering was applied to refine instructions, ensuring models chose the best links to reach the target. If the model's response was incorrect, the path was either corrected or aborted.

We compare the performance of different models and prompting strategies, identifying Qwen with a simple prompt as the best-performing model. However, when compared to human performance, humans consistently outperform LLMs. To understand this disparity, we analyze the strategies used by human players and LLMs. Our findings reveal that humans tend to navigate by first reaching a central hub before proceeding to the target. When we prompt Qwen to adopt a similar strategy, its performance remains suboptimal, indicating limitations in its ability to emulate human-like problem-solving.

### Task-3: Analyse crowd performance vs average performance

Our idea is to start a game with a given source `src` and target `dst` that might not have been played before. We then exploit all the data of the previous games. To choose the second page to click on, we aggregate all the paths that either have source `src` and destination `dst`, or those that go through `src` and have target `dst`. This way, we have the next page each player chose after `src`. We select our next page using majority voting. We repeat this operation until we reach `dst` (we call this procedure the crowd algorithm). For Condorcet's jury theorem to apply, we need to maximise the number of voters at each step. To do this, we chose paths that maximize 'voter scores'. We call the voter score of a path the minimum number of voters encountered by the crowd algorithm. (ie: at each step of the path, we guarantee a certain number of voters)

We run this algorithm on each (`src`, `dst`) tuple with a voter score > 50 and compare the results with what the real players obtained on average for the same (`src`,`dst`) tuple.

We find that in 98% of the games played, the crowd outperforms the average of individual people.


## Contributions 

Hassen Aissa: Prompt engineering and plots of LLMs part, top 10 pairs analysis, data story
Yasmine Chaker: Generating LLM paths, plots of LLMs part, data story
Lysandre Costes: Algorithms and plots of page position of clicked links part, top 50 pairs analysis, data story
Réza Machraoui: Algorithms and plots of crowd performance part, data story
Matisse Vanschalkwijk: Website design, data story


