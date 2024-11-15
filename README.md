# What can Wikispeedia teach us about making decisions in life?

! [](data/tadaa_meme.jpg)


## Abstract

<!-- 164 words -->

In Wikispeedia, players move from a source to a target article with minimal clicks—similar to how we pursue life goals, seeking the most efficient path. Our project draws three key lessons from these games for making better decisions.

First, we observe that players who skimmed through pages often missed critical information, impacting their success. This reflects a tendency to rely on surface-level data, highlithing that thorough research and examining all available information often lead to better outcomes.

Second, while LLMs can aid decision-making, over-relying on them is risky; they don’t always offer the best or most consistent advice.

Lastly, consulting others can enrich our choices by leveraging crowd knowledge. However, it’s important to recognize that crowds don’t always succeed. Groupthink can distort judgment, so it’s crucial to remain critical and deliberate rather than blindly following popular opinion.

These insights, while inspired by a game, underline the value of thoughtful, well-informed decisions in real life—where complexities go beyond simply choosing the next link.

NB: When we talk about performance in this project, we consider the shortness of the paths as the criterion.


## Research questions

- Does people lazyness affect their performance? Is there a noticeable difference in the page position of clicked links between successful and unsuccessful players?
- Can we always trust LLMs? Is it safe to take their answers as facts without second checking?
- Will a crowd of players do better than the same players individually? In other words, does Condorcet's jury theorem apply? If it applies, what are the recurring patterns for the cases in which the crowd fails to beat the individuals?

## Additional data

- **Generated Paths by Qwen-3B**
  We generated paths for the 10 most played source-target pairs in order to compare LLM performance to human players performance. We discuss the generation strategy in the following section.

## Project plans & Methods

### Task 1: Analyse page position of clicked links

We obtain the x and y coordinates of the links by using Selenium. It enables us to open the html file of the articles in a simulated browser and then select the links we are interested in. We set the window size of the browser to 1920x1080 because it is the most common. When the next article in path is accessible via several links, we select the coordinates of all these links.
We run this algorithm on each path of the 50 most popular source-target peers and compare the results between the optimal paths, those of people who didn't finish the game and those of people who did.

### Task 2: Analyse LLM performance in Wikispeedia

We use Qwen3b-4-bit-quantized as it fits in our GPU resources and it gives good results. We start by doing prompt engineering in order to get the model to understand the task.
If the word returned is not in the list we keep the conversation context with the model and send this new request asking it to correct itself. <br>
By this, we make the model rethink its answer and give it another chance to correct its choice. If the model persists in choosing a word not in the list, we consider the path failed. Otherwise, we recompute the new list to choose from by getting the links in the page that the model chose. In order to avoid looping in a circular fashion indefinitely, we remove from this list all the articles that the model already visited. If the model reaches the target word, then we consider the path a success and we stop the algorithm. Otherwise, we reset the model context in order to make the prompt short enough to fit into our GPU resources and we redo the same steps.
We only allow the model to run for 50 steps after which we consider the path as a failure.<br>

For each source-target pair, we make the model play 100 games in order to get statistically relevant paths that will allow us to make meaningful comparison to the human players paths.

### Task 3: Analyse crowd performance vs average performance

Our idea is to start a game with a given source `src` and target `dst` that might not have been played before. We then exploit all the data of the previous games. To choose the second page to click on, we aggregate all the paths that either have source `src` and destination `dst`, or those that go through `src` and have target `dst`. This way, we have the next page each player chose after `src`. We select our next page using majority voting. We repeat this operation until we reach `dst` (we call this procedure the crowd algorithm). For Condorcet's jury theorem to apply, we need to maximise the number of voters at each step. To do this, we chose paths that maximize 'voter scores'. We call the voter score of a path the minimum number of voters encountered by the crowd algorithm. (ie: at each step of the path, we guarantee a certain number of voters)

We run this algorithm on each (`src`, `dst`) tuple with a voter score > 50 and compare the results with what the real players obtained on average for the same (`src`,`dst`) tuple.

## Other explorations that didn't yield

We studied in depth the change of the paths specificities (Hubs, most visited articles...) over the time. We wanted to see if big events (ex: World Cup) affected the way players thought and found out it was not the case. We also studied what the characteristics of the graph are in order to cluster it and find the categories that would constitute a "joker" shortcut from any article to another. These ideas did not give fruitful results.

## Proposed timeline

22.11.2023: Data handling, preprocessing <br>
29.11.2023: Implement tasks <br>
06.12.2023: Compile final data analysis <br>
13.12.2023: Finalize visualizations and start writing the data story <br>
20.12.2023: Clean the repository and finalize the data story webpage <br>

## Team organization 

Task 1: Lysandre Costes <br>
Task 2: Hassen Aissa - Yasmine Chaker <br>
Task 3: Réza Machraoui - Matisse Vanschalkwijk <br>
