# What can wikispeedia teach us about making decisions in life?

## Abstract

Our project explores principles for thoughtful choices, using insights from the Wikispeedia dataset. 
First, it's crucial not to rely solely on large language models (LLMs) for guidance, as they provide data on topics but not necessarily the best course of action. Consulting others (leveraging crowd knowledge) can also enrich decision-making, though it’s important to understand that crowds don’t always succeed; examining specific paths reveals how groupthink and misjudgments can lead to failure. Recognizing that everything is interconnected—similar to the "small world" phenomenon—highlights that even seemingly minor information can be significant. Analyzing failed attempts to reach a target page in Wikispeedia, we find that some users may not have examined the full content of each page or made a thorough effort. This may reflect a tendency to stick to surface-level information, suggesting that better decisions often come from fully exploring available information rather than stopping at the first glance.

When we talk about performance here, we consider the shortness of the paths as the criterion.
Also, It's important to clarify that extrapolating the advice found on Wikispeedia to decision-making in life in general is an exaggeration and should be taken with a grain of salt.


## Research questions
 - Does people lazyness affect their performance ? Is there a noticeable difference in the page position of clicked links between successful and unsuccessful players? Do people click more on subjects they already know more about?
 - Will a crowd of players do better than the same players indicidually ? In other words, does Condorcet's jury theorem apply ?  If it applies, what are the recurring patterns for the cases in which the crowd fails to beat the individuals ?
 - 
 - 
 - 

## Additional data
 - **Wikipedia article traffic :**
 To represent the popularity of the concepts for task 1, we will use the number of views of the corresponding Wikipedia pages. 
 To obtain this data, we scrapped the [wikishark](https://www.wikishark.com/) website. The data will span from 2008 to 2014, allowing for examinations over the Wikispeedia dataset timeframe.


## Project plans & Methods

### Task 1: Analyse page position of clicked links
We obtain the x and y coordinates of the links by using Selenium. It enables us to open the html file of the articles in a simulated browser and then select the links we are interested in. We set the window size of the browser to 1920x1080 because it is the most common. When the next articlein path is accessible via several links, we select the coordinates of all these links.
We run this algorithm on each path of the 50 most popular source-target peers and compare the results between the optimal paths, those of people who didn't finish the game and those of people who did.


### Task 2: Analyse the popularity of concepts chosen
We use the wikipedia article traffic data (cf. Additional data) to modelize the popularity of the articles. Then we look at the frequency with which concepts are chosen and their popularity. 
We can compare these values with the players' performances to discover interesting patterns showing the impact of exploring new concepts.

### Task 3:

### Task 4:

### Task 5:

### Task 6:

### Task 7:


## Proposed timeline

22.11.2023: Data handling, preprocessing <br>
29.11.2023: Implement tasks <br>
06.12.2023: Compile final data analysis <br>
13.12.2023: Finalize visualizations and start writing the data story <br>
20.12.2023: Clean the repository and finalize the data story webpage <br>

## Team organization

Yasmine Chaker: task X <br>
Hassen Aissa: task X <br>
Reza Machraoui: task X <br>
Matisse Van Schalkwijk: task X <br>
Lysandre Costes: tasks 1 and 2 <br>