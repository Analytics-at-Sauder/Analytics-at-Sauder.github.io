## Critical Path Analysis

#### Author: Charlie Cao

In this Notebook, we are going to explore ways we can use Python to look for and visualize critical paths in projects. We encourage you to create your own Jupytor Notebook and follow along. You can also download this Notebook along with any accompanying data in the [Notebooks and Data](https://github.com/Master-of-Business-Analytics/Notebooks_and_Data) GitHub repository. Alternatively, if you do not have Python or Jupyter Notebook installed yet, you may experiment with a virtual Notebook by launching Binder or Syzygy below (learn more about these two tools in the [Resource](https://analytics-at-sauder.github.io/resource.html) tab). 

<a href="https://ubc.syzygy.ca/jupyter/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAnalytics-at-Sauder%2FNB0013_Critical_Path_Analysis&urlpath=tree%2FNB0013_Critical_Path_Analysis%2Fnb0013_critical_path_analysis.ipynb&branch=master" target="_blank" class="button">Launch Syzygy (UBC)</a>

<a href="https://pims.syzygy.ca/jupyter/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAnalytics-at-Sauder%2FNB0013_Critical_Path_Analysis&urlpath=tree%2FNB0013_Critical_Path_Analysis%2Fnb0013_critical_path_analysis.ipynb&branch=master" target="_blank" class="button">Launch Syzygy (Google)</a>

<a href="https://mybinder.org/v2/gh/Analytics-at-Sauder/NB0013_Critical_Path_Analysis/master?filepath=nb0013_critical_path_analysis.ipynb" target="_blank" class="button">Launch Binder</a>

## Background

---

A critical path is essentially the bottleneck of a project. Typically, in Project Management, the various tasks and dependicies would first be analyzed and identified using Work Breakdown Structure (WBS) and Program Evaluation and Review Technique (PERT) before they are pipelined into Critical Path Analysis (CPA). In order to reduce the amount of time that a project takes, a manager should seek to shorten tasks along the critical path. One way of finding the critical path is by identifying the path with zero slack (or float). Slack is a buffer where a delay in task completion may not affect the timeline of the entire project, and it is calculated by subtracting the earliest finishing time (EF) from the latest finishing time (LF), or equivalently, subtracting the earliest start time (ES) from the latest start time (LS). You can read more about CPA in this [HBR article](https://hbr.org/1963/09/the-abcs-of-the-critical-path-method)

Let's consider the following example:

- Task A requires 3 days, and it can be started at any time.
- Task B requires 5 days, and it can also be started at any time.
- Task C requires 2 days, and it can only start after *both* Task A and Task B are finished.
- Task D requires 3 days, and it can only start after Task A is finished.
- Task E requires 5 days, and it can only start after *both* Task C and Task D are finished.

![Critical Path](nb0013_images/p13_01.png)

By calculating the slack of each task, we can quickly identify that the critical path is *B -> C -> E*, since there is zero slack along the path connecting these three tasks. While the critical path was fairly simple to identify here, this analysis becomes much more complicated when managers are presented with more tasks for each project. Furthermore, when certain tasks are compressed, the critical path might change. It would be incredibly mechanical and tedious to repeat this process to identify the critical path for every single project we will ever face; therefore, in this Project, we will experiment with some packages that can be used to help Project Managers visualize the dependencies of different projects, identify the critical path, and understand how the critical path changes as completion times of different tasks are reduced.

## Finding and Visualizing Critical Path in Python

---

In this Notebook, we are going to use two packages, `networkx` (dependent on `matplotlib`) and `criticalpath` (install using [pip](https://docs.python.org/3/installing/index.html) if you do not have these packages yet). We encourage you to read the documentations for [networkx](https://networkx.github.io/documentation/networkx-2.4/), a useful package for visualizing networks, as well as [criticalpath](https://pypi.org/project/criticalpath/), a simple developer package specifically designed to look for the critical path in a series of processes. First, let's import the packages:


```python
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import networkx as nx
from criticalpath import Node
import plotly.express as px
from IPython.display import Image
```

### Set Up the Tasks and Visualize the Dependencies

Before we start looking for the critical path, we have to set up a task list, as well as the sequence and dependencies of the tasks. We do so by creating two lists of tuples. In the task list, each tuple contains the name of the task and any attribute(s) of the task; the attribute(s) should be presented in a dictionary. In the dependency list, each tuple contains two tasks, with the second one dependent on the first. After the task and dependency lists are setup, we can initialize a directed graph (with arrows) and import the two lists:


```python
# set up the tasks:
tasks = [("A", {"Duration": 3}), 
         ("B", {"Duration": 5}), 
         ("C", {"Duration": 2}), 
         ("D", {"Duration": 3}), 
         ("E", {"Duration": 5})]

# set up the dependencies along all paths:
dependencies = [("A", "C"), 
                ("B", "C"), 
                ("A", "D"),
                ("C", "E"), 
                ("D", "E")]

# initialize (directed) graph
G = nx.DiGraph() 

# add tasks and dependencies (edges)
G.add_nodes_from(tasks)
G.add_edges_from(dependencies)
```

One thing to notice is that, by default, `networkx` randomly positions the nodes every time it generates a plot. This is technically not a problem, as long as the ordering of the tasks remain the same; however, we usually want their graphs to be predictable and to understand. To do so, it is a good idea to specify the (x, y) coordinates of the nodes. It may be helpful for you to draw the plot on a piece of paper first.


```python
# set up the (arbitrary) positions of the tasks (nodes):
pos_nodes = {"A": (1, 3), 
             "B": (1, 1), 
             "C": (2, 2), 
             "D": (3, 3), 
             "E": (4, 2)}

# draw the nodes
nx.draw(G, with_labels=True, pos=pos_nodes, node_color='lightblue', arrowsize=20)


# set up the (arbitrary) positions of the durations labels (attributes):
pos_attrs = {node:(coord[0], coord[1] + 0.2) for node, coord in pos_nodes.items()}
attrs = nx.get_node_attributes(G, 'Duration')

# draw (write) the node attributes (duration)
nx.draw_networkx_labels(G, pos=pos_attrs, labels=attrs)


# set a little margin (padding) for the graph so the labels are not cut off
plt.margins(0.1)
```


![png](nb0013_critical_path_analysis_files/nb0013_critical_path_analysis_5_0.png)


### Searching for the Critical Path

Now that we have the network of tasks visualized (which will be helpful later when we want to draw the critical path), we can proceed to actually looking for the critical path. This is where the `criticalpath` package comes in handy. Intead of manually calculating the possible slack of each task, we can just load the tasks, their durations, and any dependencies into an `object` (project) using the `criticalpath` package:


```python
# initialize a "project":
proj = Node('Project')

# load the tasks and their durations:
for t in tasks:
    proj.add(Node(t[0], duration=t[1]["Duration"]))

# load the dependencies (or sequence):
for d in dependencies:
    proj.link(d[0],d[1])

# update the "project":
proj.update_all()
```

Once we update the "project" with tasks and dependencies using the `update_all()` method, we can extract the critical path and the total project duration using the following commands:


```python
# proj.get_critical_path() will return a list of nodes
# however, we want to store them as strings so that they can be easily used for visualization later
crit_path = [str(n) for n in proj.get_critical_path()]

# get the current duration of the project
proj_duration = proj.duration

print(f"The current critical path is: {crit_path}")
print(">"*50)
print(f"The current project duration is: {proj_duration} days")
```

    The current critical path is: ['B', 'C', 'E']
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    The current project duration is: 12 days
    

### Visualize the Critical Path

A picture is worth a thousand words, and it would be much faster for people to understand which tasks can be compressed, and the impact of reducing these tasks, through a visualized network. Therefore, we bring back the graph from above, but now, we layer our critical path on top of the plot we had earlier:


```python
# create a list of edges using the current critical path list:
crit_edges = [(n, crit_path[i+1]) for i, n in enumerate(crit_path[:-1])]

# first, recreate the network visualization:
nx.draw(G, with_labels=True, pos=pos_nodes, node_color='lightblue', arrowsize=20)
nx.draw_networkx_labels(G, pos=pos_attrs, labels=attrs)

# now add the critical path as an additional layer on top of the original graph:
nx.draw_networkx_edges(G, pos=pos_nodes, edgelist=crit_edges, width=10, alpha=0.5, edge_color='r')

# again, leaving some margin so the labels are not cut off
plt.margins(0.1)
```


![png](nb0013_critical_path_analysis_files/nb0013_critical_path_analysis_11_0.png)


Looking for the critical path can easily become a hassle in big projects with ever changing timelines of various tasks. The `networkx` and `criticalpath` packages allow us to find and visualize the critical path - the path that can use the most improvement - much more quickly and easily. Now that we have covered the basics, we have created a challenge for you below so that you can apply these techniques to a more complicated problem by yourself. Do not forget to go back and read the documentations of the different packages, if you want to explore different ways of solving this problem!

#### Gantt Charts

Another way of visualizing the critical path is by plotting out the project's Gantt Chart, which help to visualize the timeline of the tasks, including their start time, their duration, their end times, and their dependencies:


```python
proj_startdate = date.today()

proj_schedule = pd.DataFrame([dict(Task = key, 
                                   Start = datetime.date.today(), 
                                   Finish = datetime.date.today() + datetime.timedelta(val['Duration']), 
                                   Status = 'NA')
                              for key, val in dict(tasks).items()])

for key, val in dict(tasks).items():
    dep = [d for d in dependencies if d[1] == key]
    prev_tasks = [t[0] for t in dep]
    if prev_tasks:
        prev_finish = proj_schedule[proj_schedule.Task.isin(prev_tasks)]['Finish'].max()
        proj_schedule.loc[proj_schedule.Task == key, 'Start'] = prev_finish
        proj_schedule.loc[proj_schedule.Task == key, 'Finish'] = prev_finish + datetime.timedelta(val['Duration'])
        
proj_schedule.loc[proj_schedule.Task.isin(crit_path), 'Status'] = 'Critical Path'
        
display(proj_schedule)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Task</th>
      <th>Start</th>
      <th>Finish</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>2020-07-29</td>
      <td>2020-08-01</td>
      <td>NA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>2020-07-29</td>
      <td>2020-08-03</td>
      <td>Critical Path</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>2020-08-03</td>
      <td>2020-08-05</td>
      <td>Critical Path</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>2020-08-01</td>
      <td>2020-08-04</td>
      <td>NA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E</td>
      <td>2020-08-05</td>
      <td>2020-08-10</td>
      <td>Critical Path</td>
    </tr>
  </tbody>
</table>
</div>



```python
fig = px.timeline(proj_schedule, x_start="Start", x_end="Finish", y="Task", color="Status")
fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
Image(fig.to_image(format="png"))
```




![png](nb0013_critical_path_analysis_files/nb0013_critical_path_analysis_14_0.png)



## References

---

https://hbr.org/1963/09/the-abcs-of-the-critical-path-method <br>
https://medium.com/@yujeshmalekushrestha/the-critical-path-analysis-in-project-management-94c75f77932f
