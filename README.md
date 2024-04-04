# Visual_VRP

This project develops a reinforcement learning (RL) based policy aimed at optimizing solutions for complex supply chain mathematical models, specifically focusing on the Traveling Salesman Problem (TSP) and the Capacitated Vehicle Routing Problem (CVRP). Leveraging the power of Unity, the project also provides an interactive visualization platform. This visualization component enables users to intuitively understand the problem space, the RL policy's decision-making process, and the efficiency of the solutions generated. 

The Solver section of the code is revised upon from the literature "Solve Routing Problems with a Residual Edge-Graph Attention Neural Network".

## Dependencies
+ Python
+ Pytorch
+ torch_geometric
+ Numpy


## Repository Sturcture

The following gives a brief overview of the contents.

```
SupplyChainVideo                // Result of our code
SupplyChainPython               // Solver of TSP and VRP
VRP_project                     // Unity visualization of TSP and VRP
```

## Video

### Traveling Salesman Problem (Node = 20)

![Gif For TSP (Node = 20)](/SupplyChainVideo/TSP20.gif)

### Traveling Salesman Problem (Node = 50)

![Gif For TSP (Node = 20)](/SupplyChainVideo/TSP50.gif)

### Traveling Salesman Problem (Node = 100)

![Gif For TSP (Node = 20)](/SupplyChainVideo/TSP100.gif)

### Vehicle Routing Problem (Node = 20)

![Gif For TSP (Node = 20)](/SupplyChainVideo/VRP20.gif)

### Vehicle Routing Problem (Node = 50)

![Gif For TSP (Node = 20)](/SupplyChainVideo/VRP50.gif)
