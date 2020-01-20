t1 <- Sys.time()
cat("\014") 
options(max.print=1000000)
#https://kateto.net/networks-r-igraph
#https://stackoverflow.com/questions/15999877/correctly-color-vertices-in-r-igraph

library(ggplot2)
library(ggthemes)
library(extrafont)
library(plyr)
library(scales)
library(igraph)


# practice 
# full_graph <- graph( edges=c(1,2, 1,3, 1,4), n=3, directed=T ) 
# plot(full_graph, edge.arrow.size=.5, vertex.color="gold", vertex.size=15, vertex.frame.color="gray", vertex.label.color="black", 
#      vertex.label.cex=0.8, vertex.label.dist=3, edge.curved=0.2) 

# project#1 
# nodes_ <- read.csv("/Users/akond/Documents/AkondOneDrive/OneDrive/JobPrep-TNTU2019/grant-writing/nsf-small/prelim-evidence/Grant-Project1-NODES.csv", header=T, as.is=T)
# edges_ <- read.csv("/Users/akond/Documents/AkondOneDrive/OneDrive/JobPrep-TNTU2019/grant-writing/nsf-small/prelim-evidence/Grant-Project1-EDGES.csv", header=T, as.is=T)
# the_graph <- graph_from_data_frame(d=edges_, vertices=nodes_, directed=T) 
# colrs <- c("gold", "red")
# V(the_graph)$color <-  ifelse(V(the_graph)$type == 1, "gold", "red")
# 
# plot(the_graph, edge.arrow.size=.5,  vertex.size=10, vertex.frame.color="gray", vertex.label=NA,
#      edge.curved=0.1, edge.lty=1, ylim=c(0, 0.5), xlim=c(0, 0.5), edge.arrow.mode=0 )




t2 <- Sys.time()
print(t2 - t1)  
rm(list = setdiff(ls(), lsf.str()))