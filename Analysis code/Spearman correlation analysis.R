#install.packages("ggplot2")
#install.packages("ggpubr")
#install.packages("ggExtra")
#install.packages("shiny")

library(ggplot2)
library(ggpubr)
library(ggExtra)

setwd("D:/Correlation")

inputFile="LiDAR_LR.txt"      
gene1="manually_measure"             
gene2="UAV_LR"              

rt=read.table(inputFile,sep="\t",header=T,check.names=F,row.names=1)
x=as.numeric(rt[gene1,])
y=as.numeric(rt[gene2,])

df1=as.data.frame(cbind(x,y))
corT=cor.test(x,y,method="spearman")
cor=corT$estimate
pValue=corT$p.value
p1=ggplot(df1, aes(x, y)) + 
   xlab(gene1)+ylab(gene2)+
   geom_point()+ geom_smooth(method="lm",formula = y ~ x) + theme_bw()+
   stat_cor(method = 'spearman', aes(x =x, y =y))
p1

p2=ggMarginal(p1, type = "density", xparams = list(fill = "orange"),yparams = list(fill = "blue"))
p2