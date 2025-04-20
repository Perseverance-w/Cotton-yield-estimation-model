#install.packages("factoextra")
#install.packages("cluster")

library(factoextra)
library(cluster)

df <- read.csv("D:/inputfile.csv",header = T,row.names = 1)
#scale each variable to have a mean of 0 and sd of 1
df <- scale(df)

fviz_nbclust(df, kmeans, method = "wss")

gap_stat <- clusGap(df,
                    FUN = kmeans,
                    nstart = 25,
                    K.max = 10,
                    B = 50)

#plot number of clusters vs. gap statistic
fviz_gap_stat(gap_stat)

set.seed(1)

km <- kmeans(df, centers = 5, nstart = 25)

km

km$cluster
write.table(km$cluster, file = "group.txt",sep = "\t",row.names = T,col.names = NA,quote = F)

#plot results of final k-means model
fviz_cluster(km, data = df,
             palette = "jco",
             ggtheme = theme_minimal())