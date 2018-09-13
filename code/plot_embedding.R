library(Ckmeans.1d.dp)
library(ggplot)
library(ggrepel)

# Plots the full embedding as well the two top-level clusters in the first PC

# The absolute path to and base filename of the .data, .sounds, and .contexts files to read.
DATA_ROOT <- "/your/path/here/vector_data/parupa_trigram_ppmi"
PLOT_DIR <- "/your/path/here/plot_data/"

# Plot dimensions in pixels
WIDTH = 1055
HEIGHT = 670
  
# Read files
values <- read.csv(paste(DATA_ROOT,".data",sep=""), sep=" ", header=FALSE)
sounds <- read.csv(paste(DATA_ROOT,".sounds",sep=""), sep=" ", header=FALSE, 
                   stringsAsFactors=FALSE, colClasses = c("character"))
contexts <- read.csv(paste(DATA_ROOT,".contexts",sep=""), sep=" ", header=FALSE)
rownames(values) <- t(sounds)
colnames(values) <- t(contexts)

# Plot the PCA of the entire embedding
png(paste(PLOT_DIR, basename(DATA_ROOT), '_full.png', sep=''), width=WIDTH, height=HEIGHT)
pca <- prcomp(values, center=TRUE)
plot_data <- data.frame(pca$x[,1:2])
plot_data$sound <- rownames(plot_data)
ggplot(plot_data, aes(PC1, PC2, label=sound)) +
  geom_point() +
  geom_label_repel(size=10, force=6) +
  theme(axis.title=element_text(size=20), axis.text = element_text(size=14))
dev.off()

# Get the two top-level clusters.
clusters <- Ckmeans.1d.dp(pca$x[,1], 2)

# Plot the PCA of the first cluster
png(paste(PLOT_DIR, basename(DATA_ROOT), '_pc1.png', sep=''), width=WIDTH, height=HEIGHT)
pca1 <- prcomp(values[clusters$cluster == 1,], center=TRUE)
plot_data <- data.frame(pca1$x[,1:2])
plot_data$sound <- rownames(plot_data)
ggplot(plot_data, aes(PC1, PC2, label=sound)) +
  geom_point() +
  geom_label_repel(size=10, force=6) +
  theme(axis.title=element_text(size=20), axis.text = element_text(size=14))
dev.off()

# Plot the PCA of the second cluster
png(paste(PLOT_DIR, basename(DATA_ROOT), '_pc2.png', sep=''), width=WIDTH, height=HEIGHT)
pca2 <- prcomp(values[clusters$cluster == 2,], center=TRUE)
plot_data <- data.frame(pca2$x[,1:2])
plot_data$sound <- rownames(plot_data)
ggplot(plot_data, aes(PC1, PC2, label=sound)) +
  geom_point() +
  geom_label_repel(size=10, force=6) +
  theme(axis.title=element_text(size=20), axis.text = element_text(size=14))
dev.off()