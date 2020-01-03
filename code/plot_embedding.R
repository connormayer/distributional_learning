library(Ckmeans.1d.dp)
library(ggplot2)
library(ggrepel)

# Plots the full embedding as well the two top-level clusters in the first PC

# The absolute path to and base filename of the .data, .sounds, and .contexts files to read.
DATA_ROOT <- "/your/path/here/vector_data/parupa_trigram_ppmi"
PLOT_DIR <- "/your/path/here/plot_data/"

# Plot dimensions in pixels
WIDTH = 1055 * 4
HEIGHT = 670 * 5

axis_title_size = 200
label_size_main = 50
box_padding = 5
force = 2
segment_size = 5
label_size = 3
label_padding = 2

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
  geom_point(size=30) +
  geom_label_repel(size=label_size_main, force=force, segment.size=segment_size, label.size=label_size, label.padding = label_padding, box.padding=box_padding) +
  theme(axis.title=element_text(size=axis_title_size), axis.text = element_text(size=100))
dev.off()

# Get the two top-level clusters.
clusters <- Ckmeans.1d.dp(pca$x[,1], 2)

# Plot the PCA of the first cluster
png(paste(PLOT_DIR, basename(DATA_ROOT), '_pc1.png', sep=''), width=WIDTH, height=HEIGHT)
pca1 <- prcomp(values[clusters$cluster == 1,], center=TRUE)
plot_data <- data.frame(pca1$x[,1:2])
plot_data$sound <- rownames(plot_data)
ggplot(plot_data, aes(PC1, PC2, label=sound)) +
  geom_point(size=30) +
  geom_label_repel(size=label_size_main, force=force, segment.size=segment_size, label.size=label_size, label.padding = label_padding, box.padding=box_padding) +
  theme(axis.title=element_text(size=axis_title_size), axis.text = element_text(size=100))
dev.off()

# Plot the PCA of the second cluster
png(paste(PLOT_DIR, basename(DATA_ROOT), '_pc2.png', sep=''), width=WIDTH, height=HEIGHT)
pca2 <- prcomp(values[clusters$cluster == 2,], center=TRUE)
plot_data <- data.frame(pca2$x[,1:2])
plot_data$sound <- rownames(plot_data)
ggplot(plot_data, aes(PC1, PC2, label=sound)) +
  geom_point(size=30) +
  geom_label_repel(size=label_size_main, force=force, segment.size=segment_size, label.size=label_size, label.padding = label_padding, box.padding=box_padding) +
  theme(axis.title=element_text(size=axis_title_size), axis.text = element_text(size=100))
dev.off()