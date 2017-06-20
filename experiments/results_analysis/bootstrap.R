raw_data = head(read.csv('~/Desktop/CP.csv', stringsAsFactors=FALSE),-2)

raw_matches = as.numeric(raw_data$X.matches)
raw_gt = as.numeric(raw_data$X.cells.in.GT)
raw_model = as.numeric(raw_data$X.model)

counts = data.frame(raw_matches, raw_gt, raw_model)
colnames(counts) <- c("matches", "gt", "model")

B <- 10000
n <- 50

ps <- numeric(length = B)
rs <- numeric(length = B)
f1s <- numeric(length = B)

for (b in 1:B) {
  # select random indices
  i <- sample(1:n, size = 20, replace = TRUE)
  
  matches <- sum(counts$matches[i])
  nuclei_in_gt <- sum(counts$gt[i])
  nuclei_in_model <- sum(counts$model[i])
  
  # calculate p, r and f1
  p = matches / nuclei_in_model
  r = matches / nuclei_in_gt
  f1 = 2 * (p * r) / (p + r)
  
  
  ps[b] <- p
  rs[b] <- r
  f1s[b] <- f1
  
}

theme_set(theme_gray(base_size = 14))
qplot(
  f1s,
  geom = "histogram",
  binwidth = 0.005,
  main="Distribution of F1 Score",
  fill=I("gray"),
  col=I("black"), 
  alpha=I(.2),
  ylab="Count",
  xlab="F1 Score"
)
mean(f1s)
sd(f1s)
