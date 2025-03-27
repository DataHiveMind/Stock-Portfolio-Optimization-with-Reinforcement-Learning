# src/analyze_results.R
library(ggplot2)

results <- read.csv("backtest_results.csv")

# Plot the equity curve
ggplot(results, aes(x = X, y = Balance)) +
  geom_line() +
  labs(title = "Equity Curve", x = "Time", y = "Balance") +
  theme_minimal()

# Save the plot
ggsave("equity_curve.png")