library(DescTools)

# Deterministic
# 5 x 5
df_det_5_5 <- rbind(read.csv('/Users/mymac/alegaater_pred_prey/tests/results/deterministic_greedy_prob_5_5.csv'), 
                        read.csv('/Users/mymac/alegaater_pred_prey/tests/results/deterministic_prob_dest_5_5.csv'))

model <- lm(df_det_5_5$Rewards ~ df_det_5_5$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "df_det_5_5$Agent", conf.level=0.95)
print(tukey)

# 10 x 10
df_det_10_10 <- rbind(read.csv('/Users/mymac/alegaater_pred_prey/tests/results/deterministic_greedy_prob_10_10.csv'), 
                          read.csv('/Users/mymac/alegaater_pred_prey/tests/results/deterministic_prob_dest_10_10.csv'))

model <- lm(df_det_10_10$Rewards ~ df_det_10_10$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "df_det_10_10$Agent", conf.level=0.95)
print(tukey)

# 15 x 15
df_det_15_15 <- rbind(read.csv('/Users/mymac/alegaater_pred_prey/tests/results/deterministic_greedy_prob_15_15.csv'), 
                          read.csv('/Users/mymac/alegaater_pred_prey/tests/results/deterministic_prob_dest_15_15.csv'))

model <- lm(df_det_15_15$Rewards ~ df_det_15_15$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "df_det_15_15$Agent", conf.level=0.95)
print(tukey)

# Combined
df_det_5_5$Rewards = (df_det_5_5$Rewards - min(df_det_5_5$Rewards)) / (max(df_det_5_5$Rewards) - min(df_det_5_5$Rewards))
df_det_10_10$Rewards = (df_det_10_10$Rewards - min(df_det_10_10$Rewards)) / (max(df_det_10_10$Rewards) - min(df_det_10_10$Rewards))
df_det_15_15$Rewards = (df_det_15_15$Rewards - min(df_det_15_15$Rewards)) / (max(df_det_15_15$Rewards) - min(df_det_15_15$Rewards))

final_df_det <- rbind(df_det_5_5, df_det_10_10, df_det_15_15)

model <- lm(final_df_det$Rewards ~ final_df_det$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "final_df_det$Agent", conf.level=0.95)
print(tukey)




# Non-deterministic
# 5 x 5
df_non_det_5_5 <- rbind(read.csv('/Users/mymac/alegaater_pred_prey/tests/results/nondeterministic_greedy_prob_5_5.csv'), 
                        read.csv('/Users/mymac/alegaater_pred_prey/tests/results/nondeterministic_prob_dest_5_5.csv'))

model <- lm(df_non_det_5_5$Rewards ~ df_non_det_5_5$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "df_non_det_5_5$Agent", conf.level=0.95)
print(tukey)

# 10 x 10
df_non_det_10_10 <- rbind(read.csv('/Users/mymac/alegaater_pred_prey/tests/results/nondeterministic_greedy_prob_10_10.csv'), 
                          read.csv('/Users/mymac/alegaater_pred_prey/tests/results/nondeterministic_prob_dest_10_10.csv'))

model <- lm(df_non_det_10_10$Rewards ~ df_non_det_10_10$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "df_non_det_10_10$Agent", conf.level=0.95)
print(tukey)

# 15 x 15
df_non_det_15_15 <- rbind(read.csv('/Users/mymac/alegaater_pred_prey/tests/results/nondeterministic_greedy_prob_15_15.csv'), 
                          read.csv('/Users/mymac/alegaater_pred_prey/tests/results/nondeterministic_prob_dest_15_15.csv'))

model <- lm(df_non_det_15_15$Rewards ~ df_non_det_15_15$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "df_non_det_15_15$Agent", conf.level=0.95)
print(tukey)

# Combined
df_non_det_5_5$Rewards = (df_non_det_5_5$Rewards - min(df_non_det_5_5$Rewards)) / (max(df_non_det_5_5$Rewards) - min(df_non_det_5_5$Rewards))
df_non_det_10_10$Rewards = (df_non_det_10_10$Rewards - min(df_non_det_10_10$Rewards)) / (max(df_non_det_10_10$Rewards) - min(df_non_det_10_10$Rewards))
df_non_det_15_15$Rewards = (df_non_det_15_15$Rewards - min(df_non_det_15_15$Rewards)) / (max(df_non_det_15_15$Rewards) - min(df_non_det_15_15$Rewards))

final_df_non_det <- rbind(df_non_det_5_5, df_non_det_10_10, df_non_det_15_15)

model <- lm(final_df_non_det$Rewards ~ final_df_non_det$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "final_df_non_det$Agent", conf.level=0.95)
print(tukey)


# Mixed
# 5 x 5
df_mix_5_5 <- read.csv('/Users/mymac/alegaater_pred_prey/tests/results/mixed_5_5.csv')

model <- lm(df_mix_5_5$Rewards ~ df_mix_5_5$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "df_mix_5_5$Agent", conf.level=0.95)
print(tukey)

# 10 x 10
df_mix_10_10 <- read.csv('/Users/mymac/alegaater_pred_prey/tests/results/mixed_10_10.csv')

model <- lm(df_mix_10_10$Rewards ~ df_mix_10_10$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "df_mix_10_10$Agent", conf.level=0.95)
print(tukey)

# 15 x 15
df_mix_15_15 <- read.csv('/Users/mymac/alegaater_pred_prey/tests/results/mixed_15_15.csv')

model <- lm(df_mix_15_15$Rewards ~ df_mix_15_15$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "df_mix_15_15$Agent", conf.level=0.95)
print(tukey)

# Combined
df_mix_5_5$Rewards = (df_mix_5_5$Rewards - min(df_mix_5_5$Rewards)) / (max(df_mix_5_5$Rewards) - min(df_mix_5_5$Rewards))
df_mix_10_10$Rewards = (df_mix_10_10$Rewards - min(df_mix_10_10$Rewards)) / (max(df_mix_10_10$Rewards) - min(df_mix_10_10$Rewards))
df_mix_15_15$Rewards = (df_mix_15_15$Rewards - min(df_mix_15_15$Rewards)) / (max(df_mix_15_15$Rewards) - min(df_mix_15_15$Rewards))

final_df_mix <- rbind(df_mix_5_5, df_mix_10_10, df_mix_15_15)

model <- lm(final_df_mix$Rewards ~ final_df_mix$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "final_df_mix$Agent", conf.level=0.95)
print(tukey)




# Combined across all categories
final_df <- rbind(final_df_det, final_df_non_det, final_df_mix)

model <- lm(final_df$Rewards ~ final_df$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "final_df$Agent", conf.level=0.95)
print(tukey)


