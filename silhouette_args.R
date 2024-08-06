#!/usr/bin/env Rscript
library(dplyr)
library(ggplot2)
library(tidyr)
library(emmeans)
library(lme4)

k0 <- 2
n_k <- 14
n_args <- 2
if (length(commandArgs(trailingOnly = TRUE)) != n_args) {
  cat("Usage: Rscript script.R <tipo_de_silaba> <nro_de_archivo>\n")
  quit(status = 1)
}
# Extract arguments
tipo <- commandArgs(trailingOnly = TRUE)[1]
counter <- commandArgs(trailingOnly = TRUE)[2]

tipo<-"beta"
counter<-24
tipo<-"machos"
counter<-8
#tipo <-"alfa"
#counter<-6
base_path <- paste0("/home/ttdduu/lsd/tesislab/entrenamientos/metricas/sils/",tipo,"/sils_R_agrego_EE")

# Dynamic filename for CSV using paste0
csv_filename <- paste0(base_path,"/datos_crudos_silhouette-",counter,".csv")

# Read CSV file without header
data <- read.csv(csv_filename,header=TRUE)
data <-t(t(data)) # odio R. sin esto, no me flattenea el df

df_long <- data.frame(

  k = rep(k0:n_k),  # Create repeated values for the independent variable
  value = as.vector(data[, -1])  # Flatten the dataframe into a vector
)
n <- nrow(df_long)
train <- rep(1:ceiling(n / 13), each = 13)[1:n]

# Add the train column to df_long
df_long$train <- train

# Print the updated dataframe
#print(df_long)

model <- lmer(as.numeric(value) ~ as.factor(k) + (1|train),data = df_long,) # glmer no tiene para modelar varz
model

library (lmerTest) #llamo a la biblioteca para realizar el modelo
ranova(model)

shapiro.test(ranef(model)$train$"(Intercept)")

#scatter <- ggplot(df_long, aes(x = k, y = value)) +
  #geom_point() +
  #stat_summary(geom = "errorbar", fun.data = mean_se)
  #labs(x = "Independent Variable", y = "Measurement", title = "Measurements vs Independent Variable")

"""

================ alfa
> model <- lmer(as.numeric(value) ~ as.factor(k) + (1|train),data = df_long,) # glmer no tiene para modelar varz
model

Linear mixed model fit by REML ['lmerModLmerTest']
Formula: as.numeric(value) ~ as.factor(k) + (1 | train)
   Data: df_long
REML criterion at convergence: -1375.485
Random effects:
 Groups   Name        Std.Dev.
 train    (Intercept) 0.01381
 Residual             0.03050
Number of obs: 364, groups:  train, 28
Fixed Effects:
   (Intercept)   as.factor(k)3   as.factor(k)4   as.factor(k)5   as.factor(k)6   as.factor(k)7   as.factor(k)8   as.factor(k)9  as.factor(k)10  as.factor(k)11
      0.612126       -0.009042        0.022470        0.077325        0.126737        0.162407        0.196427        0.163117        0.142581        0.126663
as.factor(k)12  as.factor(k)13  as.factor(k)14
      0.102627        0.088269        0.062870

> library (lmerTest) #llamo a la biblioteca para realizar el modelo
ranova(model)
ANOVA-like table for random-effects: Single term deletions

Model:
as.numeric(value) ~ as.factor(k) + (1 | train)
            npar logLik     AIC   LRT Df Pr(>Chisq)
<none>        15 687.74 -1345.5
(1 | train)   14 672.55 -1317.1 30.38  1  3.552e-08 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

> shapiro.test(ranef(model)$train$"(Intercept)")

        Shapiro-Wilk normality test

data:  ranef(model)$train$"(Intercept)"
W = 0.9737, p-value = 0.6823

============== beta
model <- lmer(as.numeric(value) ~ as.factor(k) + (1|train),data = df_long,) # glmer no tiene para modelar varz
Linear mixed model fit by REML ['lmerModLmerTest']
Formula: as.numeric(value) ~ as.factor(k) + (1 | train)
   Data: df_long
REML criterion at convergence: -2860.541
Random effects:
 Groups   Name        Std.Dev.
 train    (Intercept) 0.03838
 Residual             0.03231
Number of obs: 780, groups:  train, 60
Fixed Effects:
   (Intercept)   as.factor(k)3   as.factor(k)4   as.factor(k)5   as.factor(k)6   as.factor(k)7   as.factor(k)8   as.factor(k)9  as.factor(k)10  as.factor(k)11
      0.627446       -0.004535       -0.026670       -0.033371       -0.034327       -0.031911       -0.030074       -0.025679       -0.032087       -0.029334
as.factor(k)12  as.factor(k)13  as.factor(k)14
     -0.033841       -0.039047       -0.035775


> library (lmerTest) #llamo a la biblioteca para realizar el modelo
ranova(model)
ANOVA-like table for random-effects: Single term deletions

Model:
as.numeric(value) ~ as.factor(k) + (1 | train)
            npar logLik     AIC    LRT Df Pr(>Chisq)
<none>        15 1430.3 -2830.5
(1 | train)   14 1180.1 -2332.3 500.27  1  < 2.2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

> shapiro.test(ranef(model)$train$"(Intercept)")

        Shapiro-Wilk normality test

data:  ranef(model)$train$"(Intercept)"
W = 0.96943, p-value = 0.2761
============== macho
> model
Linear mixed model fit by REML ['lmerMod']
Formula: as.numeric(value) ~ as.factor(k) + (1 | train)
   Data: df_long
REML criterion at convergence: -1489.632
Random effects:
 Groups   Name        Std.Dev.
 train    (Intercept) 0.01895
 Residual             0.02728
Number of obs: 377, groups:  train, 29
Fixed Effects:
   (Intercept)   as.factor(k)3   as.factor(k)4   as.factor(k)5   as.factor(k)6   as.factor(k)7   as.factor(k)8   as.factor(k)9  as.factor(k)10  as.factor(k)11
       0.55948         0.02871         0.03575         0.05580         0.07777         0.08404         0.08234         0.07215         0.06939         0.06516
as.factor(k)12  as.factor(k)13  as.factor(k)14
       0.06013         0.05408         0.04337

> library (lmerTest) #llamo a la biblioteca para realizar el modelo
ranova(model)
ANOVA-like table for random-effects: Single term deletions

Model:
as.numeric(value) ~ as.factor(k) + (1 | train)
            npar logLik     AIC    LRT Df Pr(>Chisq)
<none>        15 744.82 -1459.6
(1 | train)   14 700.93 -1373.9 87.773  1  < 2.2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

        Shapiro-Wilk normality test

data:  ranef(model)$train$"(Intercept)"
W = 0.96681, p-value = 0.4767

"""

#model <- lm(as.numeric(value) ~ as.factor(df_long$k), data = df_long)

r1<-residuals(model,type="pearson") #guardamos los residuos de Pearson = estandarizados
pred1<-fitted(model)
library(car)
lev = leveneTest(model)
shap = shapiro.test(r1)


res_pred <- ggplot(df_long, aes(x=fitted(model), y=rstandard(model)))+geom_point(size=2)+geom_abline(slope=0, intercept=0) + geom_abline(slope=0, intercept=-2, color="red", linetype="dashed") + geom_abline(slope=0, intercept=2, color="red", linetype="dashed")  + ggtitle("Gráfico de RE vs predichos") +ylab("RE") # --> OK

qq <- ggplot(df_long, aes(sample=residuals(model)))+stat_qq() + stat_qq_line()+ggtitle('QQ plot')

anov <- anova(model)

treat.means <- emmeans(model, ~k)
em <- treat.means

# Perform Dunnett test
options(emmeans= list(emmeans = list(infer = c(TRUE, TRUE)),contrast = list(infer = c(TRUE, TRUE))))
con <- contrast(treat.means, adjust='bonf', method='dunnett', ref = 7,alternative="greater")


plot_comparaciones <- plot(con, comparisons = TRUE,)

resumen_modelo<-as.data.frame(em)
plot_EE <- ggplot(resumen_modelo, aes(x=k, y=emmean)) + #grafico las predicciones
  labs(x="k") + labs(y="coeficiente de silhouette") +
  geom_errorbar(aes(ymin=emmean-SE, ymax=emmean+SE),  width=0.2)+
  ggtitle("Coeficientes de silhouette para cada K", "Media ± error estándar de cada K a partir del modelo de comparación de medias")


ggsave(filename = paste0(base_path, "/scatter-", counter, ".png"), plot = scatter,width=5,height=5)

write.csv(lev, file = paste0(base_path, "/lev-", counter, ".csv"))

shap_df <- data.frame(W = shap[["statistic"]], pvalue = shap[["p.value"]])

write.csv(shap_df, file = paste0(base_path, "/shapiro-", counter, ".csv"), row.names = FALSE)

png(
  filename = paste0(base_path, "/boxplot_homocedasticidad-", counter, ".png"),width=1000,height=500
)
boxplot(r1~df_long$k,xlab="K clusters", ylab="Residuos estandarizados")
dev.off()

ggsave(filename = paste0(base_path, "/qqplot-", counter, ".png"), plot = qq,width=5,height=5)

ggsave(filename = paste0(base_path, "/residuos_vs_predichos-", counter, ".png"), width=5,height=5,plot = res_pred)

write.csv(anov, file = paste0(base_path, "/anova", "-", counter, ".csv"))

write.csv(con, file = paste0(base_path,"/dunnett", "-", counter, ".csv"))

ggsave(filename = paste0(base_path, "/dunnett_confint", "-", counter, ".png"), plot = plot_comparaciones,width=10,height=5)

# Extract residuals
residuals <- residuals(model, type = "pearson")

# Split residuals by levels of the independent variable k
split_resid <- split(residuals, df_long$k)

# Perform Shapiro-Wilk test for each level
shapiro_results <- lapply(split_resid, shapiro.test)

# Extract p-values from the test results
p_values <- sapply(shapiro_results, function(x) x$p.value)

# Identify which levels of the independent variable k have non-normal residuals
not_normal_levels <- names(p_values)[p_values < 0.05]

# Print out the levels of k with non-normal residuals
print('======== no normales')
print(not_normal_levels)

ggsave(filename = paste0(base_path, "/plot_EE-", counter, ".png"), plot = plot_EE,width=8,height=5)
