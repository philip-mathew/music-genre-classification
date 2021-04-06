---
title: "Music_Genre_Classification"
author: "Philip Mathew"
date: "03/04/2021"
output: html_document
---

#```{r}
#loading libraries
library(dplyr)
library(ggplot2)
library(stringr)
library(tidyr)
library(h2o)
library(psych)

#loading dataframes
df_genres <- read.csv("features_3_sec.csv")

#scaling the data
df_genres_scaled <- data.frame(scale(df_genres[3:59]))

```


#```{r}
#PCA 
fa.parallel(df_genres_scaled, fa = "pc")

#Parallel analysis suggests that the number of factors =  NA  and the number of components =  10 
```


#```{r}
#PCA with varimax rotate 
df_genres_pca <- principal(df_genres_scaled, nfactors = 10, rotate = "varimax")

#PCA Results
df_genres_pca
```

#```{r}
#PCA scores to transformed data 
df_genres_transformed <- as.data.frame(df_genres_pca$scores)
head(df_genres_transformed)

```

#```{r}
#adding genre label
df_genres_transformed$label <- df_genres$label 
head(df_genres_transformed)

df_genres_reg <- data.frame(scale(df_genres[3:59]))
df_genres_reg$label <- df_genres$label 

head(df_genres_reg)

```

#```{r}
#Splitting test, train, 70/30 split - transformed
df_genres_size_tf <- floor(0.70*nrow(df_genres_transformed))
set.seed(1977)
df_genres_train_tf <- sample(seq_len(nrow(df_genres_transformed)), size = df_genres_size_tf)
df_genres_train_set_tf <- df_genres_transformed[df_genres_train_tf, ]
df_genres_val_set_tf <- df_genres_transformed[-df_genres_train_tf, ]

#Splitting test, train, 70/30 split - regular
df_genres_size_reg <- floor(0.70*nrow(df_genres_reg))
set.seed(1977)
df_genres_train_reg <- sample(seq_len(nrow(df_genres_reg)), size = df_genres_size_reg)
df_genres_train_set_reg <- df_genres_reg[df_genres_train_reg, ]
df_genres_val_set_reg <- df_genres_reg[-df_genres_train_reg, ]

print(paste0("Regular: Rows of train data: " , nrow(df_genres_train_set_reg)," & Rows of Validation data: ", nrow(df_genres_val_set_reg)))
print(paste0("Transformed: Rows of train data: " , nrow(df_genres_train_set_reg)," & Rows of Validation data: ", nrow(df_genres_val_set_reg)))
```

#```{r}
#Using H20 for modelling - Multi-Class Classification

#Algorithms to try:

#Decision Trees.
#Naive Bayes.
#Random Forest.
#Gradient Boosting.

h2o.init()

#h2o frames - transformed
genres_train_tf.hex <- as.h2o(x = df_genres_train_set_tf, destination_frame = "genres_train_tf.hex")
genres_test_tf.hex <- as.h2o(x = df_genres_val_set_tf, destination_frame = "genres_test_tf.hex")

#h2o frames - regular
genres_train_reg.hex <- as.h2o(x = df_genres_train_set_reg, destination_frame = "genres_train_reg.hex")
genres_test_reg.hex <- as.h2o(x = df_genres_val_set_reg, destination_frame = "genres_test_reg.hex")

```


#```{r}
#distributed random forest - transformed
genres_drf_tf <- h2o.randomForest(y = "label", training_frame = genres_train_tf.hex, nfolds = 5, seed = 1234)

#Summary
genres_drf_tf
```

#```{r}
#confustion matrix
h2o.confusionMatrix(genres_drf_tf)

```

#```{r}
#variable importance
h2o.varimp_plot(genres_drf_tf)

```
#```{r}
#performance test - transformed
perf_drf_tf <- h2o.performance(genres_drf_tf,newdata = genres_test_tf.hex)

h2o.confusionMatrix(perf_drf_tf)

#23.06% error
```


#```{r}
#naive bayes classifier - transformed
genres_nbs_tf <- h2o.naiveBayes(y = "label", training_frame = genres_train_tf.hex, nfolds = 10, seed = 1234)

genres_nbs_tf
```

#```{r}
#performance-transformed
perf_nbs_tf <- h2o.performance(genres_nbs_tf,newdata = genres_test_tf.hex)

h2o.confusionMatrix(perf_nbs_tf)

#54% error with naive bayes
```


#```{r}
#xgboost - tf
genres_xgboost_tf <- h2o.xgboost(y = "label", training_frame = genres_train_tf.hex, nfolds = 10)

genres_xgboost_tf
```
#```{r}
#performance xgboost_tf
perf_xgboost_tf <- h2o.performance(genres_xgboost_tf, newdata = genres_test_tf.hex)

h2o.confusionMatrix(perf_xgboost_tf)
#25% error
```

#```{r}
#GLM - multiclassification 
genres_glm_tf <- h2o.glm(y = "label", training_frame = genres_train_tf.hex, family='multinomial',solver='L_BFGS', lambda = 1e-2)

genres_glm_tf
```

#```{r}
#performance glm - tf
perf_glm_tf <- h2o.performance(genres_glm_tf,newdata = genres_test_tf.hex)

h2o.confusionMatrix(perf_glm_tf)
#49.78% error
```

#```{r}
#distributed random forest - regular
genres_drf_reg <- h2o.randomForest(y = "label", training_frame = genres_train_reg.hex, nfolds = 10, seed = 1234)

#Summary
genres_drf_reg
```
#```{r}
#variable importance
h2o.varimp_plot(genres_drf_reg)

```

#```{r}
#performance drf - reg
perf_drf_reg <- h2o.performance(genres_drf_reg,newdata = genres_test_reg.hex)

h2o.confusionMatrix(perf_drf_reg)
#14.65% error - Best so far! 
```
#```{r}
#xgboost - reg
genres_xgboost_reg <- h2o.xgboost(y = "label", training_frame = genres_train_reg.hex)
genres_xgboost_reg
```

#```{r}
#variable importance
h2o.varimp_plot(genres_xgboost_reg)

```

#```{r}
#performance drf - reg
perf_xgboost_reg <- h2o.performance(genres_xgboost_reg,newdata = genres_test_reg.hex)

h2o.confusionMatrix(perf_xgboost_reg)
#13.01% error - Best so far! - 87% accuracy 
```

#```{r}
#glm-reg
genres_glm_reg <- h2o.glm(y = "label", training_frame = genres_train_reg.hex, family='multinomial',solver='L_BFGS', lambda = 1e-4)

genres_glm_reg
```
#```{r}
#variable importance
h2o.varimp_plot(genres_glm_reg)

```

#```{r}
#performance glm - reg
perf_glm_reg <- h2o.performance(genres_glm_reg,newdata = genres_test_reg.hex)

h2o.confusionMatrix(perf_glm_reg)
#28.76% error
```

#```{r}
#naive bayes classifier - regular
genres_nbs_reg <- h2o.naiveBayes(y = "label", training_frame = genres_train_reg.hex, nfolds = 10)

genres_nbs_reg
#only 52% accuracy
```

#```{r}
#Deep learning - Reg
genres_dl_reg <- h2o.deeplearning(y = "label", training_frame = genres_train_reg.hex, epochs = 1, nfolds = 5)
genres_dl_reg
```

#```{r}
#variable importance
h2o.varimp_plot(genres_dl_reg)
```
#```{r}
#performance deep learning - reg
perf_dl_reg <- h2o.performance(genres_dl_reg,newdata = genres_test_reg.hex)

h2o.confusionMatrix(genres_dl_reg)
#26.91% error

```

#```{r}
#Deep learning - Reg-2
genres_dl_reg_m2 <- h2o.deeplearning(y = "label", training_frame = genres_train_reg.hex, nfolds = 10)
genres_dl_reg_m2
```

#```{r}
#plotting
plot(genres_dl_reg_m2)
```

#```{r}
#performance deep learning - reg
perf_dl_reg_m2 <- h2o.performance(genres_dl_reg_m2,newdata = genres_test_reg.hex)

h2o.confusionMatrix(perf_dl_reg_m2)

# 13.85% error 
```

#```{r}
#Deep learning - Reg-3
genres_dl_reg_m3 <- h2o.deeplearning(y = "label", training_frame = genres_train_reg.hex, nfolds = 10, epochs = 30)

genres_dl_reg_m3

```
#```{r}
#plotting
plot(genres_dl_reg_m3)

```

#```{r}
#performance deep learning - reg
perf_dl_reg_m3 <- h2o.performance(genres_dl_reg_m3,newdata = genres_test_reg.hex)

h2o.confusionMatrix(perf_dl_reg_m3)
#Error of 11% <- Best so far! 
```

