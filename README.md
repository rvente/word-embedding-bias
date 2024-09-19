Learned Gender Bias in Wikipedia Word Embeddings persists throught time,
as measured by WEAT Scores derived from the first 330 MB 

Word embeddings exhibit desirable properties when converting natural language
to numerical vector representations. However, embeddings often internalize
associations that parrot stereotypes pertaining to  race, gender, and culture.
Researchers have attempted to mitigate word embedding bias by altering the
model after training, or changing the loss function, but to address bias
comprehensively, Brunet et al. (2018) turn to the data where these
biases originate. This work examines the differences in word embedding
representations learned of Wikipedia a decade apart, finding minimal
differences between them. This work finds that larger window sizes correlate
with higher WEAT scores and higher performance on analogy and similarity
datasets.
