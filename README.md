# mutation_tests_deep_learning

This repository is used to explore the mutation tests and the deep learning. I will explore how to use neural networks to predict the mutation score given the existing mutations and its context.

In addition, I will train the RNN model and the transformer model to generate the mutations

# Pipeline
The basic architecture is that the context of the code, such as how many if statements, how many for loops and what statements wanted to inject the mutations would be provided, and these requirements will be converted to the valid inputs. Then I will use the transformer model to generate the mutations, and the new mutation operators and other context information provided will be fed into the neural network model, and the mutation score will be predicted. 

