# LLM Sentiment Analysis Project

## Project Task
The task for this project is sentiment analysis. The goal is to classify text data into positive or negative sentiment categories using a pre-trained language model.

## Dataset
The dataset used for this project is the IMDb dataset, which contains 50,000 movie reviews labeled as positive or negative. This dataset is widely used for sentiment analysis tasks and provides a balanced set of reviews for training and evaluation.

## Pre-trained Model
The pre-trained model selected for this project is `bert-base-uncased` from Hugging Face. BERT (Bidirectional Encoder Representations from Transformers) is chosen because of its strong performance on various NLP tasks, including sentiment analysis. The `bert-base-uncased` model is a smaller version of BERT that is efficient and effective for this task.

## Performance Metrics
The performance of the new model is evaluated using the following metrics:
- **Accuracy**: Measures the percentage of correct predictions.
- **F1 Score**: The mean of precision and recall, providing a balance between the two.
- **Precision**: Measures the proportion of true positive predictions out of all positive predictions.
- **Recall**: Measures the proportion of true positive predictions out of all actual positives.

### Evaluation Results

| Metric      | Value  |
|-------------|--------|
| Accuracy    | 0.92   |
| F1 Score    | 0.93   |
| Precision   | 0.90   |
| Recall      | 0.92   |

### Interpretation

- The model achieved an accuracy of 92%, indicating that it correctly classified 92% of the validation samples.
- The F1 score of 91% shows a good balance between precision and recall.
- Precision and recall values are also high, suggesting that the model performs well in identifying both positive and negative sentiments.

## Hyperparameters
The following hyperparameters are considered most important and relevant while optimizing the model:
- **Batch Size**: The number of samples processed before the model is updated. A smaller batch size is used to manage memory usage.
- **Learning Rate**: The step size at each iteration while moving toward a minimum of the loss function.
- **Number of Epochs**: The number of complete passes through the training dataset. The training only needed 5 epochs before no loss was detected.
 [4111/4690 1:20:29 < 11:20, 0.85 it/s, Epoch 8.76/10]
Epoch	Training Loss	Validation Loss
1	0.159600	0.228323
2	0.187400	0.245628
3	0.063000	0.341017
4	0.051500	0.446141
5	0.003300	0.531508
6	0.000100	0.602241
7	0.000100	0.666269
8	0.000100	0.691672


## Relevant Links
- **Model on Hugging Face**: [Link to the model](https://huggingface.co/blairjdaniel/blairjdaniel_LLM_model)
- **Dataset**: [IMDb Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

