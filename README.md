# Model Performance Comparison on Baby Product Dataset

This project explores the performance of two different models on a dataset related to baby products. The dataset is pre-processed and predictions are made by both models. The models' predictions are then compared using various performance metrics.

## Dataset

The dataset is a collection of baby products with various attributes like `name`, `description`, etc. The ground truth for the dataset is stored in `ground_truth_baby.csv`, and it was generated from the following sql query:
```
select 
	mmpc.finch_id, 
	mmpc .finch_cat_id , 
	mmpc .finch_cat_name, 
	pdwfi."name" , 
	pdwfi.description, 
	pdwfi.additionalproperties, 
	pdwfi.features,
	pdwfi.brand_name
from product_data.manually_matched_product_categories mmpc 
left join finch_id.product_data_with_finch_ids pdwfi 
on pdwfi .finch_id  = mmpc .finch_id 
inner join sources_of_truth.finch_categories fc 
on mmpc .finch_cat_id  = fc.finch_cat_id 
where fc.finch_cat_name_a ilike '%baby%'
limit 2000
```

This file contains the actual categories (labels) of the baby products, which are used to evaluate the performance of the models.

## Pre-processing

A function named `textify_data` is applied to the dataset after loading it, which transforms the dataset into a format suitable for analysis by preparing the text data. The dataset is then converted to a list of dictionaries, 
with each dictionary representing a record in the dataset, this format is necessary for subsequent prediction functions.

## Model Predictions

Two models are used in this study:

1. Model T (`t_baby`) The model trained on just the nice target data!
2. Model TA (`ta_baby`) The model trained on both the target and amazon corpus! 

For both models, a function `get_preds_wrapper` is used to generate predictions. These predictions are merged with the ground truth data based on the `finch_id` key.

## Filtering Predictions

Only predictions with a confidence level greater than 0.8 are considered for evaluation to ensure the comparison is based on high-confidence predictions.

Furthermore, the comparison is narrowed down to only those categories present in the original input dataset to maintain consistency.

## Performance Metrics

The performance metrics evaluated are:

- Accuracy
- Precision (Weighted)
- Recall (Weighted)
- F1 Score (Weighted)

These metrics are calculated for each model and visualized using horizontal bar charts. Precision, recall, and F1 score are calculated in a weighted manner which takes into account the imbalance in the distribution of actual categories.

## Visualization

The performance metrics for both models are visualized in a horizontal bar chart, allowing for an easy comparison of Model T and Model TA across different metrics.

In addition to performance metrics, confusion matrices for both models are generated and displayed as heatmaps, providing a detailed view of the models' performances in terms of true positives, false positives, true negatives, and false negatives.

![image](https://github.com/fincholiver/TA_T_categorization_model_comparison/assets/107002591/d09e2830-f040-4f3c-98a4-a2bab9a8679a)
![image](https://github.com/fincholiver/TA_T_categorization_model_comparison/assets/107002591/872ccb54-3793-415b-a1c0-0b0238441570)
![image](https://github.com/fincholiver/TA_T_categorization_model_comparison/assets/107002591/c439659e-ae60-4fb2-be91-a95f621d1c4c)
