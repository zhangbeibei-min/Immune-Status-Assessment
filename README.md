
# Immune-Status-Assessment
Our approach involved the development of a three-platform model to standardize CBC data. We then employed an EM-GMM model to cluster participants based on their immune status. Utilizing RF, LightGBM, and XGboost algorithms, we assigned weights to each CBC index and constructed a comprehensive immune status evaluation model.Compared to previous research, our model is more scientific and feasible in terms of operation, providing a robust framework for quantitative assessment.

# Workflow
## ![image name](https://github.com/zhangbeibei-min/Immune-Status-Assessment/blob/main/Figure/WorkFlow.jpg)


# Installation
## **[link](https://github.com/zhangbeibei-min/Immune-Status-Assessment.git)**



#  Requirement
- Python 3.7.6
- sklearn 0.22.1
- numpy 1.21.6
- scipy 1.5.2
- pandas 1.0.1
- lightgbm 3.2.0
- xgboost 1.5.2

#  Note
1. Three_platform_model.py  : Standardize CBC data
2. Immune_status_cluster.py : Used for immune state clustering of CBC data
3. Immune_status_weight.py : Find the correlation coefficient
4. Immune_status_score.py : Score for immune status
5. Cubic_polynomial_fitting.py : To find the relationship between age and immune status score

The code runs in the order shown above.