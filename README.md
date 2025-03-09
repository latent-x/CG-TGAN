# CG-TGAN: Conditional Generative Adversarial Networks with Graph Neural Networks for Tabular Data Synthesizing
This code is the implementation of "CG-TGAN: Conditional Generative Adversarial Networks with Graph Neural Networks for Tabular Data Synthesizing".

The paper has been accepted by AAAI 2025.

## Requirements
* requirement.txt
* We experimented with T4 and L4 GPUs in the Google Colab environment. 

## Usage
* experiments.ipynb

    You can run the code by modifying 2nd cell(Define Dataset Information).

    Simply modify where each column belongs.

    (For Classification Dataset)

    * problem_type = 'Classification'
    * classification_col = [target_col]
    * classifiers_for_utility = ['lr', 'dt', 'rf']

    (For Regression Dataset)

    * problem_type = 'Regression'
    * regression_col = [target_col]
    * classifiers_for_utility = ['l_reg', 'lasso', 'B_ridge']

## References
Our work and code refer two existing works.
* CTGAN official implementation: https://github.com/sdv-dev/CTGAN
* CTAB-GAN official implementation: https://github.com/Team-TUD/CTAB-GAN
