# Configuration Parameters and Options

Schematic below shows the workflow of wattile layered with configuration groups described below.

![alt text](workflow_configs_group.png)


### data input

- `data_dir`: *str*

    directory containing the data config and csv
    
- `data_config`: *TBD*

    TBD
    
- `start_time`: *TBD*

    TBD
    
- `end_time`: *TBD*

    TBD
    
- `predictor_columns`: *List[str]*

    pre-defined predictor variables
    
- `target_var`: *str*

    column name and target variable in the input data that model will predict
    
    
### data processing

- `feat_time`: *dict*

    TBD

  - `month_of_year`: *list[str] ("sincos")*

      month of year methodology
      
  - `day_of_week`: *list[str] ("binary_reg", "binary_fuzzy")*

      day of week methodology

  - `hour_of_day`: *list[str] ("sincos", "binary_reg", "binary_fuzzy")*

      hour of day methodology

  - `holidays`: *boolean*

      indicator of whether holidays are taken into consideration in the modeling
      
- `resample`: *dict*

    TBD

  - `bin_interval`: *pandas timedelta*
  
      TBD
      
  - `bin_closed`: *str ("left", "right")*
  
      TBD
      
  - `bin_label`: *str ("left", "right")*
  
      TBD
      
- `feat_stats`: *dict*

    TBD

  - `active`: *boolean*
  
      specify whether or not to use rolling window statistics
      
  - `window_width`: *pandas timedelta*
  
      specifies the number of minutes to use for the window.
      
- `feat_timelag`: *dict*

    TBD

  - `lag_interval`: *pandas timedelta*
  
      TBD
  
  - `lag_count`: *int*
  
      TBD
    
- `input_output_window`: *dict*

    TBD

  - `window_width_source`: *pandas timedelta*
  
      TBD
      
  - `window_width_futurecast`: *pandas timedelta*

      TBD
      
  - `window_width_target`: *pandas timedelta*
  
      TBD
      
- `random_seed`: *int*

    random seed to group data into sequential chunks and also the seed number to fix the randomness in torch package
    
- `sequential_splicer`: *dict*

    group datasets together into sequential chunks just for data split

    - `active`: *boolean*

        defines whether splicer is applied

    - `window_width`: *pandas timedelta*

        defines the window size of splicer

- `data_split`: *str ("x:y:z" where x + y + z = 100)*

    training, validation, and testing data ratio, respectively
    
- `train_size_factor`: *int*

    ensure to pick a training set size that we can then later split into mini batches that have some desired number of samples in each batch. Purely for computational efficiency


### learning algorithm

- `arch_type`: *str ("RNN")*

    TBD
    
- `arch_type_variant`: *str ("vanilla" or "lstm")*

    RNN architecture type
    
- `use_case`: *str ("train", "prediction", "validation")*

    train - use case for training a model

    validation - use case for validating an existing (previously trained) model

    prediction - use case for applying data on a trained model for deployment purpose
    
- `arch_version`: *int ("alfa", "bravo", or "charlie")*

    model version
 
- `num_epochs`: *int*

    number of epochs 
    
- `hidden_size`: *int*

    number of hidden nodes  
   
- `num_layers`: *int*

    number of layer dimension   
    
- `quantiles`: *list[floats]* (floats must be 0-1)

    quantile list
    
- `transformation_method`: *str ("minmaxscale" or "standard")*

    data normalization methods
    
- `train_batch_size`: *int*

    size of batch in the training data. It is used to calculate number of batches in the training data

- `val_batch_size`: *int*

    size of batch in the validation data. It is used to calculate number of batches in the validation data

- `train_val_split`: *str*

    method to split training and validation data including random

- `run_resume`: *boolean*

    indicator to resume from a previous training session

- `weight_decay`: *float*

    parameter for optimizer

- `lr_config`: *dict*

    Learning rate configuration

    - `base`: *float*
    - `schedule`: *boolean*
    - `type`: *str*
    - `factor`: *float*
    - `min`: *float*
    - `patience`: *int*
    - `step_size`: *int*

- `smoothing_alpha`: *float*

    smoothing alpha for pinball loss and quantile loss function

- `eval_frequency`: *int*

    frequency (every n iterations) to save the model

- `test_method`: *str ("external", "internal")*

    defines the source of testing data, including internal (using training data) or external (using h5 file) 


### data output

- `exp_dir`: *str*

    Directory  where model output
    
- `plot_comparison`: *TBD*

    TBD
    
- `plot_comparison_portion_start`: *TBD*

    TBD
    
- `plot_comparison_portion_end`: *TBD*

    TBD
