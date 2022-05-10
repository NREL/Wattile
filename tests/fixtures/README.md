# Config Options
- `building`: str
- `target_var`: str
- `start_year`: int
- `start_month`: int
- `start_day`: int
- `end_year`: int
- `end_month`: int
- `end_day`: int
- `data_time_interval_mins`: int
- `predictor_columns`: List[str]
- `arch_version`: int
- `exp_id`: str
- `arch_type`: str
- `arch_type_variant`: str
- `transformation_method`: str
- `train_batch_size`: int
- `val_batch_size`: int
- `convert_csvs`: boolean
- `exp_dir`: str
- `data_dir`: str
- `resample_freq`: int
- `sequence_freq_min`: int
- `splicer`: dict
    - `active`: boolean
    - `time`: str
- `rolling_window`: dict
    - `active`: boolean

         specify whether or not to use rolling window statistics
    - `type`: Literal["binned", "rolling"]

        **binned** - This method creates min, max, and mean features for each original feature, computed by calculating the statistic over that last N minutes, separated into 
        stationary bins. This has the same effect as downsampling the data to a lower frequency.
        
        **rolling** - This method creates min, max, and mean features for each original feature, computed by calculating the statistic over that last N minutes in a rolling fashion. The time frequency of the original data is preserved.

    - `mintues`: int

        Specifies the number of minutes to use for the window. For type binned, this is the size of the downsampling. This should be higher than configs["resample_freq"], since the rolling windows are calculated after this step. For type rolling, this is the size of the rolling window.
- `window`: int
- `EC_future_gap_min`: int
- `DOW`: list[str]
- `MOY`: list[str]
- `HOD`: list[str]
- `Holidays`: boolean
- `S2S_stagger`: dict
    - `initial_num`: int
    - `decay`: int
    - `secondary_num`: int
- `train_size_factor`: int
- `train_val_split`: str
- `data_split`: str
- `random_seed`: int
- `qs`: list[floats]
- `use_case`: str
- `run_resume`: boolean
- `num_epochs`: int
- `weight_decay`: float
- `hidden_nodes`: int
- `layer_dim`: int
- `eval_frequency`: int
- `lr_config`: dict
    - `base`: float
    - `schedule`: boolean
    - `type`: str
    - `factor`: float
    - `min`: float
    - `patience`: int
    - `step_size`: int
- `smoothing_alpha`: float
- `test_method`: str
