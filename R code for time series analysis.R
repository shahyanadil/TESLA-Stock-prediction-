rm(list = ls())
library(ggplot2)
library(dplyr)
library(reshape2)
# Modeling
library(tidymodels)
library(modeltime)

# Core
library(tidyverse)
library(timetk)
library(lubridate)
library(janitor)
library(tidyquant)
# Visualization
library(ggthemes)
library(ggsci)

# Tables 
library(gt)

library(keras)
library(tensorflow)


data_tesla <- read.csv('C:/Users/shahy/Desktop/R/Project/HistoricalData_TSLA.csv')
data_tesla['Symbol'] <- 'TSLA'
data_tesla <- data_tesla[,c(1,7,2,3,4,5,6)]
names(data_tesla) <-tolower(names(data_tesla))
#removing dollar sign
data_tesla$close.last <- gsub("\\$", "", data_tesla$close.last)
data_tesla$open <- gsub("\\$", "", data_tesla$open) 
data_tesla$high <- gsub("\\$", "", data_tesla$high)
data_tesla$low <- gsub("\\$", "", data_tesla$low)
#changing datatype and intrducing new column
colnames(data_tesla)[3] <- 'close'
data_tesla$close <- as.integer(data_tesla$close)
data_tesla$open <- as.integer(data_tesla$open) 
data_tesla$high <- as.integer(data_tesla$high)
data_tesla$low <- as.integer(data_tesla$low)
data_tesla$date <- as.Date(data_tesla$date , format = "%m/%d/%Y")
data_tesla <- data_tesla %>% mutate(difference = close - open)
data_tesla <- data_tesla %>% mutate(percentage_up_drop_open = (difference/open)*100)
#data_tesla <- data_tesla %>% mutate(percentage_up_drop_close = (difference/close)*100)


#View table 
start_date <- '2021'
end_date <- '2022'
filtered_data <- data_tesla %>% filter_by_time(.start_date = '2017', .end_date = '2022')
filtered_data %>% mutate(date = as.Date(date)) %>%
  head(10) %>% gt() %>%
  tab_header(
    title = "NYSE (TESLA Stock)",
    subtitle = glue::glue("{start_date} to {end_date}")
  ) %>% fmt_date(column = date, date_style = 3) %>% fmt_currency(columns = c(open, high, low, close, difference),
                                                                 currency = "USD") %>% fmt_number(columns = volume, suffixing =  TRUE)

#visualization
filtered_data %>% plot_time_series(date, close, .interactive=F,.smooth=F, .color_var=symbol,
                                   .line_size=1) + 
  theme_minimal() + 
  scale_y_continuous(labels = scales::dollar_format()) + 
  scale_color_tron()


filtered_data %>% plot_time_series(date, difference, .interactive=F,.smooth=F, .color_var=symbol,
                   .line_size=1) + 
  theme_minimal() + 
  scale_y_continuous(labels = scales::dollar_format()) + 
  scale_color_tron()

#transformation
trans_tbl <- filtered_data %>% 
  mutate(close = log_interval_vec(close, limit_lower = 0, offset = 1)) %>% 
  mutate(close = standardize_vec(close))


# Store the assign variables in order to make an ivnersion
# Just for demosntrations I will show how we get the mean and standard deviation
# Use to do the inversions at the end
inversion_tbl <- trans_tbl %>% 
  summarise(
    log_mean = mean(close),
    log_mean = as.numeric(format(log_mean, digits = 15)),
    sd_rev = sd(close),
    sd_rev = as.numeric(format(sd_rev, digits = 15))
  )


inv_mean <- inversion_tbl$log_mean
inv_sd <- inversion_tbl$sd_rev


# Our forecast horizon will be 60 days
forecast_h <- 60
lag_period <- 60
rolling_avg <- c(30,60,90,180)

complete_tbl <- trans_tbl %>% 
  bind_rows(
    future_frame(.data = ., .date_var = date, .length_out = forecast_h)
  ) %>%
  tk_augment_lags(close, .lags = lag_period) %>% 
  tk_augment_slidify(
    .value = close_lag60,
    .f = mean,
    .period = rolling_avg,
    .align = "center",
    .partial = TRUE
  ) %>% 
  rename_with(.cols = contains("lag"), .fn = ~str_c("lag_", .))


# Without the future frame (We will use this table for most of our tasks)
complete_prepared_tbl <- complete_tbl %>% 
  filter(!is.na(close))

forecast_tbl <- complete_tbl %>% 
  filter(is.na(close))


# Lag roll 60 has a better upward trend 
complete_prepared_tbl %>% 
  select(date, symbol, close, contains("lag")) %>% 
  pivot_longer(cols = close:lag_close_lag60_roll_180) %>%
  plot_time_series(date, value, name, .smooth=F, .interactive=F, .line_size=1) + 
  scale_color_locuszoom() + theme_dark() + theme(
    panel.background = element_rect(fill = "#2D2D2D"),
    legend.key = element_rect(fill = "#2D2D2D"),
    legend.position = "bottom") + 
  labs(
    title = "Lags Plot (Tesla Stock)",
    subtitle = "A glimpse of the full table",
    x = "Date",
    y = "Close Price", 
    caption = "Note: Lag 60 days seems to better catch the trend"
  )

#splitting the dataset
split_frame <- complete_prepared_tbl  %>% 
  time_series_split(
    .date_var = date,
    assess = "8 weeks",
    cumulative = TRUE
  )



split_frame %>% 
  tk_time_series_cv_plan() %>% 
  plot_time_series_cv_plan(date, close, .interactive=F, .line_size=1) + 
  scale_color_locuszoom() + theme_dark() + theme(
    panel.background = element_rect(fill = "#2D2D2D"),
    legend.key = element_rect(fill = "#2D2D2D"),
    legend.position = "bottom") + 
  labs(
    title = "Dataframe Split",
    caption = "We use this before refitting"
  )


# Prepare the base of the recipe

recipe_basic_specs <- recipe(close ~ date, data = training(split_frame)) %>% 
  step_timeseries_signature(date) %>% 
  step_fourier(date, period = c(30,60,90,180), K=1) %>% 
  step_rm(matches("(iso)|(xts)|(hour)|(minute)|(second)|(am.pm)|(date_quarter)")) %>% 
  step_normalize(matches("(index.num)|(yday)")) %>% 
  step_dummy(all_nominal(), one_hot = TRUE)


# Models to pick: Random Forest, Arima (Univariate), GLMNET

# Random Forest Model Multivariate
model_rf_fitted <- rand_forest() %>% 
  set_engine("ranger") %>% 
  set_mode("regression") 


# Workflow we apply to multivariates
workflow_rf_fitted <- workflow() %>% 
  add_recipe(recipe_basic_specs) %>% 
  add_model(model_rf_fitted) %>% 
  fit(training(split_frame))



# ARIMA Model (Univariate)



# ARIMA MultiVariate
model_arima_multi_fitted <- arima_reg() %>% 
  set_engine("auto_arima")

workflow_arima_multi_fitted <- workflow() %>% 
  add_recipe(recipe_basic_specs) %>% 
  add_model(model_rf_fitted) %>% 
  fit(training(split_frame))



# Bundle the models into a table
models_tbl <- modeltime_table(
  workflow_rf_fitted,
  workflow_arima_multi_fitted
) %>% 
  update_model_description(1, "RandomForest Model") %>% 
  update_model_description(3, "ARIMA MultiVariate")


calibration_tbl <- models_tbl %>% 
  modeltime_calibrate(
    new_data = testing(split_frame)
  ) 


calibration_tbl %>% 
  modeltime_accuracy(
    metric_set = default_forecast_accuracy_metric_set()
  ) %>% 
  table_modeltime_accuracy()


#visualization of the dataset
calibration_tbl %>% 
  modeltime_forecast(
    actual_data = complete_prepared_tbl,
    conf_interval = 0.95
  ) %>% 
  plot_modeltime_forecast(.interactive = F, .line_size=1, .conf_interval_fill = "white") +
  theme(
    panel.background = element_rect(fill = "#2D2D2D"),
    legend.key = element_rect(fill = "#2D2D2D"),
    legend.position = "bottom") + scale_colour_ucscgb() +
  labs(
    title = "TSLA Stock",
    subtitle = "Forecasting on the test set"
  ) + theme(legend.position = "bottom")


#refit the data
calibration_tbl %>% 
  # Trains on the whole training set
  # It says Updated on the training set because parameters have changed when refitting
  modeltime_refit(data = complete_prepared_tbl) %>% 
  # H depends on time column not good for models depending on external regressors
  modeltime_forecast(
    new_data = forecast_tbl,
    actual_data = complete_prepared_tbl
  ) %>% 
  plot_modeltime_forecast(.interactive = F, .line_size=1, 
                          .conf_interval_fill = "white",
                          .legend_max_width = 25) + 
  theme(
    panel.background = element_rect(fill = "#2D2D2D"),
    legend.key = element_rect(fill = "#2D2D2D"),
    legend.position = "bottom") + scale_color_tron() +
  labs(
    title = "Forecasting Forward",
    subtitle = "TSLA Stock"
  )

#retransform the data to its original form. and visualize it 
refit_tbl <- calibration_tbl %>% 
  modeltime_refit(data = complete_prepared_tbl) 

refit_tbl %>% 
  modeltime_forecast(
    new_data = forecast_tbl,
    actual_data = complete_prepared_tbl
  ) %>% 
  mutate(across(.value:.conf_hi, .fns = ~ standardize_inv_vec(
    x = ., 
    mean = 2.26601292196241, 
    sd = 0.531019665873533
  ))) %>% 
  mutate(across(.value:.conf_hi, .fns = ~log_interval_inv_vec(
    x = ., 
    limit_lower = 0,
    limit_upper = 852.1999875,
    offset = 1
  ))) %>% 
  plot_modeltime_forecast(.interactive = F, .line_size=1, .conf_interval_fill = "white") + 
  theme(
    panel.background = element_rect(fill = "#2D2D2D"),
    legend.key = element_rect(fill = "#2D2D2D"),
    legend.position = "bottom") + scale_color_tron() +
  labs(
    title = "Inverse Transformation Forecast",
    subtitle = "GOOGL Stock"
  ) + scale_y_continuous(labels = scales::dollar_format())

