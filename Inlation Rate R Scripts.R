library(readxl)
library(forecast)
library(keras)
library(Metrics)
library(tidyverse)
library(kernlab)

infrate2018 <- read_excel("D:/Data for Research/Inflation Rate/infrate2018.xls")
inflation <- ts(infrate2018$`Inflation Rate`, frequency = 12, start = c(1958,1), end = c(2024,1))

ts.plot(inflation, xlab = "Year", ylab = "Inflation Rate", lwd = 2)

summary(infrate2018$`Inflation Rate`)
sd(infrate2018$`Inflation Rate`)

####################
#                  #
#  Seasonal ARIMA  #
#                  #
####################

#Data splitting 
train_ARIMA <- window(inflation, end = c(2010,10))
length(train_ARIMA)
test_ARIMA <- window(inflation, start = c(2010,11), end = c(2024,1))
length(test_ARIMA)

tseries::adf.test(train_ARIMA) 
ARIMA_Inflation <- auto.arima(train_ARIMA, seasonal = TRUE, stationary = TRUE, ic = "aic", trace = TRUE)
lmtest::coeftest(ARIMA_Inflation)
checkresiduals(ARIMA_Inflation)
Box.test(ARIMA_Inflation$residuals, type = "Lj")
forecast::accuracy(ARIMA_Inflation)

rmse(ARIMA_Inflation$x,ARIMA_Inflation$fitted)
mae(ARIMA_Inflation$x,ARIMA_Inflation$fitted)
mape(ARIMA_Inflation$fitted,ARIMA_Inflation$x)*100


ARIMA_valid <- Arima(c(train_ARIMA,test_ARIMA), model = ARIMA_Inflation)
length(ARIMA_valid$fitted[635:793])
forecast(ARIMA_valid, h = 23)

#Plot on Train Set
ts.plot(train_ARIMA, ARIMA_valid$fitted[1:634], col = c('blue','red'), lwd = 2)
legend("topleft", col = c('blue','red'), legend = c('Actual','SARIMA'), lwd = 2, inset=0.025)

#Plot on Test Set
ts.plot(test_ARIMA, ARIMA_valid$fitted[635:793], col = c('blue','red'), lwd = 2)
legend("topleft", col = c('blue','red'), legend = c('Actual','SARIMA'), lwd = 2, inset=0.025)

forecast::accuracy(test_ARIMA,ARIMA_valid$fitted[635:793])

############################
#                          #
#  Long Short-Term Memory  #
#                          #
############################

#Data Preprocessing
data <- infrate2018$`Inflation Rate`

#Splitting the data into training and validation sets
train_data <- data[1:634]
valid_data <- data[635:793]

 Scaling the data
max_value <- max(train_data)
min_value <- min(train_data)
scale_data <- function(data) {
  scaled_data <- (data - min_value) / (max_value - min_value)
  return(as.data.frame(scaled_data))  # Ensure the result is a data frame
}
inverse_scale_data <- function(data) data * (max_value - min_value) + min_value

train_scaled <- scale_data(train_data)
valid_scaled <- scale_data(valid_data)

#Preparing the data for the deep learning models
create_dataset <- function(data, look_back = 1) {
  x_data <- list()
  y_data <- list()
  
  for (i in 1:(nrow(data) - look_back)) {
    x_data[[i]] <- data[i:(i + look_back - 1), ]
    y_data[[i]] <- data[i + look_back, ]
  }
  
  x_array <- array(unlist(x_data), dim = c(length(x_data), look_back, ncol(data)))
  y_array <- array(unlist(y_data), dim = c(length(y_data)))
  
  return(list(x = x_array, y = y_array))
}


look_back <- 1
train_dataset <- create_dataset(train_scaled, look_back)
valid_dataset <- create_dataset(valid_scaled, look_back)

#Fine Tuning#
fit_lstm_model <- function(units, dropout_rate, epochs, batch_size, learning_rate, train_dataset, valid_dataset) {
  model <- keras_model_sequential() %>%
    layer_lstm(units = units, input_shape = c(look_back, 1)) %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer_adam(learning_rate = learning_rate), 
    metrics = c('mean_absolute_error')
  )
  
  history <- model %>% fit(
    x = train_dataset$x, y = train_dataset$y,
    epochs = epochs,
    batch_size = batch_size,
    validation_data = list(valid_dataset$x, valid_dataset$y),
    verbose = 0 
  )
  
  list(model = model, val_loss = min(history$metrics$val_loss))
}


units_options <- c(50, 100)
dropout_rate_options <- c(0.2, 0.3)
epochs_options <- c(50, 100)
batch_size_options <- c(1, 32)
learning_rate_options <- c(0.001, 0.01)

best_model <- NULL
best_val_loss <- Inf
best_params <- list()

for (units in units_options) {
  for (dropout_rate in dropout_rate_options) {
    for (epochs in epochs_options) {
      for (batch_size in batch_size_options) {
        for (learning_rate in learning_rate_options) {
          cat(sprintf("Testing units=%d, dropout_rate=%f, epochs=%d, batch_size=%d, lr=%f\n",
                      units, dropout_rate, epochs, batch_size, learning_rate))
          
          result <- fit_lstm_model(units, dropout_rate, epochs, batch_size, learning_rate, train_dataset, valid_dataset)
          
          if (result$val_loss < best_val_loss) {
            best_val_loss <- result$val_loss
            best_model <- result$model
            best_params <- list(units=units, dropout_rate=dropout_rate, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
          }
        }
      }
    }
  }
}

cat("Best parameters:\n")
print(best_params)
cat(sprintf("Best validation loss: %f\n", best_val_loss))


#End of Fine Tuning#


#Defining the LSTM model
model_lstm <- keras_model_sequential() %>%
  layer_lstm(units = 100, input_shape = c(look_back, 1)) %>% #Assuming 1 feature per time step
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1)

model_lstm %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(learning_rate = 0.01),
  metrics = c('mean_absolute_error')
)

#Training the model
history_lstm <- model_lstm %>% fit(
  x = train_dataset$x, y = train_dataset$y,
  epochs = 100,
  batch_size = 32,
  validation_data = list(valid_dataset$x, valid_dataset$y)
)

#Making predictions
train_predict_lstm <- model_lstm %>% predict(train_dataset$x)
valid_predict_lstm <- model_lstm %>% predict(valid_dataset$x)

#Inverse scaling for plotting and evaluation
train_predict_lstm <- inverse_scale_data(train_predict_lstm)
train_predict_lstm <- ts(train_predict_lstm, start = c(1958,2), end = c(2010,10), frequency = 12)
train_y_lstm <- inverse_scale_data(train_dataset$y)
train_y_lstm <- ts(train_y_lstm, start = c(1958,2), end = c(2010,10), frequency = 12)


valid_predict_lstm <- inverse_scale_data(valid_predict_lstm)
valid_predict_lstm <- ts(valid_predict_lstm, start = c(2010,12), end = c(2024,1), frequency = 12)
valid_y_lstm <- inverse_scale_data(valid_dataset$y)
valid_y_lstm <- ts(valid_y_lstm, start = c(2010,12), end = c(2024,1), frequency = 12)

#Plotting results
plot(train_y_lstm, type = 'l', col = 'blue', ylab = 'Inflation Rate', lwd = 2)
lines(train_predict_lstm, col = 'red', lwd = 2)
legend('topleft', lty=c(1,1), col = c('blue','red'), legend = c('Actual','LSTM'), cex=0.7,inset=0.025, lwd = 2)

plot(valid_y_lstm, type = 'l', col = 'blue', ylab = 'Inflation Rate', lwd = 2)
lines(valid_predict_lstm, col = 'red', lwd = 2)
legend('topleft', lty=c(1,1), col = c('blue','red'), legend = c('Actual','Fitted'), cex=0.7,inset=0.025, lwd = 2)

rmse(train_predict_lstm,train_y_lstm)
mae(train_predict_lstm,train_y_lstm)
mape(train_predict_lstm,train_y_lstm)*100

rmse(valid_predict_lstm,valid_y_lstm)
mae(valid_predict_lstm,valid_y_lstm)
mape(valid_predict_lstm,valid_y_lstm)*100


#################################
#                               #
#  Gaussian Process Regression  #
#                               #
#################################

#Process the data
infrate2018$Date <- as.Date(paste(infrate2018$Year, infrate2018$Month, "01", sep="-"), "%Y-%B-%d")
data_gpr <- infrate2018 %>% select(Date, 'Inflation Rate')

#Normalize the inflation rates
max_rate_gpr <- max(data_gpr$`Inflation Rate`, na.rm = TRUE)
min_rate_gpr <- min(data_gpr$`Inflation Rate`, na.rm = TRUE)
infrate2018$`Inflation Rate` <- (infrate2018$`Inflation Rate` - min_rate_gpr) / (max_rate_gpr - min_rate_gpr)

date_seq_gpr <- seq(from = min(infrate2018$Date), to = max(infrate2018$Date), by = "month")
X_gpr <- as.matrix(date_seq_gpr) #Feature matrix
y_gpr <- as.matrix(infrate2018$`Inflation Rate`) #Target variable

#Splitting the data into training and test sets
set.seed(123) #for reproducibility
train_size_gpr <- floor(0.8 * length(y_gpr))
X_train_gpr <- X_gpr[1:train_size_gpr, , drop = FALSE]
y_train_gpr <- y_gpr[1:train_size_gpr]
X_test_gpr <- X_gpr[(train_size_gpr + 1):length(y_gpr), , drop = FALSE]
y_test_gpr <- y_gpr[(train_size_gpr + 1):length(y_gpr)]

#Define the Gaussian process with a Radial Basis Function kernel
gp_model <- gausspr(X_train_gpr, y_train_gpr, kernel = "rbfdot")

#Make predictions on the training set
predictions_train_gpr <- predict(gp_model, X_train_gpr)

#Make predictions on the test set
predictions_test_gpr <- predict(gp_model, X_test_gpr)

#Inverse the normalization for actual comparison
predictions_train_gpr <- (predictions_train_gpr * (max_rate_gpr - min_rate_gpr)) + min_rate_gpr
predictions_test_gpr <- (predictions_test_gpr * (max_rate_gpr - min_rate_gpr)) + min_rate_gpr
y_train_gpr <- (y_train_gpr * (max_rate_gpr - min_rate_gpr)) + min_rate_gpr
y_test_gpr <- (y_test_gpr * (max_rate_gpr - min_rate_gpr)) + min_rate_gpr

rmse(predictions_test_gpr,y_test_gpr)
mae(predictions_test_gpr,y_test_gpr)
mape(predictions_test_gpr,y_test_gpr)*100

#Plot of the trainset
plot(ts(y_train_gpr, frequency = 12, start = c(1958,1), end = c(2010,10)), type = 'l', col = 'blue', ylab = 'Inflation Rate', lwd = 2)
lines(ts(predictions_train_gpr, frequency = 12, start = c(1958,1), end = c(2010,10)), col = 'red', lwd = 2)
legend('topleft', lty=c(1,1), col = c('blue','red'), legend = c('Actual','GPR'), cex=0.7,inset=0.025, lwd = 2)

#Plotting the results
plot(infrate2018$Date, infrate2018$`Inflation Rate` * (max_rate_gpr - min_rate_gpr) + min_rate_gpr, type = 'l', col = 'blue', ylab = 'Inflation Rate', main = 'GPR Predictions vs Actual', lwd = 2)
lines(X_train_gpr, predictions_train_gpr, col = 'red', lwd = 2)
lines(X_test_gpr, predictions_test_gpr, col = 'green', lwd = 2)
legend("topright", legend=c("Actual", "Train Predictions", "Test Predictions"), col=c("blue", "red", "green"), lty=1, cex=0.7,inset=0.025, lwd = 2)


#################
#  Plot of all  #
#################

plot(test_ARIMA, col = 'black', type = 'l', ylab = 'Inflation Rate', lwd = 2, ylim = c(-1,10))
lines(ts(ARIMA_valid$fitted[635:793], frequency = 12, start = c(2010,11, end = c(2024,1))), col = 'blue', lwd = 2)
lines(valid_predict_lstm, col = 'red', lwd = 2)
lines(ts(predictions_test_gpr, frequency = 12, start = c(2010,11), end = c(2024,1)), col = 'green', lwd = 2)
legend("topleft", col = c('black','blue','red','green'), legend = c('Actual','SARIMA','LSTM','GPR'), lwd = 2, inset=0.025, cex = 0.7)
