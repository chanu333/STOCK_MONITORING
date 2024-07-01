extern crate reqwest;
extern crate serde;
extern crate serde_json;
extern crate ndarray;
extern crate linfa;
extern crate linfa_bayes;
extern crate plotters;

use reqwest::Error as ReqwestError;
use serde::Deserialize;
use ndarray::{Array1, Array2};
use linfa::{
    dataset::Dataset,
    prelude::*,
};
use linfa_bayes::GaussianNb;
use plotters::prelude::*;
use std::collections::HashSet;
use std::convert::From;

#[derive(Debug, Clone)]
struct StockData {
    symbol: String,
    price: f64,
    volume: u64,
    timestamp: String,
}

#[derive(Debug)]
enum CustomError {
    ReqwestError(reqwest::Error),
    ParseError(String),
    NotEnoughClasses,
    NaiveBayesError(linfa_bayes::NaiveBayesError), 
}

impl From<reqwest::Error> for CustomError {
    fn from(err: reqwest::Error) -> Self {
        CustomError::ReqwestError(err)
    }
}

impl From<serde_json::Error> for CustomError {
    fn from(err: serde_json::Error) -> Self {
        CustomError::ParseError(err.to_string())
    }
}

impl From<linfa_bayes::NaiveBayesError> for CustomError {
    fn from(err: linfa_bayes::NaiveBayesError) -> Self {
        CustomError::NaiveBayesError(err)
    }
}

/// Fetch stock data from Alpha Vantage API.
/// 
/// # Arguments
///
/// * `symbol` - The stock symbol to fetch data for.
/// * `api_key` - The API key for accessing Alpha Vantage.
///
/// # Returns
///
/// A Result containing a vector of StockData or a CustomError.
async fn fetch_stock_data(symbol: &str, api_key: &str) -> Result<Vec<StockData>, CustomError> {
    let url = format!("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={}&interval=1min&apikey={}", symbol, api_key);
    let response = reqwest::get(&url).await?.json::<serde_json::Value>().await?;
    let stock_data = parse_alpha_vantage_response(response)?;
    Ok(stock_data)
}

/// Parse Alpha Vantage API response into a vector of StockData structs.
/// 
/// # Arguments
///
/// * `response` - The JSON response from the Alpha Vantage API.
///
/// # Returns
///
/// A Result containing a vector of StockData or a CustomError.
fn parse_alpha_vantage_response(response: serde_json::Value) -> Result<Vec<StockData>, CustomError> {
    let time_series = response["Time Series (1min)"].as_object().ok_or(CustomError::ParseError("Invalid JSON format".into()))?;

    let mut stock_data = Vec::new();
    for (timestamp, data) in time_series {
        let stock = StockData {
            symbol: response["Meta Data"]["2. Symbol"].as_str().ok_or(CustomError::ParseError("Missing symbol".into()))?.to_string(),
            price: data["1. open"].as_str().ok_or(CustomError::ParseError("Missing price".into()))?.parse().unwrap_or(0.0), // Default to 0.0 on parse failure
            volume: data["5. volume"].as_str().ok_or(CustomError::ParseError("Missing volume".into()))?.parse().unwrap_or(0), // Default to 0 on parse failure
            timestamp: timestamp.to_string(),
        };
        stock_data.push(stock);
    }

    Ok(stock_data)
}

/// Preprocess stock data into feature and target arrays for model training.
/// 
/// # Arguments
///
/// * `data` - A vector of StockData.
///
/// # Returns
///
/// A tuple containing feature and target arrays.
fn preprocess_data(data: Vec<StockData>) -> (Array2<f64>, Array1<usize>) {
    let features: Vec<Vec<f64>> = data.iter().map(|stock| vec![stock.price, stock.volume as f64]).collect();
    let mut target: Vec<usize> = Vec::new();

    // Generate target values based on price increase (1) or decrease (0)
    for i in 0..data.len() - 1 {
        if data[i].price < data[i + 1].price {
            target.push(1); // Price increased
        } else {
            target.push(0); // Price stayed the same or decreased
        }
    }

    // Convert feature and target vectors to ndarray
    let features_array = Array2::from_shape_vec((features.len(), features[0].len()), features.into_iter().flatten().collect()).unwrap();
    let target_array = Array1::from(target);

    (features_array, target_array)
}

/// Train a Gaussian Naive Bayes model using the provided feature and target arrays.
/// 
/// # Arguments
///
/// * `features` - The feature array.
/// * `target` - The target array.
///
/// # Returns
///
/// A Result containing a trained GaussianNb model or a CustomError.
fn train_model(features: &Array2<f64>, target: &Array1<usize>) -> Result<GaussianNb<f64, usize>, CustomError> {
    let dataset = Dataset::new(features.clone(), target.clone());
    let model = GaussianNb::params().fit(&dataset)?;
    Ok(model)
}

/// Predict target values using the trained model.
/// 
/// # Arguments
///
/// * `model` - The trained GaussianNb model.
/// * `features` - The feature array for making predictions.
///
/// # Returns
///
/// An array of predicted target values.
fn predict(model: &GaussianNb<f64, usize>, features: &Array2<f64>) -> Array1<usize> {
    model.predict(features)
}

/// Calculate the accuracy of predictions.
/// 
/// # Arguments
///
/// * `predictions` - The array of predicted target values.
/// * `target` - The array of actual target values.
///
/// # Returns
///
/// The accuracy as a floating-point number.
fn calculate_accuracy(predictions: &Array1<usize>, target: &Array1<usize>) -> f64 {
    let correct_predictions = predictions.iter().zip(target.iter()).filter(|&(pred, actual)| pred == actual).count();
    let accuracy = correct_predictions as f64 / target.len() as f64;
    accuracy
}

/// Generate a plot of model accuracy using the plotters crate.
/// 
/// # Arguments
///
/// * `accuracy` - The calculated accuracy of the model.
fn plot_accuracy(accuracy: f64) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new("accuracy.png", (640, 480)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Model Accuracy", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..100, 0.0..1.0)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        vec![(0, accuracy), (100, accuracy)],
        &RED,
    ))?
    .label(format!("Accuracy: {:.2}%", accuracy * 100.0))
    .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    chart.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw()?;

    Ok(())
}

/// Generate a plot of stock price and volume over time.
/// 
/// # Arguments
///
/// * `data` - A vector of StockData to be plotted.
fn plot_stock_data(data: &[StockData]) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new("stock_data.png", (640, 480)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let max_price = data.iter().map(|d| d.price).fold(0./0., f64::max);
    let min_price = data.iter().map(|d| d.price).fold(0./0., f64::min);

    let (left, right) = root_area.split_horizontally(320);

    // Plotting stock price
    let mut chart = ChartBuilder::on(&left)
        .caption("Stock Price Over Time", ("sans-serif", 20).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..data.len(), min_price..max_price)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().enumerate().map(|(i, d)| (i, d.price)),
        &BLUE,
    ))?
    .label("Price")
    .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw()?;

    // Plotting stock volume
    let max_volume = data.iter().map(|d| d.volume).max().unwrap_or(1) as f64;

    let mut chart = ChartBuilder::on(&right)
        .caption("Stock Volume Over Time", ("sans-serif", 20).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..data.len(), 0.0..max_volume)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().enumerate().map(|(i, d)| (i, d.volume as f64)),
        &GREEN,
    ))?
    .label("Volume")
    .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &GREEN));

    chart.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw()?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), CustomError> {
    let api_key = "IE2BY8KFGEVSA6L6";
    let symbol = "IBM";

    let stock_data = fetch_stock_data(symbol, api_key).await?;
    println!("Fetched data: {:?}", stock_data);

    let (features, target) = preprocess_data(stock_data.clone());

    // Check if we have at least two distinct classes in the target data
    let distinct_classes: HashSet<_> = target.iter().collect();
    if distinct_classes.len() < 2 {
        return Err(CustomError::NotEnoughClasses);
    }

    let model = train_model(&features, &target)?;
    let predictions = predict(&model, &features);

    let accuracy = calculate_accuracy(&predictions, &target);
    println!("Naive Bayes Accuracy: {:.2}%", accuracy * 100.0);

    plot_accuracy(accuracy).expect("Failed to create accuracy plot");
    plot_stock_data(&stock_data).expect("Failed to create stock data plot");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test parsing Alpha Vantage API response
    #[test]
    fn test_parse_alpha_vantage_response() {
        let json_str = r#"
        {
            "Meta Data": {
                "2. Symbol": "IBM"
            },
            "Time Series (1min)": {
                "2023-03-10 16:00:00": {
                    "1. open": "123.45",
                    "5. volume": "1000"
                },
                "2023-03-10 16:01:00": {
                    "1. open": "123.50",
                    "5. volume": "1100"
                }
            }
        }
        "#;
        let response: serde_json::Value = serde_json::from_str(json_str).unwrap();
        let result = parse_alpha_vantage_response(response).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].symbol, "IBM");
    }

    /// Test data preprocessing
    #[test]
    fn test_preprocess_data() {
        let stock_data = vec![
            StockData { symbol: "IBM".to_string(), price: 123.45, volume: 1000, timestamp: "2023-03-10 16:00:00".to_string() },
            StockData { symbol: "IBM".to_string(), price: 123.50, volume: 1100, timestamp: "2023-03-10 16:01:00".to_string() }
        ];
        let (features, target) = preprocess_data(stock_data);
        assert_eq!(features.shape(), &[2, 2]);
        assert_eq!(target.len(), 1);
        assert_eq!(target[0], 1);
    }

    /// Test accuracy calculation
    #[test]
    fn test_calculate_accuracy() {
        let predictions = Array1::from(vec![1, 0, 1, 1]);
        let target = Array1::from(vec![1, 0, 0, 1]);
        let accuracy = calculate_accuracy(&predictions, &target);
        assert!((accuracy - 0.75).abs() < std::f64::EPSILON);
    }
}
