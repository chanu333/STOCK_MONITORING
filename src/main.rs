extern crate reqwest;
extern crate serde;
extern crate serde_json;
extern crate ndarray;
extern crate linfa;
extern crate linfa_bayes;

use reqwest::Error as ReqwestError;
use serde::Deserialize;
use ndarray::{Array1, Array2};
use linfa::{
    dataset::Dataset,
    prelude::*,
};
use linfa_bayes::GaussianNb;
use std::collections::HashSet;
use std::convert::From; 

#[derive(Debug)]
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

async fn fetch_stock_data(symbol: &str, api_key: &str) -> Result<Vec<StockData>, CustomError> {
    let url = format!("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={}&interval=1min&apikey={}", symbol, api_key);
    let response = reqwest::get(&url).await?.json::<serde_json::Value>().await?;
    let stock_data = parse_alpha_vantage_response(response)?;
    Ok(stock_data)
}

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

fn preprocess_data(data: Vec<StockData>) -> (Array2<f64>, Array1<usize>) {
    let features: Vec<Vec<f64>> = data.iter().map(|stock| vec![stock.price, stock.volume as f64]).collect();
    let mut target: Vec<usize> = Vec::new();

    for i in 0..data.len() - 1 {
        if data[i].price < data[i + 1].price {
            target.push(1); // Price increased
        } else {
            target.push(0); // Price stayed the same or decreased
        }
    }

    let features_array = Array2::from_shape_vec((features.len(), features[0].len()), features.into_iter().flatten().collect()).unwrap();
    let target_array = Array1::from(target);

    (features_array, target_array)
}

fn train_model(features: &Array2<f64>, target: &Array1<usize>) -> Result<GaussianNb<f64, usize>, CustomError> {
    let dataset = Dataset::new(features.clone(), target.clone());
    let model = GaussianNb::params().fit(&dataset)?;
    Ok(model)
}

fn predict(model: &GaussianNb<f64, usize>, features: &Array2<f64>) -> Array1<usize> {
    model.predict(features)
}

fn calculate_accuracy(predictions: &Array1<usize>, target: &Array1<usize>) -> f64 {
    let correct_predictions = predictions.iter().zip(target.iter()).filter(|&(pred, actual)| pred == actual).count();
    let accuracy = correct_predictions as f64 / target.len() as f64;
    accuracy
}

#[tokio::main]
async fn main() -> Result<(), CustomError> {
    let api_key = "IE2BY8KFGEVSA6L6";
    let symbol = "IBM";

    let stock_data = fetch_stock_data(symbol, api_key).await?;
    println!("Fetched data: {:?}", stock_data);

    let (features, target) = preprocess_data(stock_data);

    // Check if we have at least two distinct classes in the target data
    let distinct_classes: HashSet<_> = target.iter().collect();
    if distinct_classes.len() < 2 {
        return Err(CustomError::NotEnoughClasses);
    }

    let model = train_model(&features, &target)?;
    let predictions = predict(&model, &features);

    let accuracy = calculate_accuracy(&predictions, &target);
    println!("Naive Bayes Accuracy: {:.2}%", accuracy * 100.0);

    Ok(())
}
