//! Training pipeline for the Linguae character-level language detector.
//!
//! This crate ports the former Python training script to Rust. It loads the
//! shared datasets under `crates/linguae-data/data`, cleans and splits them,
//! builds a character vocabulary, trains a CharCNN with Candle, evaluates it,
//! renders SVG plots with plotters and exports a safetensors artifact that
//! `linguae-core` and the Python server can load.

pub mod config;
pub mod convert;
pub mod data;
pub mod dataset;
pub mod error;
pub mod export;
pub mod metrics;
pub mod model;
pub mod plots;
pub mod rng;
pub mod source;
pub mod train;
pub mod verify;
pub mod vocab;

pub use error::{Error, Result};
