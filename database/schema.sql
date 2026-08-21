-- MySQL Schema for Life Cost Platform

CREATE DATABASE IF NOT EXISTS lifecost;
USE lifecost;

-- Rent Listings Table
CREATE TABLE IF NOT EXISTS rent_listings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    locality VARCHAR(255) NOT NULL,
    bhk INT NOT NULL,
    bathrooms INT NOT NULL,
    sqft FLOAT NOT NULL,
    city VARCHAR(100) NOT NULL,
    rent FLOAT NOT NULL,
    INDEX (locality),
    INDEX (city)
);

-- Metro Routes / Fares Table
CREATE TABLE IF NOT EXISTS metro_routes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    route_name VARCHAR(255) NOT NULL,
    distance_km FLOAT NOT NULL,
    num_stops INT NOT NULL,
    travel_min FLOAT NOT NULL,
    fare_inr INT NOT NULL
);

-- Grocery Items Table
CREATE TABLE IF NOT EXISTS groceries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    category VARCHAR(100) NOT NULL,
    name VARCHAR(255) NOT NULL,
    quantity_raw VARCHAR(100),
    quantity_norm_g FLOAT,
    price FLOAT NOT NULL,
    price_per_100g FLOAT
);
