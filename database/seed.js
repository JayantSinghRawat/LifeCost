// database/seed.js
const fs = require('fs');
const path = require('path');
const mysql = require('mysql2/promise');
require('dotenv').config();

const RENT_PATH = path.join(__dirname, '..', 'Scrapping_Manual', 'renting', 'listings.json');
const METRO_PATH = path.join(__dirname, '..', 'Scrapping_Manual', 'Metro', 'bhopal_metro_fares.json');
const GROCERY_PATH = path.join(__dirname, '..', 'Scrapping_Manual', 'grocery', 'Bhopal_blinkit_results_462010.json');

async function seed() {
  console.log('Starting Database Seeding...');
  
  const connectionOpts = {
    host: process.env.DB_HOST || '127.0.0.1',
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || '',
  };

  let connection;
  try {
    connection = await mysql.createConnection(connectionOpts);
    console.log('Connected to MySQL server.');
  } catch (err) {
    console.error('Failed to connect to MySQL server. Ensure MySQL is running and credentials are correct.');
    console.error('Error:', err.message);
    process.exit(1);
  }

  try {
    // Create database and tables
    await connection.query('CREATE DATABASE IF NOT EXISTS lifecost');
    await connection.query('USE lifecost');
    console.log('Database lifecost initialized.');

    // Drop tables if they exist to refresh seed
    await connection.query('DROP TABLE IF EXISTS rent_listings');
    await connection.query('DROP TABLE IF EXISTS metro_routes');
    await connection.query('DROP TABLE IF EXISTS groceries');

    await connection.query(`
      CREATE TABLE rent_listings (
        id INT AUTO_INCREMENT PRIMARY KEY,
        locality VARCHAR(255) NOT NULL,
        bhk INT NOT NULL,
        bathrooms INT NOT NULL,
        sqft FLOAT NOT NULL,
        city VARCHAR(100) NOT NULL,
        rent FLOAT NOT NULL,
        INDEX (locality),
        INDEX (city)
      )
    `);

    await connection.query(`
      CREATE TABLE metro_routes (
        id INT AUTO_INCREMENT PRIMARY KEY,
        route_name VARCHAR(255) NOT NULL,
        distance_km FLOAT NOT NULL,
        num_stops INT NOT NULL,
        travel_min FLOAT NOT NULL,
        fare_inr INT NOT NULL
      )
    `);

    await connection.query(`
      CREATE TABLE groceries (
        id INT AUTO_INCREMENT PRIMARY KEY,
        category VARCHAR(100) NOT NULL,
        name VARCHAR(255) NOT NULL,
        quantity_raw VARCHAR(100),
        quantity_norm_g FLOAT,
        price FLOAT NOT NULL,
        price_per_100g FLOAT
      )
    `);
    console.log('Tables created successfully.');

    // 1. Seed Rent Listings
    if (fs.existsSync(RENT_PATH)) {
      const rawRent = JSON.parse(fs.readFileSync(RENT_PATH, 'utf8'));
      console.log(`Loaded ${rawRent.length} raw rent listings.`);

      const rentValues = [];
      for (const item of rawRent) {
        const details = String(item.details || '');
        const priceStr = String(item.price || '');
        const locationStr = String(item.location || '');

        // Extract BHK
        const bhkMatch = details.match(/(\d+)\s*(BHK|RK|BRK)/i);
        const bhk = bhkMatch ? parseInt(bhkMatch[1], 10) : null;
        if (!bhk || bhk < 1 || bhk > 6) continue;

        // Extract Bathrooms
        const bathMatch = details.match(/(\d+)\s*Bathroom/i);
        const bathrooms = bathMatch ? parseInt(bathMatch[1], 10) : bhk;

        // Extract Sqft
        const sqftMatch = details.match(/(\d[\d,]*)\s*sqft/i);
        const sqft = sqftMatch ? parseFloat(sqftMatch[1].replace(/,/g, '')) : null;
        if (!sqft || sqft < 100 || sqft > 20000) continue;

        // Extract Rent Price
        const priceMatch = priceStr.replace(/\s/g, '').match(/\d[\d,]*/);
        const rent = priceMatch ? parseFloat(priceMatch[0].replace(/,/g, '')) : null;
        if (!rent || rent < 1000 || rent > 500000) continue;

        // Extract Locality & City
        const parts = locationStr.split(',').map(p => p.trim().toUpperCase());
        const city = parts[parts.length - 1] || 'BHOPAL';
        const locality = parts[0] || 'OTHER';

        rentValues.push([locality, bhk, bathrooms, sqft, city, rent]);
      }

      if (rentValues.length > 0) {
        await connection.query(
          'INSERT INTO rent_listings (locality, bhk, bathrooms, sqft, city, rent) VALUES ?',
          [rentValues]
        );
        console.log(`Seeded ${rentValues.length} rent listings.`);
      }
    } else {
      console.warn('listings.json not found, skipping rent listings seeding.');
    }

    // 2. Seed Metro Routes
    if (fs.existsSync(METRO_PATH)) {
      const rawMetro = JSON.parse(fs.readFileSync(METRO_PATH, 'utf8'));
      console.log(`Loaded ${rawMetro.length} raw metro routes.`);

      const metroValues = [];
      for (const item of rawMetro) {
        const from = item.From || '';
        const to = item.To || '';
        const routeName = `${from} - ${to}`;

        const distanceMatch = (item.Distance || '').match(/([\d.]+)/);
        const distance = distanceMatch ? parseFloat(distanceMatch[1]) : 0;

        const stopsMatch = (item.Stations || '').match(/(\d+)/);
        const stops = stopsMatch ? parseInt(stopsMatch[1], 10) : 0;

        const timeMatch = (item['Est. Time'] || '').match(/(\d+)/);
        const time = timeMatch ? parseFloat(timeMatch[1]) : 0;

        const fareMatch = (item['Est. Fare'] || '').match(/(\d+)/);
        const fare = fareMatch ? parseInt(fareMatch[1], 10) : 10;

        metroValues.push([routeName, distance, stops, time, fare]);
      }

      if (metroValues.length > 0) {
        await connection.query(
          'INSERT INTO metro_routes (route_name, distance_km, num_stops, travel_min, fare_inr) VALUES ?',
          [metroValues]
        );
        console.log(`Seeded ${metroValues.length} metro routes.`);
      }
    } else {
      console.warn('bhopal_metro_fares.json not found, skipping metro seeding.');
    }

    // 3. Seed Grocery Items
    if (fs.existsSync(GROCERY_PATH)) {
      const rawGrocery = JSON.parse(fs.readFileSync(GROCERY_PATH, 'utf8'));
      console.log(`Loaded ${rawGrocery.length} raw grocery items.`);

      const groceryValues = [];
      for (const item of rawGrocery) {
        const name = item.name || '';
        const category = item.term || 'Groceries';
        const priceStr = item.price || '';
        if (!priceStr.includes('₹')) continue;

        const priceMatch = priceStr.replace(/,/g, '').match(/[\d.]+/);
        if (!priceMatch) continue;
        const price = parseFloat(priceMatch[0]);

        const qtyStr = (item.quantity || '').toLowerCase();
        let qtyNorm = null;
        const qtyMatch = qtyStr.match(/([\d.]+)\s*(kg|g|ltr|l|ml|pcs?|dozen|nos?)/i);
        if (qtyMatch) {
          let val = parseFloat(qtyMatch[1]);
          const unit = qtyMatch[2].toLowerCase();
          if (unit === 'kg' || unit === 'ltr' || unit === 'l') {
            val *= 1000;
          } else if (['pcs', 'pc', 'nos', 'no', 'dozen'].includes(unit)) {
            val *= 100; // fictional 100g weight logic matching Python script
          }
          qtyNorm = val;
        }

        const pricePer100 = (qtyNorm && qtyNorm > 0) ? (price / qtyNorm * 100) : null;

        groceryValues.push([category, name, qtyStr, qtyNorm, price, pricePer100]);
      }

      if (groceryValues.length > 0) {
        await connection.query(
          'INSERT INTO groceries (category, name, quantity_raw, quantity_norm_g, price, price_per_100g) VALUES ?',
          [groceryValues]
        );
        console.log(`Seeded ${groceryValues.length} grocery items.`);
      }
    } else {
      console.warn('Bhopal_blinkit_results_462010.json not found, skipping grocery seeding.');
    }

    console.log('Database seeding completed successfully!');
  } catch (err) {
    console.error('Error during database seeding:', err.message);
  } finally {
    await connection.end();
  }
}

if (require.main === module) {
  seed();
}
